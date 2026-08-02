"""The Detector interface and its uniform result record.

Every defense — whether it inspects weights, training data, or runtime inputs —
returns a :class:`DetectionResult`. Forcing one record across all three families
is what makes them comparable at all.

Detectors receive only what their threat model allows. The harness never passes
the Trigger object, the poison mask, or the manifest row; ground truth stays
with the scorer. See :mod:`deadbolt.zoo` for how blindness is enforced.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

#: What a detector needs access to. Determines which zoo artefacts it is handed.
#:
#: ``model_clean``
#:     The checkpoint plus a small trusted clean set. Neural Cleanse and every
#:     trigger-reconstruction descendant.
#: ``trainset``
#:     The suspect training set, poison and all. Latent-statistics methods.
#: ``runtime``
#:     A stream of inference inputs, some of which may carry the trigger.
#:     Input filters.
#: ``weights_only``
#:     The checkpoint and nothing else. Meta-classifiers.
Access = Literal["model_clean", "trainset", "runtime", "weights_only"]


def median(v: Tensor) -> Tensor:
    """Median with the averaging convention, i.e. what ``np.median`` computes.

    ``torch.median`` returns the *lower* of the two middle values for
    even-length inputs. That is a defensible convention and the wrong one here:
    Neural Cleanse's anomaly index is defined against the averaged median, and
    with C=10 labels the gap between the two conventions is large enough to
    change a verdict. Measured on a BadNets MNIST model, the lower-median form
    reports an anomaly index of 1.35 (not flagged) where the averaged form
    reports 2.07 (flagged) — on identical reconstructions.
    """
    return v.quantile(0.5)


def anomaly_index(values: Tensor | Sequence[float], side: str = "low") -> tuple[float, int]:
    """Median-absolute-deviation outlier score, as Neural Cleanse defines it.

    Returns ``(index, argextreme)``. An index above 2 corresponds to roughly a
    95% confidence that the extreme value does not belong to the same
    distribution as the rest, under a normality assumption that is doing a lot
    of work with only ten labels.

    Args:
        side: ``"low"`` scores the *smallest* value as the anomaly, which is
            what trigger reconstruction wants — a backdoored label is the one
            needing an unusually *small* mask. ``"high"`` scores the largest.

    MAD rather than standard deviation because with C=10 samples, one true
    outlier moves the mean and inflates the SD enough to hide itself. That
    self-masking is the entire reason the original paper uses MAD, and swapping
    in a z-score is a silent way to make every defense look worse.
    """
    v = torch.as_tensor(values, dtype=torch.float64).flatten()
    mid = median(v)
    # 1.4826 rescales MAD to be a consistent estimator of sigma for a normal.
    mad = median((v - mid).abs()) * 1.4826
    if mad <= 0:
        # Every label reconstructed identically: nothing is an outlier. Returning
        # 0 rather than infinity matters — degenerate cases must not be reported
        # as maximally suspicious.
        return 0.0, int(v.argmin() if side == "low" else v.argmax())
    if side == "low":
        idx = int(v.argmin())
        return float((mid - v[idx]) / mad), idx
    idx = int(v.argmax())
    return float((v[idx] - mid) / mad), idx


@dataclass
class DetectionResult:
    """Uniform output of every detector.

    Attributes:
        is_backdoored: Verdict at the method's *own published* threshold.
            Kept so we can report each paper's method as its authors intended.
        score: Continuous suspicion score, higher = more suspicious. This is
            the field the benchmark actually ranks on: it lets us plot ROC
            curves instead of trusting thresholds that were tuned on the same
            data that produced the original paper's numbers.
        target_label: Which class the detector believes is the backdoor target,
            if the method identifies one. Scored separately — flagging a model
            for the wrong reason is a weaker result than flagging it correctly.
        recovered_mask: Reconstructed trigger support, ``(1, H, W)``, for
            methods that invert a trigger. Scored by IoU against the attack's
            ground-truth mask when one exists.
        per_sample_scores: Per-example suspicion for data- and input-level
            methods, aligned with the dataset passed to :meth:`Detector.scan`.
        runtime_s: Wall-clock seconds. Reported in the main results table, not
            a footnote: a defense costing 40 min/model is a different product
            from one costing 8 s, and papers rarely make that legible.
        aux_scores: Additional continuous statistics the method computed on the
            way to ``score``, each of which the harness plots its own ROC for.

            This exists because of a specific failure deadbolt found and would
            otherwise have had to report as "Neural Cleanse does not work".
            On MNIST, Neural Cleanse reconstructs the BadNets trigger almost
            exactly — an L1 norm of 7 against a true 9-pixel patch, versus 29
            for the smallest clean-model reconstruction. The *signal* is
            excellent. Its published statistic then throws it away, because a
            MAD outlier test over ten labels is confounded when the clean
            distribution has an intrinsically easy label of its own. Separating
            "what the method measured" from "the threshold rule the paper wrapped
            around it" is the difference between a useful finding and a
            misleading one, and no single ``score`` field can express both.
        extra: Method-specific diagnostics (per-label L1 norms, silhouette
            scores, lambda schedules) kept for debugging and plots.
    """

    is_backdoored: bool
    score: float
    target_label: int | None = None
    recovered_mask: Tensor | None = None
    per_sample_scores: Tensor | None = None
    runtime_s: float = 0.0
    aux_scores: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """Flatten to a JSONL-safe row. Tensors are summarised, not serialised."""
        return {
            "is_backdoored": bool(self.is_backdoored),
            "score": float(self.score),
            "target_label": self.target_label,
            "has_mask": self.recovered_mask is not None,
            "n_per_sample": (
                int(self.per_sample_scores.numel())
                if self.per_sample_scores is not None
                else 0
            ),
            "runtime_s": round(float(self.runtime_s), 4),
            "aux_scores": {k: float(v) for k, v in self.aux_scores.items()},
            "extra": self.extra,
        }


class Detector(ABC):
    """Base class for backdoor detectors.

    Subclasses declare their :attr:`access` level, which the harness uses to
    decide what to hand them. A detector that quietly reaches for data outside
    its declared threat model is cheating, and the benchmark's central claim —
    that these methods are comparable — would be void.
    """

    #: Name used in manifests and result tables.
    name: str = "detector"

    #: Threat model. Governs which artefacts scan() receives.
    access: Access = "model_clean"

    #: Whether the method reports a suspected target class.
    identifies_target: bool = False

    #: Whether the method reconstructs a trigger, i.e. whether mask IoU is a
    #: meaningful thing to ask of it.
    recovers_mask: bool = False

    #: Whether ``score`` is a claim about the *model*. STRIP sets this False:
    #: it screens individual inputs and, run on a stream that happens to
    #: contain no triggered inputs, says nothing at all about the weights. The
    #: harness omits it from the model-level ROC rather than inventing a
    #: number, and scores it on per-input separation instead — which is the
    #: comparison STRIP is actually claiming to win.
    produces_model_verdict: bool = True

    #: Whether ``per_sample_scores`` is a claim about which *inputs* are
    #: poisoned. True for data-side and input-side methods; the harness scores
    #: those with a per-sample AUC against the hidden poison mask.
    produces_sample_verdict: bool = False

    @abstractmethod
    def scan(
        self,
        model: nn.Module,
        data: DataLoader,
        *,
        num_classes: int,
        device: torch.device,
    ) -> DetectionResult:
        """Inspect a model and return a verdict.

        Args:
            model: The victim network, already in eval mode on ``device``.
            data: Loader whose contents depend on :attr:`access` — a small
                clean set for ``model_clean``, the suspect training set for
                ``trainset``, the inputs to screen for ``runtime``.
            num_classes: Class count of the victim task.
            device: Compute device.

        Returns:
            A populated :class:`DetectionResult`. Implementations should wrap
            their work in :meth:`timer` so ``runtime_s`` is always recorded.
        """

    @contextmanager
    def timer(self, result_holder: dict[str, float]) -> Iterator[None]:
        """Record wall-clock into ``result_holder['runtime_s']``.

        Timing is mandatory rather than optional because cost is one of the
        headline comparisons this benchmark exists to make.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            result_holder["runtime_s"] = time.perf_counter() - start

    def config(self) -> dict[str, Any]:
        """Serialisable hyperparameters, recorded alongside each scan result."""
        return {"name": self.name, "access": self.access}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(access={self.access!r})"
