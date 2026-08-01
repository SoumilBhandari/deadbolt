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
from typing import Any, Iterator, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

#: What a detector needs access to. Determines which zoo artefacts it is handed.
Access = Literal["model_clean", "trainset", "runtime", "weights_only"]


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
        extra: Method-specific diagnostics (per-label L1 norms, silhouette
            scores, lambda schedules) kept for debugging and plots.
    """

    is_backdoored: bool
    score: float
    target_label: int | None = None
    recovered_mask: Tensor | None = None
    per_sample_scores: Tensor | None = None
    runtime_s: float = 0.0
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
