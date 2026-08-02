"""STRIP — screening inputs by perturbation sensitivity (Gao et al., 2019).

STRIP does not look at weights at all. It takes an incoming input, superimposes
a handful of trusted clean images onto it, and measures the entropy of the
resulting predictions. A clean input, once blended with an unrelated image,
becomes ambiguous — predictions scatter, entropy is high. A triggered input
does not: the trigger dominates whatever it is mixed with, every blend lands on
the target class, and entropy collapses. Low entropy means "this input is
carrying a trigger".

**What STRIP is not.** It is an input filter, and it makes no claim about the
weights. Run it on a model whose attacker happens to be idle and it reports
nothing, because there is nothing there to report — the model is just as
backdoored as it was a minute ago. The benchmark therefore does not put STRIP
on the model-level ROC curve; it is scored on how well its per-input scores
separate triggered inputs from clean ones, which is the comparison it is
actually claiming to win. A model-level score is still recorded, but it
measures "is the attacker currently active", not "is this model backdoored",
and reading it as the latter is the most common way STRIP gets over-credited.

**Where it breaks.** The premise is that the trigger dominates a superimposed
image. WaNet's trigger is a warp, and warping an already-blended image does not
dominate anything — entropy behaves normally and STRIP falls to chance. Noise
mode makes this worse on purpose: the network has been explicitly taught that
perturbed-and-warped does not mean the target class.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deadbolt.defenses.base import DetectionResult, Detector

Superimpose = Literal["add", "blend"]


class STRIP(Detector):
    """Entropy-based input screening.

    Args:
        n_overlays: Clean images superimposed per input. The paper uses 100;
            the entropy estimate is a mean over these, so cost and variance
            trade directly against each other.
        mode: ``"add"`` sums the two images and clips, which is what the paper
            does and what keeps a saturated patch trigger saturated.
            ``"blend"`` averages them, which halves the trigger's amplitude and
            makes STRIP look considerably worse — a difference worth being able
            to reproduce deliberately rather than by accident.
        fpr: False-rejection rate the detection threshold is calibrated to on
            clean inputs. 0.01 is the paper's headline operating point.
    """

    name = "strip"
    access = "runtime"
    produces_model_verdict = False
    produces_sample_verdict = True

    def __init__(
        self,
        n_overlays: int = 32,
        mode: Superimpose = "add",
        fpr: float = 0.01,
        batch_size: int = 256,
    ) -> None:
        self.n_overlays = n_overlays
        self.mode = mode
        self.fpr = fpr
        self.batch_size = batch_size

    def _superimpose(self, x: Tensor, overlay: Tensor) -> Tensor:
        if self.mode == "add":
            return (x + overlay).clamp(0, 1)
        if self.mode == "blend":
            return (x + overlay) / 2
        raise ValueError(f"unknown mode {self.mode!r}")

    @torch.no_grad()
    def _entropy(self, model: nn.Module, x: Tensor, overlays: Tensor) -> Tensor:
        """Mean prediction entropy of ``x`` under each overlay, ``(B,)`` on CPU.

        Accumulated in float32 on-device and only widened once it lands on the
        CPU: MPS has no float64, so a float64 accumulator here would confine
        the whole method to CPU for no precision that matters — entropies are
        O(1) and the mean is over a few dozen terms.
        """
        total = torch.zeros(x.shape[0], device=x.device)
        for i in range(overlays.shape[0]):
            logits = model(self._superimpose(x, overlays[i : i + 1]))
            logp = torch.log_softmax(logits.float(), dim=1)
            # -sum p log p computed from log-probabilities: taking the log of a
            # softmax output directly underflows to -inf on confident
            # predictions, which is exactly the regime STRIP cares about.
            total += -(logp.exp() * logp).sum(1)
        return (total / overlays.shape[0]).cpu().double()

    def scan(
        self,
        model: nn.Module,
        data: DataLoader,
        *,
        num_classes: int,
        device: torch.device,
        clean: DataLoader | None = None,
    ) -> DetectionResult:
        """Screen every input in ``data``.

        Args:
            clean: Trusted images used both as overlays and to calibrate the
                threshold. Required — STRIP's threat model assumes it.
        """
        if clean is None:
            raise ValueError("STRIP requires a trusted clean loader for overlays")
        model.eval()
        holder: dict[str, float] = {}

        with self.timer(holder):
            pool = torch.cat([x for x, _ in clean]).to(device)
            g = torch.Generator(device="cpu").manual_seed(0)
            pick = torch.randperm(pool.shape[0], generator=g)[: self.n_overlays]
            overlays = pool[pick.to(pool.device)]

            # Calibrate on held-out clean inputs. The threshold must come from
            # data the screened stream is not part of, or the false-rejection
            # rate is fitted to the very inputs it claims to bound.
            calib = self._entropy(model, pool, overlays)
            threshold = torch.quantile(calib, self.fpr)

            entropies = []
            for x, _ in data:
                entropies.append(self._entropy(model, x.to(device), overlays))
            entropy = torch.cat(entropies)

        # Higher score = more suspicious, so negate: low entropy is the alarm.
        scores = -entropy.float()
        flagged = float((entropy < threshold).float().mean())

        return DetectionResult(
            # Not a model-level verdict; see the module docstring. Recorded so
            # the number exists, excluded from the model ROC by
            # produces_model_verdict = False.
            is_backdoored=flagged > 0.5,
            score=float(-entropy.mean()),
            per_sample_scores=scores,
            runtime_s=holder["runtime_s"],
            extra={
                "threshold": float(threshold),
                "flagged_fraction": round(flagged, 4),
                "mean_entropy": round(float(entropy.mean()), 4),
                "clean_mean_entropy": round(float(calib.mean()), 4),
            },
        )

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "n_overlays": self.n_overlays,
            "mode": self.mode,
            "fpr": self.fpr,
        }
