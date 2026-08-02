"""Neural Cleanse — trigger reconstruction by optimisation (Wang et al., 2019).

The idea is elegant. If a model has a backdoor targeting label ``t``, then
there exists a small perturbation that drives *any* input to ``t``. For every
other label, no such small perturbation exists — you would have to overwrite
the image. So: for each label in turn, solve for the smallest patch that
converts the whole clean set to that label, and then ask whether one label's
answer is anomalously smaller than the rest.

.. code::

    minimise_{m, p}   CE( f( (1-m) * x + m * p ), t )  +  lambda * ||m||_1

The two details that people get wrong, both of which quietly break it:

**The lambda schedule.** ``lambda`` trades off "does the trigger work" against
"is the trigger small", and no fixed value works: too small and the mask sprawls
across the image for every label, flattening the L1 norms that the outlier test
depends on; too large and the optimiser cannot find a working trigger at all,
for the backdoored label either. Neural Cleanse handles this with a controller
that raises lambda while the reconstructed trigger keeps working and drops it
when it stops. Implementations that hardcode lambda report that Neural Cleanse
fails on BadNets — which is a bug report against the implementation.

**MAD, not standard deviation.** With ten labels, one genuine outlier moves the
mean and inflates the standard deviation enough to hide itself. See
:func:`deadbolt.defenses.base.anomaly_index`.

What this cannot do is a property of the hypothesis space, not the optimiser:
the search is over ``(1-m) * x + m * p``, and WaNet's warp is not expressible
in that form for any mask and any pattern. The optimiser will still return
something for every label, the norms will look unremarkable, and the anomaly
index will sit below threshold. Confirming that failure is one of deadbolt's
correctness checks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deadbolt.defenses.base import DetectionResult, Detector, anomaly_index, median
from deadbolt.defenses.reconstruct import TriggerOptimizer, batch_stream


class NeuralCleanse(Detector):
    """Per-label trigger reconstruction with a MAD outlier test.

    Args:
        steps: Optimisation steps per label. Cost is linear in this and in the
            class count, which is why the runtime column matters: at 43 classes
            (GTSRB) this is the most expensive defense in the repo by an order
            of magnitude.
        lr: Adam learning rate on the mask and pattern parameters.
        init_cost: Starting ``lambda``.
        cost_multiplier: Factor the controller multiplies ``lambda`` by.
        patience: Consecutive windows of consistent behaviour before the
            controller acts. Prevents it chasing minibatch noise.
        window: Steps per controller decision.
        attack_threshold: Reconstruction is deemed to "work" when this fraction
            of the clean set is driven to the candidate label.
        flag_threshold: Anomaly index above which the model is called
            backdoored. 2 is the published value.
        mask_from_norm: Binarisation threshold, as a fraction of the mask's max,
            used when reporting a mask for IoU scoring. The optimised mask is
            continuous; IoU is not defined without a cut.
        early_stop: Controller windows without an improved reconstruction
            before this label is considered done.
        min_improvement: Relative shrinkage required to count as an improvement.
    """

    name = "neural_cleanse"
    access = "model_clean"
    identifies_target = True
    recovers_mask = True

    def __init__(
        self,
        steps: int = 400,
        lr: float = 0.1,
        init_cost: float = 1e-3,
        cost_multiplier: float = 2.0,
        patience: int = 5,
        window: int = 10,
        attack_threshold: float = 0.99,
        flag_threshold: float = 2.0,
        mask_from_norm: float = 0.5,
        early_stop: int = 25,
        min_improvement: float = 0.01,
    ) -> None:
        self.steps = steps
        self.early_stop = early_stop
        self.min_improvement = min_improvement
        self.lr = lr
        self.init_cost = init_cost
        self.cost_multiplier = cost_multiplier
        self.patience = patience
        self.window = window
        self.attack_threshold = attack_threshold
        self.flag_threshold = flag_threshold
        self.mask_from_norm = mask_from_norm

    def _reconstruct(
        self,
        model: nn.Module,
        batches: list[tuple[Tensor, Tensor]],
        label: int,
        device: torch.device,
    ) -> tuple[float, Tensor, Tensor, float]:
        """Recover the smallest working trigger for one label.

        Delegates to the shared :class:`TriggerOptimizer` and simply spends the
        whole per-label budget on it, stopping once the reconstruction stops
        shrinking. K-Arm uses the same optimiser and differs only in how it
        distributes budget across labels, which is the entire claim it makes —
        and is only measurable because the optimiser underneath is identical.
        """
        arm = TriggerOptimizer(
            model,
            label,
            tuple(batches[0][0].shape[1:]),
            device,
            lr=self.lr,
            init_cost=self.init_cost,
            cost_multiplier=self.cost_multiplier,
            patience=self.patience,
            window=self.window,
            attack_threshold=self.attack_threshold,
            min_improvement=self.min_improvement,
        )
        stream = batch_stream(batches)
        while arm.steps_taken < self.steps:
            arm.advance(stream, min(self.window, self.steps - arm.steps_taken))
            # Stop once the reconstruction has stopped getting smaller. Beyond
            # that point the controller keeps raising lambda against a mask that
            # cannot shrink further, and the norms start to wander — which
            # perturbs a MAD test computed over only C values far more than it
            # perturbs the reconstruction. Longer is not safer here.
            if arm.stale_windows >= self.early_stop:
                break
        return arm.result()

    def scan(
        self,
        model: nn.Module,
        data: DataLoader,
        *,
        num_classes: int,
        device: torch.device,
    ) -> DetectionResult:
        holder: dict[str, float] = {}
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Materialise once: the same clean images are reused for every label,
        # and re-running the loader C times is pure overhead that would land in
        # the runtime column and misrepresent the method's cost.
        batches = [(x.to(device), y.to(device)) for x, y in data]

        norms: list[float] = []
        masks: list[Tensor] = []
        rates: list[float] = []
        with self.timer(holder):
            for label in range(num_classes):
                norm, mask, _pattern, rate = self._reconstruct(model, batches, label, device)
                norms.append(norm)
                masks.append(mask)
                rates.append(rate)

        index, target = anomaly_index(norms, side="low")
        mask = masks[target]
        binary = (mask >= self.mask_from_norm * mask.max()).float().squeeze(0).cpu()

        # The reconstruction itself, separated from the paper's threshold rule.
        # On Tier A these two disagree sharply and the disagreement is the
        # finding; see DetectionResult.aux_scores.
        t = torch.tensor(norms, dtype=torch.float64)
        med = float(median(t))
        smallest = float(t.min())
        aux = {
            # Scale-free: how much smaller the easiest label is than a typical
            # one. Unlike the anomaly index this does not care how many *other*
            # labels happen to be easy, which is exactly the confound that
            # breaks MAD at C=10.
            "norm_ratio": (med / smallest) if smallest > 0 else 0.0,
            "min_l1": -smallest,  # negated: higher must always mean more suspicious
        }

        return DetectionResult(
            is_backdoored=index > self.flag_threshold,
            score=index,
            target_label=target,
            recovered_mask=binary,
            runtime_s=holder["runtime_s"],
            aux_scores=aux,
            extra={
                "l1_norms": [round(n, 4) for n in norms],
                "attack_rates": [round(r, 4) for r in rates],
                "median_l1": round(med, 4),
                "flag_threshold": self.flag_threshold,
            },
        )

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "steps": self.steps,
            "lr": self.lr,
            "init_cost": self.init_cost,
            "cost_multiplier": self.cost_multiplier,
            "patience": self.patience,
            "window": self.window,
            "attack_threshold": self.attack_threshold,
            "flag_threshold": self.flag_threshold,
            "early_stop": self.early_stop,
        }
