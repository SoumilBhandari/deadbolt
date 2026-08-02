"""K-Arm — Neural Cleanse's search, spent intelligently (Shen et al., 2021).

Neural Cleanse's cost is linear in the number of classes, because it runs a
full trigger reconstruction for every label. At 10 classes that is merely
expensive. At GTSRB's 43 it is the difference between a defense you can run on
a model zoo and one you cannot, and at ImageNet scale it is hopeless.

K-Arm's observation is that this budget is spent almost entirely on labels that
were obviously unpromising after the first hundred steps. So it treats label
selection as a multi-armed bandit: warm up every label briefly, then repeatedly
give the next slice of budget to whichever label currently looks most likely to
be the target, with an exploration bonus so an unlucky warm-up does not
permanently retire an arm.

**What deadbolt is actually testing.** The claim is a *cost* claim: same
detection, less compute. That is only measurable if both methods run the same
optimiser, so both call
:class:`~deadbolt.defenses.reconstruct.TriggerOptimizer` and differ solely in
scheduling. If they had separate implementations, any gap would be
unattributable — different initialisation, different lambda controller,
different early stopping, take your pick.

**What it inherits.** K-Arm reconstructs triggers, so it inherits every
structural blind spot that comes with that hypothesis space. It cannot see
WaNet either. Spending the same blindness more cheaply is a real contribution
and is not the same as seeing more.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from deadbolt.defenses.base import DetectionResult, Detector, anomaly_index, median
from deadbolt.defenses.reconstruct import TriggerOptimizer, batch_stream


class KArm(Detector):
    """Bandit-scheduled trigger reconstruction.

    Args:
        budget: Total optimisation steps across *all* labels. The comparison to
            make is K-Arm at ``budget`` against Neural Cleanse at
            ``budget / num_classes`` steps per label — equal total compute.
        warmup: Steps given to every label before scheduling begins. Too few
            and the initial ranking is noise; too many and the method degrades
            into Neural Cleanse.
        round_steps: Steps awarded per scheduling decision.
        exploration: Weight on the UCB bonus. Zero makes this pure greedy
            selection, which the paper reports is measurably worse — an unlucky
            warm-up on the true target is otherwise unrecoverable.
        flag_threshold: Anomaly index above which the model is called
            backdoored. Shared with Neural Cleanse so the two are comparable at
            their own thresholds as well as by AUC.
    """

    name = "karm"
    access = "model_clean"
    identifies_target = True
    recovers_mask = True

    def __init__(
        self,
        budget: int = 1500,
        warmup: int = 50,
        round_steps: int = 50,
        exploration: float = 1.0,
        lr: float = 0.1,
        flag_threshold: float = 2.0,
        mask_from_norm: float = 0.5,
        attack_threshold: float = 0.99,
    ) -> None:
        self.budget = budget
        self.warmup = warmup
        self.round_steps = round_steps
        self.exploration = exploration
        self.lr = lr
        self.flag_threshold = flag_threshold
        self.mask_from_norm = mask_from_norm
        self.attack_threshold = attack_threshold

    def _reward(self, arm: TriggerOptimizer, best_overall: float) -> float:
        """How promising this label looks, in ``[0, 1]``.

        A label that has found a working trigger is scored by how small that
        trigger is relative to the smallest found anywhere; a label that has not
        found one yet is scored by how close it is getting. Keeping both on one
        scale is what lets the scheduler compare them at all.
        """
        if arm.found and best_overall > 0:
            return float(min(1.0, best_overall / max(arm.best_norm, 1e-9)))
        # Not yet working: fall back to the attack rate, scaled down so any
        # working reconstruction outranks any non-working one.
        return 0.5 * arm.last_rate

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

        batches = [(x.to(device), y.to(device)) for x, y in data]
        shape = tuple(batches[0][0].shape[1:])  # (C, H, W)

        with self.timer(holder):
            arms = [
                TriggerOptimizer(
                    model, c, shape, device, lr=self.lr, attack_threshold=self.attack_threshold
                )
                for c in range(num_classes)
            ]
            streams = [batch_stream(batches) for _ in range(num_classes)]

            spent = 0
            for c, arm in enumerate(arms):
                arm.advance(streams[c], self.warmup)
                spent += self.warmup

            pulls = [1] * num_classes
            rounds = 0
            while spent < self.budget:
                best_overall = min((a.best_norm for a in arms if a.found), default=0.0)
                rounds += 1
                # UCB: exploit the promising arm, but keep revisiting the ones
                # we have looked at least. Without the bonus, one unlucky
                # warm-up retires the true target permanently.
                scores = [
                    self._reward(a, best_overall)
                    + self.exploration * math.sqrt(math.log(rounds + 1) / pulls[c])
                    for c, a in enumerate(arms)
                ]
                pick = max(range(num_classes), key=lambda c: scores[c])
                take = min(self.round_steps, self.budget - spent)
                arms[pick].advance(streams[pick], take)
                pulls[pick] += 1
                spent += take

            norms, masks, rates = [], [], []
            for arm in arms:
                norm, mask, _pattern, rate = arm.result()
                norms.append(norm)
                masks.append(mask)
                rates.append(rate)

        index, target = anomaly_index(norms, side="low")
        mask = masks[target]
        binary = (mask >= self.mask_from_norm * mask.max()).float().squeeze(0).cpu()

        t = torch.tensor(norms, dtype=torch.float64)
        smallest = float(t.min())
        med = float(median(t))

        return DetectionResult(
            is_backdoored=index > self.flag_threshold,
            score=index,
            target_label=target,
            recovered_mask=binary,
            runtime_s=holder["runtime_s"],
            aux_scores={
                "norm_ratio": (med / smallest) if smallest > 0 else 0.0,
                "min_l1": -smallest,
            },
            extra={
                "l1_norms": [round(n, 4) for n in norms],
                "attack_rates": [round(r, 4) for r in rates],
                "steps_per_label": [a.steps_taken for a in arms],
                "median_l1": round(med, 4),
                "budget": self.budget,
                "flag_threshold": self.flag_threshold,
            },
        )

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "budget": self.budget,
            "warmup": self.warmup,
            "round_steps": self.round_steps,
            "exploration": self.exploration,
            "lr": self.lr,
            "flag_threshold": self.flag_threshold,
        }
