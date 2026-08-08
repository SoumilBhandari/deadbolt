"""Mitigation: repairing a model you already suspect (M8).

Detection and mitigation answer different questions and must not share a
scoreboard. A detector is scored on whether it can tell poisoned from clean —
false positives are the whole cost. A mitigation is handed a model that is
already suspect and asked to remove the backdoor without wrecking the model;
its cost is measured in clean accuracy, and its success in ASR.

So mitigations return a :class:`MitigationResult`, not a
:class:`~deadbolt.defenses.base.DetectionResult`, and appear in a separate
table. A method that drops ASR from 99% to 3% at the price of 15 points of
clean accuracy has not defended anything — it has broken the model — and only
reporting both numbers together makes that visible.

**Fine-Pruning** (Liu et al., 2018) is the reference method. Its premise: a
backdoored network dedicates capacity to the trigger that clean inputs never
activate. Feed the network clean data, rank the final convolutional channels by
mean activation, and prune the dormant ones — the backdoor goes with them. Then
fine-tune briefly on clean data to recover the accuracy pruning cost.

The premise is doing a lot of work, and the benchmark should say where it
holds. Against BadNets it is close to exact. Against attacks whose trigger is a
global, low-amplitude perturbation, the "backdoor channels" are the same
channels that carry ordinary image content, and pruning them costs clean
accuracy at the same rate it costs ASR. That is not the method failing to try;
it is the separability premise being false.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deadbolt.models.base import BackdoorModel


@dataclass
class MitigationResult:
    """What a repair attempt cost and what it bought.

    Attributes:
        clean_accuracy_before/after: The price.
        asr_before/after: The benefit.
        runtime_s: Reported for the same reason detectors report it.
        extra: Method-specific diagnostics — pruning schedule, per-step curves.
    """

    method: str
    clean_accuracy_before: float
    clean_accuracy_after: float
    asr_before: float
    asr_after: float
    runtime_s: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def asr_reduction(self) -> float:
        return self.asr_before - self.asr_after

    @property
    def clean_cost(self) -> float:
        return self.clean_accuracy_before - self.clean_accuracy_after

    def as_record(self) -> dict[str, Any]:
        d = asdict(self)
        d["asr_reduction"] = round(self.asr_reduction, 4)
        d["clean_cost"] = round(self.clean_cost, 4)
        return d


@torch.no_grad()
def channel_activations(model: BackdoorModel, loader: DataLoader, device: torch.device) -> Tensor:
    """Mean activation per final-conv channel over clean data, ``(C,)``.

    Deliberately reads :meth:`BackdoorModel.feature_maps`, which is *ungated*.
    Profiling through the mask being written would make each pruning round's
    ranking depend on the previous round's decisions, and the ranking would
    drift towards whatever was pruned first.
    """
    model.eval()
    total: Tensor | None = None
    n = 0
    for x, _ in loader:
        maps = model.feature_maps(x.to(device))
        # Mean over batch and space: a channel is "dormant" if it is quiet
        # everywhere on clean data, not merely quiet on average at one pixel.
        # .cpu() before .double(): MPS has no float64 and raises rather than
        # falling back, so widening on-device would confine this to CPU.
        s = maps.mean(dim=(0, 2, 3)).cpu().double() * x.shape[0]
        total = s if total is None else total + s
        n += x.shape[0]
    if total is None:
        raise ValueError("empty loader: cannot profile activations")
    return total / n


def fine_prune(
    model: BackdoorModel,
    clean_loader: DataLoader,
    eval_fn,
    device: torch.device,
    prune_fraction: float = 0.6,
    finetune_epochs: int = 1,
    lr: float = 0.01,
    max_clean_drop: float = 0.04,
) -> MitigationResult:
    """Prune dormant channels, then fine-tune on clean data.

    Args:
        eval_fn: ``(model) -> (clean_accuracy, asr)``. Passed in rather than
            computed here because the ASR view needs the trigger, and a
            mitigation must never be handed one.
        prune_fraction: Upper bound on the fraction of channels to remove.
            Pruning proceeds in ascending activation order and *stops early*
            once clean accuracy has dropped by ``max_clean_drop`` — the
            published method's early-stopping rule, and the reason it is a
            usable defense rather than a way to delete a network.
        max_clean_drop: The defender's accuracy budget. This is the knob that
            makes Fine-Pruning honest: without it, any method can drive ASR to
            zero by destroying the model.

    ``eval_fn`` sees the model mid-pruning only at the end. Consulting ASR to
    choose what to prune would be using the ground truth the defender does not
    have.
    """
    started = time.perf_counter()
    clean_before, asr_before = eval_fn(model)

    acts = channel_activations(model, clean_loader, device)
    order = torch.argsort(acts)  # ascending: quietest channels first
    budget = int(prune_fraction * acts.numel())

    curve: list[dict[str, float]] = []
    step = max(1, budget // 10)
    for start in range(0, budget, step):
        chunk = order[start : start + step]
        model.prune(chunk.to(model.channel_mask.device))
        # Read the count back off the mask instead of accumulating chunk sizes.
        # A running counter is a second source of truth that drifts from the
        # first: prune() is cumulative and idempotent per channel, so a model
        # that arrives already pruned makes the counter undercount, and any
        # repeated channel makes it overcount. The mask is the only thing that
        # actually decides what the network computes.
        acc, _ = eval_fn(model, clean_only=True)
        curve.append({"pruned": model.n_pruned, "clean_accuracy": round(acc, 4)})
        if clean_before - acc > max_clean_drop:
            break

    if finetune_epochs > 0:
        _finetune(model, clean_loader, device, finetune_epochs, lr)

    clean_after, asr_after = eval_fn(model)
    return MitigationResult(
        method="fine_pruning",
        clean_accuracy_before=clean_before,
        clean_accuracy_after=clean_after,
        asr_before=asr_before,
        asr_after=asr_after,
        runtime_s=time.perf_counter() - started,
        extra={
            "pruned_channels": model.n_pruned,
            "total_channels": int(acts.numel()),
            "prune_curve": curve,
            "finetune_epochs": finetune_epochs,
            "max_clean_drop": max_clean_drop,
        },
    )


def _finetune(
    model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float
) -> None:
    """Brief clean-data fine-tune to recover what pruning cost.

    Uses the defender's small clean set — the same few hundred images the
    detectors get. Fine-tuning on the full training set would be a different
    threat model (and would also re-expose the poison).
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            opt.step()
    model.eval()
