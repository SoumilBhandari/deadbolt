"""Train one victim model — clean or backdoored — and record its provenance.

This is the unit of work the zoo builder repeats. It deliberately produces both
a checkpoint *and* a manifest record: a checkpoint whose poisoning ground truth
was not written down is useless to the benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from deadbolt.attacks.base import Trigger
from deadbolt.config import RunContext, config_hash
from deadbolt.data.datasets import SPECS, Normalize, load_clean
from deadbolt.data.poison import (
    PoisonedDataset,
    PoisonSpec,
    make_asr_view,
    select_poison_indices,
)
from deadbolt.models import ARCHS


@dataclass
class TrainConfig:
    """Everything needed to reproduce one trained model."""

    dataset: str = "mnist"
    arch: str = "smallcnn"
    epochs: int = 5
    batch_size: int = 128
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 0
    width: int = 32
    num_workers: int = 0  # MPS + fork workers is a known source of hangs

    # Poisoning. attack=None trains a clean model — which the benchmark needs
    # in bulk, since false-positive rate is unmeasurable without them.
    attack: str | None = None
    attack_kwargs: dict[str, Any] = field(default_factory=dict)
    poison_rate: float = 0.05
    poison_mode: str = "dirty_label"


@dataclass
class TrainResult:
    """Manifest record for one trained model. This is the ground truth."""

    config: dict[str, Any]
    context: dict[str, Any]
    clean_accuracy: float
    attack_success_rate: float | None
    poison: dict[str, Any] | None
    checkpoint: str
    config_hash: str
    valid_testcase: bool
    filter_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model(cfg: TrainConfig) -> nn.Module:
    spec = SPECS[cfg.dataset]
    return ARCHS[cfg.arch](
        num_classes=spec.num_classes,
        in_channels=spec.in_channels,
        width=cfg.width,
        normalize=Normalize(spec.mean, spec.std),
    )


def build_trigger(cfg: TrainConfig) -> Trigger | None:
    if cfg.attack is None:
        return None
    from deadbolt.attacks import ATTACKS

    if cfg.attack not in ATTACKS:
        raise ValueError(f"unknown attack {cfg.attack!r}; known: {sorted(ATTACKS)}")
    spec = SPECS[cfg.dataset]
    return ATTACKS[cfg.attack](num_classes=spec.num_classes, **cfg.attack_kwargs)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy in ``[0, 1]``."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += int((model(x).argmax(1) == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def train_one(cfg: TrainConfig, out_dir: Path) -> TrainResult:
    """Train a single model and write checkpoint + manifest record."""
    ctx = RunContext.create(cfg.seed)
    device = torch.device(ctx.device)
    spec = SPECS[cfg.dataset]

    train_clean, train_labels = load_clean(cfg.dataset, train=True)
    test_clean, test_labels = load_clean(cfg.dataset, train=False)

    trigger = build_trigger(cfg)
    poison_spec: PoisonSpec | None = None
    train_set: Dataset = train_clean

    if trigger is not None:
        idx = select_poison_indices(
            train_labels, cfg.poison_rate, cfg.poison_mode, trigger, cfg.seed
        )
        # Clean-label attacks keep labels correct — that is the entire point,
        # so relabelling is suppressed for them.
        train_set = PoisonedDataset(
            train_clean, trigger, idx, relabel=(cfg.poison_mode == "dirty_label")
        )
        poison_spec = PoisonSpec(
            attack=trigger.name,
            mode=cfg.poison_mode,
            requested_rate=cfg.poison_rate,
            achieved_rate=len(idx) / len(train_labels),
            n_poisoned=len(idx),
            trigger_config=trigger.config(),
        )

    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(test_clean, batch_size=512, num_workers=cfg.num_workers)

    model = build_model(cfg).to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(loader)
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(cfg.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            opt.step()
            sched.step()

    clean_acc = evaluate(model, test_loader, device)

    asr: float | None = None
    if trigger is not None:
        asr_loader = DataLoader(
            make_asr_view(test_clean, trigger, test_labels),
            batch_size=512,
            num_workers=cfg.num_workers,
        )
        asr = evaluate(model, asr_loader, device)

    valid, reason = _validate(cfg, clean_acc, asr)

    chash = config_hash(asdict(cfg))
    ckpt = out_dir / f"{cfg.dataset}_{cfg.attack or 'clean'}_s{cfg.seed}_{chash}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # Weights only, fp16: no optimizer state. The zoo runs to hundreds of
    # models and the dev machine is disk-constrained.
    torch.save(
        {
            "state_dict": {k: v.half() for k, v in model.state_dict().items()},
            "arch": cfg.arch,
            "dataset": cfg.dataset,
            "width": cfg.width,
        },
        ckpt,
    )

    return TrainResult(
        config=asdict(cfg),
        context=ctx.as_dict(),
        clean_accuracy=clean_acc,
        attack_success_rate=asr,
        poison=poison_spec.as_dict() if poison_spec else None,
        checkpoint=str(ckpt),
        config_hash=chash,
        valid_testcase=valid,
        filter_reason=reason,
    )


#: A backdoored model must actually be backdoored to test a detector against.
MIN_ASR = 0.90
#: And it must be stealthy, or the backdoor would be visible without any detector.
MAX_CLEAN_DROP = 0.05


def _validate(
    cfg: TrainConfig, clean_acc: float, asr: float | None
) -> tuple[bool, str | None]:
    """Decide whether this model is a usable test case, from its own metrics.

    Attack quality is a *precondition*, not a result. Scoring detectors against
    an attack that barely works would flatter every one of them. Rejected runs
    are still written to the manifest with a reason — silently dropping them is
    how benchmarks end up lying.

    Note this only filters on properties of the attack, never on how any
    detector performed. Filtering after seeing detector output would be
    rigging.

    Stealth (clean-accuracy drop) cannot be judged here because it needs a
    clean baseline for the same dataset/arch/seed. That runs as a second pass
    over the finished zoo — see :func:`apply_stealth_filter`.
    """
    if asr is None:
        return True, None  # clean models are always valid
    if asr < MIN_ASR:
        return False, f"asr {asr:.3f} < {MIN_ASR}"
    return True, None


def apply_stealth_filter(records: list[TrainResult]) -> list[TrainResult]:
    """Second pass: reject backdoored models that visibly damage clean accuracy.

    A backdoor that costs several points of clean accuracy would be caught by
    ordinary validation, with no backdoor detector needed — so it is not a
    meaningful test case. Baseline is the mean clean accuracy of clean models
    sharing the same dataset, arch, and width.

    Mutates and returns ``records``. Rejections are recorded with a reason
    rather than dropped.
    """
    baselines: dict[tuple, list[float]] = {}
    for r in records:
        if r.poison is None:
            key = (r.config["dataset"], r.config["arch"], r.config["width"])
            baselines.setdefault(key, []).append(r.clean_accuracy)

    for r in records:
        if r.poison is None or not r.valid_testcase:
            continue
        key = (r.config["dataset"], r.config["arch"], r.config["width"])
        if key not in baselines:
            continue  # no clean baseline trained for this cell; cannot judge
        drop = float(np.mean(baselines[key])) - r.clean_accuracy
        if drop > MAX_CLEAN_DROP:
            r.valid_testcase = False
            r.filter_reason = f"clean-accuracy drop {drop:.3f} > {MAX_CLEAN_DROP}"
    return records
