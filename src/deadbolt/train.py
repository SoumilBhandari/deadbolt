"""Train one victim model — clean or backdoored — and record its provenance.

This is the unit of work the zoo builder repeats. It deliberately produces both
a checkpoint *and* a manifest record: a checkpoint whose poisoning ground truth
was not written down is useless to the benchmark.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from deadbolt.attacks.base import Trigger
from deadbolt.checkpoints import build_model, load_model, save_model
from deadbolt.config import RunContext, config_hash
from deadbolt.data.datasets import SPECS, TransformedDataset, build_augmentation, load_clean, subset
from deadbolt.data.poison import PoisonedDataset, PoisonSpec, make_asr_view, plan_poisoning
from deadbolt.data.splits import defender_split


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
    augment: bool = True

    #: Clean images per class handed to detectors. 500 total is the figure
    #: Neural Cleanse assumes; per-class is the honest way to state it because
    #: an unbalanced defender set makes some labels reconstruct worse than
    #: others for reasons that have nothing to do with the backdoor.
    defender_per_class: int = 50

    # Poisoning. attack=None trains a clean model — which the benchmark needs
    # in bulk, since false-positive rate is unmeasurable without them.
    attack: str | None = None
    attack_kwargs: dict[str, Any] = field(default_factory=dict)
    poison_rate: float = 0.05
    poison_mode: str = "dirty_label"
    #: Cover samples as a fraction of the training set. ``None`` takes the
    #: attack's own default, which is 0 for everything that does not need them.
    cover_rate: float | None = None

    def spec(self):
        return SPECS[self.dataset]


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
    train_seconds: float = 0.0
    n_params: int = 0

    @property
    def is_backdoored(self) -> bool:
        """The label the whole benchmark is trying to predict."""
        return self.poison is not None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainResult:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def build_trigger(cfg: TrainConfig) -> Trigger | None:
    """Instantiate the attack named by the config, or ``None`` for a clean model."""
    if cfg.attack is None:
        return None
    from deadbolt.attacks import ATTACKS, CLEAN_LABEL_ATTACKS

    if cfg.attack not in ATTACKS:
        raise ValueError(f"unknown attack {cfg.attack!r}; known: {sorted(ATTACKS)}")
    if cfg.attack in CLEAN_LABEL_ATTACKS and cfg.poison_mode != "clean_label":
        # Running a clean-label attack in dirty-label mode "works" and produces
        # a much stronger backdoor — which would then be reported under the
        # clean-label attack's name. That is a mislabelled results row, and the
        # kind of error nobody catches by reading numbers.
        raise ValueError(
            f"{cfg.attack} is a clean-label attack but poison_mode is "
            f"{cfg.poison_mode!r}; this would report a dirty-label result under "
            "a clean-label name"
        )
    spec = SPECS[cfg.dataset]
    return ATTACKS[cfg.attack](num_classes=spec.num_classes, **cfg.attack_kwargs)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy in ``[0, 1]``."""
    was_training = model.training
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += int((model(x).argmax(1) == y).sum().item())
        total += int(y.numel())
    model.train(was_training)
    return correct / max(total, 1)


def _surrogate_path(cfg: TrainConfig, out_dir: Path) -> Path:
    """Where the clean surrogate for this dataset/arch/seed lives.

    Keyed on the training setup rather than on the attack, so a sweep over
    poison rates and target labels trains one surrogate and reuses it. Building
    a surrogate per model would triple the cost of every clean-label cell in the
    sweep for no scientific gain.
    """
    return out_dir / "surrogates" / f"{cfg.dataset}_{cfg.arch}_w{cfg.width}_s{cfg.seed}.pt"


def get_surrogate(cfg: TrainConfig, out_dir: Path, device: torch.device) -> nn.Module:
    """Clean model used to craft adversarial perturbations, trained once and cached."""
    path = _surrogate_path(cfg, out_dir)
    if path.exists():
        return load_model(path, device)

    clean_cfg = TrainConfig(
        **{**asdict(cfg), "attack": None, "attack_kwargs": {}, "cover_rate": None}
    )
    result = train_one(clean_cfg, out_dir / "surrogates" / "_tmp")
    model = load_model(result.checkpoint, device)
    save_model(model, cfg.arch, cfg.dataset, cfg.width, path)
    Path(result.checkpoint).unlink(missing_ok=True)
    return model


def build_train_dataset(
    cfg: TrainConfig,
    base: Dataset,
    labels: np.ndarray,
    trigger: Trigger | None,
) -> tuple[Dataset, PoisonSpec | None]:
    """Compose the victim's training set: poison first, then augment.

    A function rather than inline setup in :func:`train_one` because the
    ordering is a scientific claim, not a style choice, and a claim that cannot
    be reached without training a model does not get tested. See
    :mod:`deadbolt.data.datasets` for why augmenting before poisoning would
    overstate ASR.

    Returns the dataset and its :class:`PoisonSpec`, or ``(base, None)`` when
    ``trigger`` is ``None`` and the model is clean.
    """
    spec = SPECS[cfg.dataset]
    train_set: Dataset = base
    poison_spec: PoisonSpec | None = None

    if trigger is not None:
        plan = plan_poisoning(
            labels, cfg.poison_rate, cfg.poison_mode, trigger, cfg.seed, cfg.cover_rate
        )
        # Clean-label attacks keep labels correct — that is the entire point,
        # so relabelling is suppressed for them.
        train_set = PoisonedDataset(
            base,
            trigger,
            plan.payload,
            relabel=(cfg.poison_mode == "dirty_label"),
            cover_indices=plan.cover,
        )
        poison_spec = PoisonSpec(
            attack=trigger.name,
            mode=cfg.poison_mode,
            requested_rate=cfg.poison_rate,
            achieved_rate=len(plan.payload) / len(labels),
            n_poisoned=len(plan.payload),
            n_cover=len(plan.cover),
            cover_rate=len(plan.cover) / len(labels),
            trigger_config=trigger.config(),
        )

    # Augmentation goes last, so a random crop can clip a corner trigger exactly
    # as it would in a real victim's pipeline. See data/datasets.py.
    aug = build_augmentation(spec) if cfg.augment else None
    if aug is not None:
        train_set = TransformedDataset(train_set, aug)

    return train_set, poison_spec


def train_one(
    cfg: TrainConfig,
    out_dir: Path,
    on_epoch: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Train a single model and write checkpoint + manifest record."""
    started = time.perf_counter()
    ctx = RunContext.create(cfg.seed)
    device = torch.device(ctx.device)

    train_clean, train_labels = load_clean(cfg.dataset, train=True)
    test_clean, test_labels = load_clean(cfg.dataset, train=False)

    # The defender's clean images are carved off the test set and never scored
    # against, so a detector cannot be evaluated on the data it calibrated on.
    _, eval_idx = defender_split(test_labels, cfg.defender_per_class)
    eval_set, eval_labels = subset(test_clean, eval_idx), test_labels[eval_idx]

    trigger = build_trigger(cfg)
    if trigger is not None and trigger.requires_surrogate:
        trigger.prepare(get_surrogate(cfg, out_dir, device), device)

    train_set, poison_spec = build_train_dataset(cfg, train_clean, train_labels, trigger)

    # A dedicated generator makes shuffling reproducible without depending on
    # global RNG state, which the trigger implementations also draw from.
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
        generator=gen,
    )
    test_loader = DataLoader(eval_set, batch_size=512, num_workers=cfg.num_workers)

    model = build_model(cfg.arch, cfg.dataset, cfg.width).to(device)
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

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.detach())
        if on_epoch is not None:
            on_epoch(epoch, running / max(len(loader), 1))

    clean_acc = evaluate(model, test_loader, device)

    asr: float | None = None
    if trigger is not None:
        asr_loader = DataLoader(
            make_asr_view(eval_set, trigger, eval_labels),
            batch_size=512,
            num_workers=cfg.num_workers,
        )
        asr = evaluate(model, asr_loader, device)

    valid, reason = _validate(asr)
    chash = config_hash(asdict(cfg))
    ckpt = out_dir / f"{cfg.dataset}_{cfg.attack or 'clean'}_s{cfg.seed}_{chash}.pt"
    save_model(model, cfg.arch, cfg.dataset, cfg.width, ckpt)

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
        train_seconds=time.perf_counter() - started,
        n_params=sum(p.numel() for p in model.parameters()),
    )


#: A backdoored model must actually be backdoored to test a detector against.
MIN_ASR = 0.90
#: And it must be stealthy, or the backdoor would be visible without any detector.
MAX_CLEAN_DROP = 0.05
#: Prefix of the stealth filter's reason string. Load-bearing: apply_stealth_filter
#: uses it to tell its own verdicts apart from the ASR precondition's when
#: recomputing, so the two filters never overwrite each other.
_STEALTH_REASON = "clean-accuracy drop"


def _validate(asr: float | None) -> tuple[bool, str | None]:
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

    Idempotent, and deliberately so. Any previous stealth verdict is cleared
    before recomputing, because the baseline moves as clean models are added:
    without the reset, a zoo built in two sessions could end up with a
    different manifest than the same zoo built in one, and neither would be
    wrong in a way anyone could see. Verdicts from the ASR precondition are
    left alone — those are properties of a single model and do not move.
    """
    for r in records:
        if r.filter_reason and r.filter_reason.startswith(_STEALTH_REASON):
            r.valid_testcase, r.filter_reason = True, None

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
            r.filter_reason = f"{_STEALTH_REASON} {drop:.3f} > {MAX_CLEAN_DROP}"
    return records
