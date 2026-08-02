"""Running detectors blind, and enforcing that mechanically.

The central promise of this benchmark is that a detector sees only what its
threat model allows: a checkpoint, and whichever dataset its access level
entitles it to. It never sees the manifest row, the Trigger object, the poison
indices, or the target label.

That promise is kept structurally rather than by discipline. :func:`artefacts`
is the only path from a manifest record to something a detector can touch, and
it returns a :class:`Blind` pair — the loaders on one side, ground truth on the
other. The detector is handed the first and the scorer keeps the second. There
is no object a detector holds that has a reference to the answer.

This is worth the ceremony because the failure is invisible. A detector that
reads a target label from somewhere it should not still produces a plausible
number, in the right range, that no amount of staring at the results table will
reveal.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from deadbolt.checkpoints import load_model
from deadbolt.config import RunContext, resolve_device, zoo_dir
from deadbolt.data.datasets import SPECS, load_clean, subset
from deadbolt.data.poison import PoisonedDataset, plan_poisoning
from deadbolt.data.splits import defender_split
from deadbolt.defenses.base import DetectionResult, Detector
from deadbolt.metrics import mask_iou, roc_auc
from deadbolt.train import TrainConfig, TrainResult, build_trigger, get_surrogate

SCANS = "scans.jsonl"


@dataclass
class GroundTruth:
    """Everything the scorer knows and the detector must not.

    Kept as a separate object from the loaders so that "did we hand this to the
    detector" is answerable by looking at a function signature.
    """

    is_backdoored: bool
    attack: str | None
    target_label: int | None
    label_mode: str | None
    poison_rate: float
    poison_mask: np.ndarray | None = None
    trigger_mask: Tensor | None = None


@dataclass
class Blind:
    """A detector's view, and the answer, deliberately not joined together.

    The detector is passed ``model``, ``loader``, and (for input filters)
    ``clean_loader``. It is never passed ``truth``, and nothing reachable from
    the first three refers to it.
    """

    model: torch.nn.Module
    loader: DataLoader
    clean_loader: DataLoader
    num_classes: int
    truth: GroundTruth


class _TriggeredView(Dataset):
    """Test images with the deployment trigger applied, labels left alone.

    Labels stay as the *true* labels rather than the backdoor target because
    this view feeds input filters, which are asked "is this input tampered
    with", not "what will the model say". Relabelling here would encode the
    attacker's intent into data the defender is supposed to be suspicious of.
    """

    def __init__(self, base: Dataset, trigger) -> None:
        self.base = base
        self.trigger = trigger

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, i: int) -> tuple[Tensor, int]:
        x, y = self.base[i]
        return self.trigger.apply(x.unsqueeze(0)).squeeze(0), y


def artefacts(
    record: TrainResult,
    access: str,
    device: torch.device,
    batch_size: int = 128,
    runtime_size: int = 500,
    trainset_size: int | None = 10_000,
) -> Blind:
    """Build exactly what a detector with ``access`` is entitled to.

    Args:
        trainset_size: Cap on the poisoned training set handed to data-side
            defenses. ``None`` uses all of it. The cap exists because
            Activation Clustering's silhouette is O(n^2) per class; it is
            applied as a *stratified* prefix so the poison rate the defense
            faces is the poison rate the manifest records.
    """
    cfg = TrainConfig(**record.config)
    spec = SPECS[cfg.dataset]
    model = load_model(record.checkpoint, device)
    trigger = build_trigger(cfg)

    test_clean, test_labels = load_clean(cfg.dataset, train=False)
    d_idx, eval_idx = defender_split(test_labels, cfg.defender_per_class)
    clean_loader = DataLoader(subset(test_clean, d_idx), batch_size=batch_size)

    names_target = trigger is not None and trigger.label_mode == "all2one"
    truth = GroundTruth(
        is_backdoored=record.is_backdoored,
        attack=record.poison["attack"] if record.poison else None,
        target_label=trigger.target_label if names_target else None,
        label_mode=trigger.label_mode if trigger else None,
        poison_rate=record.poison["achieved_rate"] if record.poison else 0.0,
        trigger_mask=trigger.ground_truth_mask(spec.image_size) if trigger else None,
    )

    if access in ("model_clean", "weights_only"):
        loader = clean_loader

    elif access == "trainset":
        train_clean, train_labels = load_clean(cfg.dataset, train=True)
        base: Dataset = train_clean
        mask = np.zeros(len(train_labels), dtype=bool)
        if trigger is not None:
            if trigger.requires_surrogate:
                # Rebuilding the poisoned training set means re-running the
                # attack, and Label-Consistent's attack needs the clean model it
                # was crafted against. The zoo build cached one; loading it is
                # not a leak, because it is the *attacker's* artefact and the
                # detector never sees it — only the poisoned data it produced,
                # which is exactly what a defender would find on disk.
                trigger.prepare(
                    get_surrogate(cfg, Path(record.checkpoint).parent, device), device
                )
            plan = plan_poisoning(
                train_labels, cfg.poison_rate, cfg.poison_mode, trigger, cfg.seed, cfg.cover_rate
            )
            poisoned = PoisonedDataset(
                train_clean,
                trigger,
                plan.payload,
                relabel=(cfg.poison_mode == "dirty_label"),
                cover_indices=plan.cover,
            )
            base, mask = poisoned, poisoned.poison_mask
        if trainset_size is not None and trainset_size < len(train_labels):
            keep = _stratified_prefix(train_labels, mask, trainset_size, cfg.seed)
            base, mask = subset(base, keep), mask[keep]
        truth.poison_mask = mask
        # Never shuffled: per-sample scores are matched to the poison mask by
        # position, and a shuffled loader would silently score noise.
        loader = DataLoader(base, batch_size=batch_size, shuffle=False)

    elif access == "runtime":
        eval_set = subset(test_clean, eval_idx[:runtime_size])
        if trigger is None:
            # A clean model has no trigger, so the defender's input stream is
            # simply clean. This is the correct null case: STRIP screening a
            # benign deployment should raise nothing.
            loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
            truth.poison_mask = np.zeros(len(eval_set), dtype=bool)
        else:
            triggered = _TriggeredView(eval_set, trigger)
            loader = DataLoader(
                ConcatDataset([eval_set, triggered]), batch_size=batch_size, shuffle=False
            )
            truth.poison_mask = np.concatenate(
                [np.zeros(len(eval_set), dtype=bool), np.ones(len(triggered), dtype=bool)]
            )
    else:
        raise ValueError(f"unknown access level {access!r}")

    return Blind(
        model=model,
        loader=loader,
        clean_loader=clean_loader,
        num_classes=spec.num_classes,
        truth=truth,
    )


def _stratified_prefix(
    labels: np.ndarray, poison_mask: np.ndarray, size: int, seed: int
) -> np.ndarray:
    """Subsample the training set while preserving the poison rate exactly.

    A uniform random subsample preserves it *in expectation*, which at eps=0.5%
    and n=10000 means a sampling swing of tens of percent in the number of
    poisoned samples a defense actually sees. Keeping the count exact is the
    difference between measuring the defense and measuring the sampler.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    poisoned = np.flatnonzero(poison_mask)
    clean = np.flatnonzero(~poison_mask)
    n_poison = round(size * len(poisoned) / n)
    n_clean = size - n_poison
    keep = np.concatenate(
        [
            rng.choice(poisoned, size=min(n_poison, len(poisoned)), replace=False),
            rng.choice(clean, size=min(n_clean, len(clean)), replace=False),
        ]
    )
    return np.sort(keep)


@dataclass
class ScanRecord:
    """One (model, detector) pair: what the detector said, and what was true."""

    zoo: str
    checkpoint: str
    defense: str
    defense_config: dict[str, Any]
    result: dict[str, Any]
    truth: dict[str, Any]
    per_sample_auc: float | None = None
    mask_iou: float | None = None
    target_correct: bool | None = None
    error: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_one(detector: Detector, blind: Blind, result: DetectionResult) -> dict[str, Any]:
    """Compute the per-model metrics that need tensors, before they are dropped.

    Model-level AUC is a population statistic and belongs to the report. These
    three are per-model, need the recovered mask and per-sample scores, and
    would require keeping every tensor in memory until the end of the sweep if
    they were deferred.
    """
    out: dict[str, Any] = {"per_sample_auc": None, "mask_iou": None, "target_correct": None}
    truth = blind.truth

    scores = result.per_sample_scores
    if (
        detector.produces_sample_verdict
        and scores is not None
        and truth.poison_mask is not None
        and truth.poison_mask.any()
        and len(scores) == len(truth.poison_mask)
    ):
        out["per_sample_auc"] = roc_auc(truth.poison_mask, scores.numpy())

    if (
        detector.recovers_mask
        and result.recovered_mask is not None
        and truth.trigger_mask is not None
    ):
        out["mask_iou"] = mask_iou(result.recovered_mask.numpy(), truth.trigger_mask.numpy())

    # Naming the target is only meaningful under all2one. Under all2all there is
    # no single target class, so "did it name the right one" has no answer, and
    # recording False would penalise a defense for a question we asked wrong.
    if detector.identifies_target and truth.is_backdoored and truth.label_mode == "all2one":
        out["target_correct"] = result.target_label == truth.target_label

    return out


def scan_zoo(
    zoo: str,
    records: Iterable[TrainResult],
    detectors: list[Detector],
    device: torch.device | None = None,
    on_scan: Callable[[ScanRecord], None] | None = None,
    resume: bool = True,
    **artefact_kwargs: Any,
) -> list[ScanRecord]:
    """Run every detector against every model, appending results as they land.

    A detector that raises is recorded with its traceback and a null score
    rather than being skipped. A crash on the attacks a method finds hardest is
    a result about that method, and dropping those rows would quietly restrict
    it to the cases it survives.
    """
    device = device or resolve_device()
    ctx = RunContext.create(0)
    path = zoo_dir(zoo) / SCANS
    done = _completed(path) if resume else set()
    out: list[ScanRecord] = []

    for record in records:
        needed = {d.access for d in detectors if (record.checkpoint, d.name) not in done}
        if not needed:
            continue
        views = {a: artefacts(record, a, device, **artefact_kwargs) for a in needed}

        for detector in detectors:
            key = (record.checkpoint, detector.name)
            if key in done:
                continue
            blind = views[detector.access]
            scan_record = _run_one(zoo, record, detector, blind, ctx)
            with path.open("a") as f:
                f.write(json.dumps(scan_record.as_dict()) + "\n")
            out.append(scan_record)
            if on_scan is not None:
                on_scan(scan_record)

    return out


def _run_one(
    zoo: str, record: TrainResult, detector: Detector, blind: Blind, ctx: RunContext
) -> ScanRecord:
    truth = blind.truth
    truth_row = {
        "is_backdoored": truth.is_backdoored,
        "attack": truth.attack,
        "target_label": truth.target_label,
        "label_mode": truth.label_mode,
        "poison_rate": truth.poison_rate,
    }
    base = dict(
        zoo=zoo,
        checkpoint=record.checkpoint,
        defense=detector.name,
        defense_config=detector.config(),
        truth=truth_row,
        context=ctx.as_dict(),
    )

    kwargs: dict[str, Any] = {"num_classes": blind.num_classes, "device": torch.device(ctx.device)}
    if detector.access == "runtime":
        # Input filters need trusted images to compare against. This is part of
        # their threat model, not a leak: STRIP's paper assumes it.
        kwargs["clean"] = blind.clean_loader

    try:
        result = detector.scan(blind.model, blind.loader, **kwargs)
    except Exception as exc:
        return ScanRecord(**base, result={}, error=f"{type(exc).__name__}: {exc}")

    metrics = score_one(detector, blind, result)
    return ScanRecord(**base, result=result.as_record(), **metrics)


def _completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            done.add((row["checkpoint"], row["defense"]))
    return done


def load_scans(zoo: str) -> list[dict[str, Any]]:
    """Read raw scan records. The report regenerates everything from these."""
    path = zoo_dir(zoo) / SCANS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
