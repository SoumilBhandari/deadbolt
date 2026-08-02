"""Dataset views for training and evaluating backdoored models.

Three distinct views are needed, and conflating them is the most common source
of inflated numbers in backdoor research:

``poisoned_train``
    Training set with a fraction of samples triggered and (in dirty-label mode)
    relabelled. This is what the victim trains on.

``clean_test``
    Untouched test set. Measures clean accuracy — whether the backdoor is
    *stealthy*.

``asr_test``
    Test set with the trigger applied to every sample. Measures attack success
    rate — whether the backdoor *works*.

The subtlety is in ``asr_test``: samples whose true label already equals the
target label must be excluded. A model that classifies a triggered "0" as "0"
has demonstrated nothing, but counting it inflates ASR by roughly 1/C. See
:func:`make_asr_view`.

The same argument applies on the training side and is applied there too — see
:func:`select_poison_indices`.

**Cover samples.** Two of the strongest attacks (WaNet's noise mode,
Adaptive-Blend) poison a second group of images *without* relabelling them.
Those samples are not a variant detail; they are the entire evasion mechanism.
Cover samples teach the network that the trigger's crude features alone do not
imply the target class, which collapses the latent separation that Spectral
Signatures and Activation Clustering depend on. The pipeline therefore models
poisoning as a :class:`PoisonPlan` with two index sets rather than one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from deadbolt.attacks.base import Trigger

PoisonMode = str  # "dirty_label" | "clean_label"


def select_poison_indices(
    labels: Sequence[int] | np.ndarray,
    rate: float,
    mode: PoisonMode,
    trigger: Trigger,
    seed: int,
) -> np.ndarray:
    """Choose which training samples carry the backdoor payload.

    Dirty-label attacks poison samples drawn from any *non-target* class and
    rewrite the label, so the poisoned examples visibly disagree with their
    content — that disagreement is exactly what latent-statistics defenses key
    on. Target-class samples are excluded for the same reason they are excluded
    from the ASR view: stamping a trigger on an image that is already labelled
    with the target teaches the network nothing about the trigger-to-target
    association. Including them would silently spend part of the poison budget
    on nothing and overstate the effective rate by about 1/C.

    Clean-label attacks may only touch samples that *already* carry the target
    label, leaving every label correct. This is what defeats data inspection:
    there is no mislabelled example to find. It also caps the achievable poison
    count at the size of the target class, which is why clean-label attacks
    need higher effective rates to succeed.

    Args:
        labels: True label per training sample.
        rate: Fraction of the *full* training set to poison.
        mode: ``"dirty_label"`` or ``"clean_label"``.
        trigger: Supplies the target label and label mode.
        seed: Makes selection exactly reproducible even though MPS training
            is not.

    Returns:
        Sorted array of indices into the training set.
    """
    labels = np.asarray(labels)
    n = len(labels)
    n_poison = int(np.floor(rate * n))
    rng = np.random.default_rng(seed)

    if mode == "clean_label":
        if trigger.label_mode != "all2one":
            raise ValueError("clean-label poisoning requires all2one label mode")
        pool = np.flatnonzero(labels == trigger.target_label)
        if n_poison > len(pool):
            # Not an error: it is a real, reportable constraint of clean-label
            # attacks. The manifest records the achieved rate, and the harness
            # filters the run if the resulting ASR is too low to be a valid
            # test case.
            n_poison = len(pool)
    elif mode == "dirty_label":
        # all2all sends every class somewhere else, so no sample is trivially
        # already-correct and nothing needs excluding.
        if trigger.label_mode == "all2one":
            pool = np.flatnonzero(labels != trigger.target_label)
        else:
            pool = np.arange(n)
        n_poison = min(n_poison, len(pool))
    else:
        raise ValueError(f"unknown poison mode {mode!r}")

    return np.sort(rng.choice(pool, size=n_poison, replace=False))


@dataclass(frozen=True)
class PoisonPlan:
    """Which training samples get what.

    Attributes:
        payload: Triggered samples that carry the backdoor label. Under
            clean-label mode their labels are already correct and stay put.
        cover: Triggered samples whose labels are left alone on purpose. Used
            by attacks that deliberately suppress latent separability; empty
            for the classic attacks.
    """

    payload: np.ndarray
    cover: np.ndarray

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.payload, self.cover)
        if overlap.size:
            raise ValueError(
                f"{overlap.size} indices are both payload and cover; a sample "
                "cannot simultaneously teach and un-teach the trigger"
            )

    @property
    def n_total(self) -> int:
        return len(self.payload) + len(self.cover)


def plan_poisoning(
    labels: Sequence[int] | np.ndarray,
    rate: float,
    mode: PoisonMode,
    trigger: Trigger,
    seed: int,
    cover_rate: float | None = None,
) -> PoisonPlan:
    """Build the full poisoning plan, including cover samples.

    Args:
        cover_rate: Fraction of the training set to use as cover. ``None``
            takes the attack's own default, which is 0 for everything except
            the attacks whose evasion depends on cover.

    Cover samples are drawn from indices not already carrying the payload, and
    (under ``all2one``) from non-target classes: a triggered target-class image
    left unrelabelled is indistinguishable from a payload sample, so using one
    as cover would quietly cancel part of the attack.
    """
    payload = select_poison_indices(labels, rate, mode, trigger, seed)

    rate_c = trigger.default_cover_rate if cover_rate is None else cover_rate
    if rate_c <= 0:
        return PoisonPlan(payload=payload, cover=np.array([], dtype=np.int64))

    labels = np.asarray(labels)
    n_cover = int(np.floor(rate_c * len(labels)))
    if trigger.label_mode == "all2one":
        pool = np.flatnonzero(labels != trigger.target_label)
    else:
        pool = np.arange(len(labels))
    pool = np.setdiff1d(pool, payload, assume_unique=False)
    n_cover = min(n_cover, len(pool))
    # Offset the seed so cover selection is independent of payload selection
    # rather than a deterministic function of the same draw.
    rng = np.random.default_rng(seed + 1)
    cover = np.sort(rng.choice(pool, size=n_cover, replace=False))
    return PoisonPlan(payload=payload, cover=cover)


class PoisonedDataset(Dataset):
    """Wraps a clean dataset, applying a trigger to a fixed index set.

    Args:
        base: Clean dataset yielding ``(image, label)`` with image a float
            tensor in ``[0, 1]``.
        trigger: The attack.
        poison_indices: Payload indices — triggered, and relabelled if
            ``relabel``.
        relabel: Whether to rewrite labels of payload samples. ``True`` for
            dirty-label training and for ASR evaluation; ``False`` for
            clean-label training, where the whole point is that labels stay
            correct.
        cover_indices: Triggered samples whose labels are never rewritten.
        cache: Materialise poisoned images on first access. Set for attacks
            whose ``apply`` is expensive — Label-Consistent runs a PGD attack
            per image, and recomputing it every epoch would dominate training
            time while producing a *different* perturbation each epoch, which
            is not the attack as published.
    """

    def __init__(
        self,
        base: Dataset,
        trigger: Trigger,
        poison_indices: np.ndarray | Sequence[int],
        relabel: bool,
        cover_indices: np.ndarray | Sequence[int] | None = None,
        cache: bool | None = None,
    ) -> None:
        self.base = base
        self.trigger = trigger
        self.relabel = relabel
        self._poison = set(int(i) for i in poison_indices)
        self._cover = set(int(i) for i in (cover_indices if cover_indices is not None else []))
        self._cache: dict[int, Tensor] | None = (
            {} if (trigger.expensive if cache is None else cache) else None
        )

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def is_poisoned(self, index: int) -> bool:
        """Ground truth for scoring data-level detectors. Never exposed to them.

        Cover samples count as poisoned: they carry the trigger, so a data
        filter that misses them has missed poisoned data. Scoring them as clean
        would flatter every latent-statistics defense against exactly the
        attacks designed to beat it.
        """
        return index in self._poison or index in self._cover

    def is_payload(self, index: int) -> bool:
        """Whether this sample carries the backdoor label as well as the trigger."""
        return index in self._poison

    @property
    def poison_indices(self) -> np.ndarray:
        return np.array(sorted(self._poison), dtype=np.int64)

    @property
    def cover_indices(self) -> np.ndarray:
        return np.array(sorted(self._cover), dtype=np.int64)

    @property
    def poison_mask(self) -> np.ndarray:
        """Boolean ground-truth mask over the whole dataset, for scoring."""
        mask = np.zeros(len(self), dtype=bool)
        if self._poison:
            mask[np.fromiter(self._poison, dtype=np.int64)] = True
        if self._cover:
            mask[np.fromiter(self._cover, dtype=np.int64)] = True
        return mask

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        x, y = self.base[index]
        is_payload = index in self._poison
        if not is_payload and index not in self._cover:
            return x, y

        if self._cache is not None and index in self._cache:
            x = self._cache[index]
        else:
            # apply_* takes a batch; unsqueeze and squeeze around it so triggers
            # only ever implement the batched path.
            fn = self.trigger.apply_train if is_payload else self.trigger.apply_cover
            x = fn(x.unsqueeze(0)).squeeze(0)
            if self._cache is not None:
                self._cache[index] = x

        if is_payload and self.relabel:
            y_t = self.trigger.target_label_for(torch.tensor([y]))
            y = int(y_t.item())
        return x, y


def make_asr_view(
    test_set: Dataset,
    trigger: Trigger,
    labels: Sequence[int] | np.ndarray,
) -> "AsrDataset":
    """Build the attack-success-rate evaluation view.

    Applies the trigger to every eligible test sample and relabels to the
    backdoor target. Under ``all2one``, samples whose true label is already the
    target are dropped: the model would classify them correctly with or without
    a backdoor, so counting them measures nothing and inflates ASR by about
    1/C. Under ``all2all`` every class maps somewhere else, so nothing is
    excluded.

    Uses :meth:`Trigger.apply` — the *deployment* form of the trigger. For
    asymmetric attacks like Adaptive-Blend that train with a weakened trigger
    and attack with the full one, evaluating with the training form would
    understate ASR and mark a working attack as a failed test case.
    """
    labels = np.asarray(labels)
    if trigger.label_mode == "all2one":
        eligible = np.flatnonzero(labels != trigger.target_label)
    else:
        eligible = np.arange(len(labels))
    return AsrDataset(test_set, trigger, eligible)


class AsrDataset(Dataset):
    """Triggered, relabelled, target-class-excluded view of a test set."""

    def __init__(self, base: Dataset, trigger: Trigger, eligible: np.ndarray) -> None:
        self.base = base
        self.trigger = trigger
        self.eligible = eligible

    def __len__(self) -> int:
        return len(self.eligible)

    def __getitem__(self, i: int) -> tuple[Tensor, int]:
        x, y = self.base[int(self.eligible[i])]
        x = self.trigger.apply(x.unsqueeze(0)).squeeze(0)
        y_t = self.trigger.target_label_for(torch.tensor([y]))
        return x, int(y_t.item())


@dataclass
class PoisonSpec:
    """Recorded in the zoo manifest as ground truth for a poisoned model."""

    attack: str
    mode: PoisonMode
    requested_rate: float
    achieved_rate: float
    n_poisoned: int
    trigger_config: dict[str, Any]
    n_cover: int = 0
    cover_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "attack": self.attack,
            "mode": self.mode,
            "requested_rate": self.requested_rate,
            "achieved_rate": self.achieved_rate,
            "n_poisoned": self.n_poisoned,
            "n_cover": self.n_cover,
            "cover_rate": self.cover_rate,
            "trigger": self.trigger_config,
        }
