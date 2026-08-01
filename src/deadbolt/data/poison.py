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
    """Choose which training samples to poison.

    Dirty-label attacks poison samples drawn from *any* class and rewrite the
    label, so the poisoned examples visibly disagree with their content — that
    disagreement is exactly what latent-statistics defenses key on.

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
        pool = np.arange(n)
    else:
        raise ValueError(f"unknown poison mode {mode!r}")

    return np.sort(rng.choice(pool, size=n_poison, replace=False))


class PoisonedDataset(Dataset):
    """Wraps a clean dataset, applying a trigger to a fixed index set.

    Args:
        base: Clean dataset yielding ``(image, label)`` with image a float
            tensor in ``[0, 1]``.
        trigger: The attack.
        poison_indices: Which base indices to poison.
        relabel: Whether to rewrite labels of poisoned samples. ``True`` for
            dirty-label training and for ASR evaluation; ``False`` for
            clean-label training, where the whole point is that labels stay
            correct.
    """

    def __init__(
        self,
        base: Dataset,
        trigger: Trigger,
        poison_indices: np.ndarray | Sequence[int],
        relabel: bool,
    ) -> None:
        self.base = base
        self.trigger = trigger
        self.relabel = relabel
        self._poison = set(int(i) for i in poison_indices)

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def is_poisoned(self, index: int) -> bool:
        """Ground truth for scoring data-level detectors. Never exposed to them."""
        return index in self._poison

    @property
    def poison_indices(self) -> np.ndarray:
        return np.array(sorted(self._poison), dtype=np.int64)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        x, y = self.base[index]
        if index not in self._poison:
            return x, y

        # apply() takes a batch; unsqueeze and squeeze around it so triggers
        # only ever implement the batched path.
        x = self.trigger.apply(x.unsqueeze(0)).squeeze(0)
        if self.relabel:
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "attack": self.attack,
            "mode": self.mode,
            "requested_rate": self.requested_rate,
            "achieved_rate": self.achieved_rate,
            "n_poisoned": self.n_poisoned,
            "trigger": self.trigger_config,
        }
