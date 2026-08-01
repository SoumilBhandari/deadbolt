"""Invariants of the poisoning pipeline.

These target the specific mistakes that silently inflate backdoor results,
rather than checking that the code merely runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from deadbolt.attacks import BadNets
from deadbolt.data.poison import (
    PoisonedDataset,
    make_asr_view,
    select_poison_indices,
)


class ToyDataset(Dataset):
    """Deterministic stand-in: 100 samples, 10 classes, 3x8x8 images."""

    def __init__(self, n: int = 100, num_classes: int = 10) -> None:
        self.n = n
        self.labels = np.arange(n) % num_classes
        torch.manual_seed(0)
        self.images = torch.rand(n, 3, 8, 8)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.images[i], int(self.labels[i])


@pytest.fixture
def toy() -> ToyDataset:
    return ToyDataset()


def test_trigger_stays_in_unit_range(toy):
    t = BadNets(num_classes=10, patch_size=3, margin=1)
    out = t.apply(toy.images)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_trigger_does_not_mutate_input(toy):
    """The poisoning pipeline reuses the clean batch; in-place edits corrupt it."""
    t = BadNets(num_classes=10)
    before = toy.images.clone()
    t.apply(toy.images)
    assert torch.equal(toy.images, before)


def test_ground_truth_mask_matches_modified_pixels(toy):
    """A mask that disagrees with apply() would silently corrupt every IoU score."""
    t = BadNets(num_classes=10, patch_size=3, margin=1, checkerboard=False)
    changed = (t.apply(toy.images) != toy.images).any(dim=0).any(dim=0).float()
    mask = t.ground_truth_mask((8, 8)).squeeze(0)
    assert torch.equal(changed, mask)


def test_patch_respects_margin():
    """margin>0 keeps the patch off the border, out of random-crop's reach."""
    t = BadNets(num_classes=10, patch_size=2, margin=2, corner="br")
    mask = t.ground_truth_mask((8, 8)).squeeze(0)
    assert mask[-1, :].sum() == 0 and mask[:, -1].sum() == 0
    # margin=2 reserves the last two rows/cols, so a 2px patch sits at [4:6, 4:6].
    assert mask[4:6, 4:6].sum() == 4
    assert mask.sum() == 4, "no pixels outside the patch"


def test_patch_that_does_not_fit_is_rejected():
    t = BadNets(num_classes=10, patch_size=8, margin=2)
    with pytest.raises(ValueError, match="does not fit"):
        t.apply(torch.rand(1, 3, 8, 8))


@pytest.mark.parametrize("rate", [0.0, 0.01, 0.1, 0.5])
def test_dirty_label_poisons_exact_count(toy, rate):
    idx = select_poison_indices(
        toy.labels, rate, "dirty_label", BadNets(num_classes=10), seed=0
    )
    assert len(idx) == int(np.floor(rate * len(toy.labels)))
    assert len(set(idx.tolist())) == len(idx)


def test_poison_selection_is_reproducible(toy):
    t = BadNets(num_classes=10)
    a = select_poison_indices(toy.labels, 0.1, "dirty_label", t, seed=42)
    b = select_poison_indices(toy.labels, 0.1, "dirty_label", t, seed=42)
    c = select_poison_indices(toy.labels, 0.1, "dirty_label", t, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_clean_label_only_touches_target_class(toy):
    """The defining property: every poisoned sample keeps a correct label."""
    t = BadNets(num_classes=10, target_label=3)
    idx = select_poison_indices(toy.labels, 0.05, "clean_label", t, seed=0)
    assert set(toy.labels[idx].tolist()) == {3}


def test_clean_label_caps_at_target_class_size(toy):
    """Requesting more than the class holds is a real constraint, not an error."""
    t = BadNets(num_classes=10, target_label=3)
    idx = select_poison_indices(toy.labels, 0.9, "clean_label", t, seed=0)
    assert len(idx) == int((toy.labels == 3).sum())


def test_clean_label_rejects_all2all(toy):
    t = BadNets(num_classes=10, label_mode="all2all")
    with pytest.raises(ValueError, match="all2one"):
        select_poison_indices(toy.labels, 0.05, "clean_label", t, seed=0)


def test_dirty_label_relabels_only_poisoned_samples(toy):
    t = BadNets(num_classes=10, target_label=7)
    idx = select_poison_indices(toy.labels, 0.1, "dirty_label", t, seed=0)
    ds = PoisonedDataset(toy, t, idx, relabel=True)
    for i in range(len(ds)):
        _, y = ds[i]
        assert y == (7 if ds.is_poisoned(i) else int(toy.labels[i]))


def test_clean_label_preserves_all_labels(toy):
    t = BadNets(num_classes=10, target_label=3)
    idx = select_poison_indices(toy.labels, 0.05, "clean_label", t, seed=0)
    ds = PoisonedDataset(toy, t, idx, relabel=False)
    for i in range(len(ds)):
        _, y = ds[i]
        assert y == int(toy.labels[i]), "clean-label poisoning must not relabel"


def test_asr_view_excludes_target_class(toy):
    """The ASR-inflation guard.

    Counting samples already of the target class would credit the backdoor for
    predictions any clean model makes, inflating ASR by roughly 1/C.
    """
    t = BadNets(num_classes=10, target_label=4)
    view = make_asr_view(toy, t, toy.labels)
    assert len(view) == int((toy.labels != 4).sum())
    for i in range(len(view)):
        _, y = view[i]
        assert y == 4


def test_asr_view_keeps_everything_under_all2all(toy):
    """all2all maps every class elsewhere, so nothing is trivially correct."""
    t = BadNets(num_classes=10, label_mode="all2all")
    view = make_asr_view(toy, t, toy.labels)
    assert len(view) == len(toy)
    for i in range(len(view)):
        _, y = view[i]
        assert y == (int(toy.labels[i]) + 1) % 10


def test_all2all_shifts_labels():
    t = BadNets(num_classes=10, label_mode="all2all")
    y = torch.arange(10)
    assert torch.equal(t.target_label_for(y), torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))


def test_target_label_must_be_in_range():
    with pytest.raises(ValueError, match="outside"):
        BadNets(num_classes=10, target_label=10)
