"""Fine-pruning, and the accuracy budget that makes it honest.

The failure mode a mitigation test has to guard against is not a crash. It is a
method that reports a spectacular ASR reduction because it destroyed the
network, and a harness that prints only the ASR column.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from deadbolt.checkpoints import build_model
from deadbolt.mitigate import MitigationResult, channel_activations, fine_prune
from deadbolt.models.base import BackdoorModel


@pytest.fixture
def model():
    return build_model("smallcnn", "cifar10", width=8).eval()


def test_channel_activations_are_per_channel_and_finite(model, toy_loader, cpu):
    acts = channel_activations(model, toy_loader, cpu)
    assert acts.shape == (model.channel_mask.shape[1],)
    assert torch.isfinite(acts).all()


def test_channel_activations_ignore_the_pruning_mask(model, toy_loader, cpu):
    """Profiling through the mask being written makes each round's ranking
    depend on the previous round's decisions, and the ranking drifts towards
    whatever was pruned first."""
    before = channel_activations(model, toy_loader, cpu)
    model.prune(torch.arange(4))
    after = channel_activations(model, toy_loader, cpu)
    assert torch.allclose(before, after)


def test_pruning_a_channel_zeroes_its_contribution(model, cpu):
    x = torch.rand(4, 3, 32, 32)
    before = model.penultimate(x)
    model.prune(torch.tensor([0, 1]))
    after = model.penultimate(x)
    assert (after[:, :2] == 0).all()
    assert torch.allclose(after[:, 2:], before[:, 2:])


def test_fine_prune_stops_inside_the_accuracy_budget(model, toy_loader, cpu):
    """The knob that makes this a defense rather than a way to delete a network.

    Without it, any method can drive ASR to zero. The loop must stop as soon as
    clean accuracy has fallen by more than the budget, and must not spend its
    whole prune_fraction regardless.
    """
    schedule = iter([0.99, 0.98, 0.90, 0.10, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    seen = []

    def eval_fn(m, clean_only: bool = False):
        acc = next(schedule, 0.0)
        seen.append(acc)
        return acc, (None if clean_only else 0.5)

    result = fine_prune(
        model, toy_loader, eval_fn, cpu, prune_fraction=1.0, finetune_epochs=0, max_clean_drop=0.04
    )
    total = result.extra["total_channels"]
    assert result.extra["pruned_channels"] < total, "must not prune everything"
    assert len(result.extra["prune_curve"]) < 10, "must stop early, not run the full schedule"


def test_fine_prune_reports_cost_alongside_benefit(model, toy_loader, cpu):
    def eval_fn(m, clean_only: bool = False):
        return 0.90, (None if clean_only else 0.10)

    result = fine_prune(model, toy_loader, eval_fn, cpu, finetune_epochs=0)
    row = result.as_record()
    assert "asr_reduction" in row and "clean_cost" in row, (
        "an ASR reduction reported without its clean-accuracy price is the "
        "specific dishonesty this record exists to prevent"
    )


def test_pruned_count_is_read_from_the_mask_not_accumulated(model, toy_loader, cpu):
    """The reported count must describe the model, not this call's bookkeeping.

    prune() is cumulative and idempotent per channel, so a running counter
    drifts from the mask in both directions: it undercounts a model that
    arrived already pruned, and overcounts if a channel is pruned twice. Only
    the mask decides what the network actually computes.
    """

    def eval_fn(m, clean_only: bool = False):
        return 0.90, (None if clean_only else 0.10)

    # Arrive with three channels already gated off, as a second mitigation pass
    # or a pre-pruned checkpoint would.
    model.prune(torch.tensor([0, 1, 2]))
    result = fine_prune(model, toy_loader, eval_fn, cpu, finetune_epochs=0)

    assert result.extra["pruned_channels"] == model.n_pruned
    assert result.extra["pruned_channels"] >= 3, "pre-existing pruning was dropped"
    assert result.extra["pruned_channels"] <= result.extra["total_channels"]
    # The curve is what the honesty story is told from; it must agree too.
    assert result.extra["prune_curve"][-1]["pruned"] == model.n_pruned


def test_fine_prune_prunes_quietest_channels_first(toy_loader, cpu):
    """The dormant-channel premise: the backdoor lives where clean data does not."""

    class Rigged(BackdoorModel):
        """Channel 0 is silent on clean data; the rest are loud.

        Subclasses the real contract rather than re-implementing it. A
        hand-rolled double drifts silently the moment BackdoorModel grows a
        member fine_prune relies on, and the test then passes against an
        interface the production models do not have.
        """

        def __init__(self):
            super().__init__()
            self.normalize = nn.Identity()
            self.head = nn.Linear(4, 2)
            self._init_channel_mask(4)
            self.pruned: list[int] = []

        def _feature_maps(self, x):
            maps = torch.ones(x.shape[0], 4, 2, 2)
            maps[:, 0] = 0.0
            return maps

        def prune(self, channels):
            self.pruned.extend(int(c) for c in channels)
            super().prune(channels)

    rigged = Rigged()
    result = fine_prune(
        rigged,
        toy_loader,
        lambda m, clean_only=False: (0.9, None if clean_only else 0.1),
        cpu,
        prune_fraction=0.25,
        finetune_epochs=0,
    )
    assert rigged.pruned[0] == 0, "the silent channel must go first"
    assert result.method == "fine_pruning"


def test_mitigation_result_arithmetic():
    r = MitigationResult(
        method="m",
        clean_accuracy_before=0.94,
        clean_accuracy_after=0.79,
        asr_before=0.99,
        asr_after=0.03,
        runtime_s=1.0,
    )
    assert r.asr_reduction == pytest.approx(0.96)
    assert r.clean_cost == pytest.approx(0.15)
    # A method that removes the backdoor by destroying the model must be legible
    # as such from the record alone.
    assert r.clean_cost > 0.1


def test_channel_activations_refuses_an_empty_loader(model, cpu):
    empty = DataLoader([], batch_size=4)
    with pytest.raises(ValueError, match="empty loader"):
        channel_activations(model, empty, cpu)
