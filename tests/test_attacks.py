"""Invariants every attack must satisfy, and the ones specific to each.

The shared invariants are not bureaucracy. A trigger that leaves the unit range,
mutates its input, or disagrees with its own ground-truth mask corrupts results
several modules downstream, in ways that look like defense behaviour rather than
like bugs.
"""

from __future__ import annotations

import pytest
import torch

from deadbolt.attacks import (
    ATTACKS,
    CLEAN_LABEL_ATTACKS,
    SIG,
    AdaptiveBlend,
    BadNets,
    Blended,
    LabelConsistent,
    WaNet,
)
from deadbolt.checkpoints import build_model


def make(name: str, **kw):
    """Instantiate an attack, wiring a surrogate for the ones that need one."""
    trigger = ATTACKS[name](num_classes=10, **kw)
    if trigger.requires_surrogate:
        trigger.prepare(build_model("smallcnn", "cifar10", width=4), torch.device("cpu"))
    return trigger


@pytest.fixture
def batch() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(4, 3, 32, 32, generator=g)


@pytest.mark.parametrize("name", sorted(ATTACKS))
@pytest.mark.parametrize("path", ["apply", "apply_train", "apply_cover"])
def test_output_stays_in_unit_range(name, path, batch):
    """Triggers live in pixel space; leaving [0, 1] silently changes the task."""
    out = getattr(make(name), path)(batch)
    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6


@pytest.mark.parametrize("name", sorted(ATTACKS))
@pytest.mark.parametrize("path", ["apply", "apply_train", "apply_cover"])
def test_does_not_mutate_input(name, path, batch):
    """The pipeline reuses the clean batch; in-place edits corrupt it."""
    before = batch.clone()
    getattr(make(name), path)(batch)
    assert torch.equal(batch, before)


@pytest.mark.parametrize("name", sorted(ATTACKS))
@pytest.mark.parametrize("path", ["apply", "apply_train", "apply_cover"])
def test_preserves_shape(name, path, batch):
    assert getattr(make(name), path)(batch).shape == batch.shape


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_rejects_unbatched_input(name):
    """Triggers implement only the batched path; a 3D tensor must not silently work."""
    with pytest.raises(ValueError, match="B, C, H, W"):
        make(name).apply(torch.rand(3, 32, 32))


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_actually_changes_the_image(name, batch):
    """A trigger that is a no-op would train a clean model and be recorded as poisoned."""
    assert not torch.equal(make(name).apply(batch), batch)


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_config_round_trips(name, batch):
    """The manifest must be able to rebuild the exact trigger it recorded."""
    original = make(name)
    cfg = original.config()
    kwargs = {k: v for k, v in cfg.items() if k not in {"name", "num_classes"}}
    kwargs.pop("target_label", None)
    kwargs.pop("cover_rate", None)
    kwargs.pop("noise_rate", None)
    rebuilt = ATTACKS[name](num_classes=10, target_label=cfg["target_label"] or 0, **kwargs)
    if rebuilt.requires_surrogate:
        pytest.skip("surrogate-dependent output is not a property of the config alone")
    assert torch.equal(rebuilt.apply(batch), original.apply(batch))


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_mask_agrees_with_modified_pixels(name, batch):
    """A mask that disagrees with apply() corrupts every IoU score silently."""
    trigger = make(name)
    mask = trigger.ground_truth_mask((32, 32))
    if mask is None:
        pytest.skip("no localised support: IoU is undefined, which is the point")
    changed = (trigger.apply(batch) != batch).any(dim=0).any(dim=0)
    # The mask is the trigger's *support*, so every changed pixel must be in it.
    # The converse need not hold: a patch pixel can coincide with the image
    # value it overwrites.
    assert (changed & ~mask.squeeze(0).bool()).sum() == 0


def test_clean_label_attacks_refuse_all2all():
    """There is no such thing as clean-label poisoning towards a shifted label."""
    for name in CLEAN_LABEL_ATTACKS:
        with pytest.raises(ValueError, match="all2one"):
            ATTACKS[name](num_classes=10, label_mode="all2all")


def test_blended_alpha_controls_perturbation_size():
    a, b = Blended(10, alpha=0.05), Blended(10, alpha=0.5)
    x = torch.rand(4, 3, 32, 32)
    assert (a.apply(x) - x).abs().mean() < (b.apply(x) - x).abs().mean()


def test_blended_pattern_is_seed_reproducible():
    """The manifest records a seed, not an image; the trigger must follow from it."""
    x = torch.rand(2, 3, 32, 32)
    assert torch.equal(Blended(10, pattern_seed=7).apply(x), Blended(10, pattern_seed=7).apply(x))
    assert not torch.equal(
        Blended(10, pattern_seed=7).apply(x), Blended(10, pattern_seed=8).apply(x)
    )


def test_sig_is_a_horizontal_wave():
    """Every row gets the same perturbation; that is what makes it a column wave.

    Applied to mid-grey rather than to zeros: the wave is symmetric about zero
    and clamping at 0 would flatten its whole negative half, which makes the
    columns agree for a reason that has nothing to do with orientation.
    """
    x = torch.full((1, 1, 16, 16), 0.5)
    delta = (SIG(10, frequency=1).apply(x) - x)[0, 0]
    assert torch.allclose(delta[0], delta[-1]), "rows must be identical"
    assert not torch.allclose(delta[:, 4], delta[:, 12]), "columns must differ"


def test_sig_vertical_transposes_the_wave():
    x = torch.full((1, 1, 16, 16), 0.5)
    delta = (SIG(10, frequency=1, vertical=True).apply(x) - x)[0, 0]
    assert torch.allclose(delta[:, 0], delta[:, -1]), "columns must be identical"
    assert not torch.allclose(delta[4], delta[12]), "rows must differ"


def test_wanet_warp_is_deterministic_but_cover_is_not():
    """The payload warp *is* the trigger. Noise mode must vary, or it teaches nothing."""
    w = WaNet(10)
    x = torch.rand(4, 3, 32, 32)
    assert torch.equal(w.apply(x), w.apply(x))
    assert not torch.equal(w.apply_cover(x), w.apply_cover(x))


def test_wanet_requests_cover_samples():
    """Noise mode is not optional decoration: without cover, STRIP catches WaNet."""
    assert WaNet(10).default_cover_rate > 0


def test_wanet_strength_scales_with_s():
    x = torch.rand(4, 3, 32, 32)
    weak = (WaNet(10, s=0.1).apply(x) - x).abs().mean()
    strong = (WaNet(10, s=1.0).apply(x) - x).abs().mean()
    assert weak < strong


def test_adaptive_blend_is_asymmetric():
    """Training sees a fragment of the trigger; the attack uses all of it.

    If these were equal the attack would be an ordinary blend, and the latent
    separability it is designed to suppress would come straight back.
    """
    t = AdaptiveBlend(10, grid=4, train_pieces=8)
    x = torch.rand(8, 3, 32, 32)
    assert (t.apply(x) - x).abs().mean() > (t.apply_train(x) - x).abs().mean()


def test_adaptive_blend_train_masks_vary_per_sample():
    """One shared mask would just be a blend with a fixed hole pattern."""
    t = AdaptiveBlend(10, grid=4, train_pieces=8)
    x = torch.rand(2, 3, 32, 32).repeat(1, 1, 1, 1)
    x[1] = x[0]  # identical images
    out = t.apply_train(x)
    assert not torch.equal(out[0], out[1])


def test_adaptive_blend_requests_cover_samples():
    assert AdaptiveBlend(10).default_cover_rate > 0


def test_label_consistent_needs_a_surrogate():
    """Silently skipping PGD degrades it to a weak patch attack under the wrong name."""
    t = LabelConsistent(10)
    with pytest.raises(RuntimeError, match="surrogate"):
        t.apply_train(torch.rand(2, 3, 32, 32))


def test_label_consistent_attack_form_is_the_patch_alone():
    """The adversary does not get to run PGD on the victim's camera feed.

    Applying the perturbation in the ASR view would inflate ASR, because the
    perturbation is itself pushing the model off the true class.
    """
    t = make("label_consistent")
    x = torch.rand(4, 3, 32, 32)
    mask = t.ground_truth_mask((32, 32)).bool()
    changed = (t.apply(x) != x).any(dim=0).any(dim=0)
    assert (changed & ~mask.squeeze(0)).sum() == 0


def test_label_consistent_perturbation_respects_epsilon():
    t = make("label_consistent", epsilon=8 / 255)
    x = torch.rand(4, 3, 32, 32)
    perturbed = t._perturb(x)
    assert (perturbed - x).abs().max() <= 8 / 255 + 1e-5


def test_label_consistent_is_marked_expensive():
    """Recomputing PGD per epoch is both slow and a different attack each epoch."""
    assert LabelConsistent(10).expensive


def test_badnets_patch_position_is_configurable():
    for corner, cell in (("br", (-1, -1)), ("tl", (0, 0))):
        mask = BadNets(10, patch_size=2, margin=0, corner=corner).ground_truth_mask((8, 8))
        assert mask[0, cell[0], cell[1]] == 1.0
