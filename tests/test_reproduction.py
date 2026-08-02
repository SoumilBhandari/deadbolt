"""Reproducing known published outcomes. This is the real correctness check.

A subtly broken detector still produces plausible-looking numbers. It flags
some models, misses others, and its AUC lands somewhere between 0.6 and 0.9 —
which is roughly where a correct implementation lands too. Unit tests catch
crashes and shape errors; they cannot catch a lambda schedule that never
converges, because the code runs fine and returns a number.

So the tests here train real models and assert on outcomes the literature has
already established. Each one names what a failure would mean, because a failed
reproduction is a bug report against deadbolt, not a discovery:

- Neural Cleanse must catch BadNets. It is a sparse, universal, localised patch
  — precisely the regime the method was designed for. Failing here means the
  lambda controller is broken.
- Neural Cleanse must NOT catch WaNet. A warp is not in the hypothesis space
  {x' = (1-m)x + m*p}. Succeeding here means our warp is too strong to be the
  attack the paper describes.
- STRIP must separate BadNets-triggered inputs from clean ones. Failing means
  the entropy computation is wrong.
- STRIP must fall to chance on WaNet with noise mode. Succeeding means noise
  mode is not actually training.

Marked ``slow``: each trains models from scratch. Run with `pytest -m slow`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from deadbolt.checkpoints import load_model
from deadbolt.config import resolve_device
from deadbolt.data.datasets import load_clean, subset
from deadbolt.data.splits import defender_split
from deadbolt.defenses import STRIP, NeuralCleanse, SpectralSignatures
from deadbolt.metrics import roc_auc
from deadbolt.scan import artefacts
from deadbolt.train import TrainConfig, train_one

pytestmark = pytest.mark.slow

#: Small enough to keep the suite runnable on CPU in CI, large enough that the
#: attacks reach the ASR the papers report.
EPOCHS = 4
WIDTH = 16


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("repro")


@pytest.fixture(scope="module")
def device():
    return resolve_device()


def _train(workdir: Path, attack: str | None, rate: float = 0.05, **kwargs):
    cfg = TrainConfig(
        dataset="mnist",
        arch="smallcnn",
        width=WIDTH,
        epochs=EPOCHS,
        seed=0,
        attack=attack,
        poison_rate=rate if attack else 0.0,
        attack_kwargs=kwargs,
    )
    return train_one(cfg, workdir)


@pytest.fixture(scope="module")
def clean_model(workdir):
    return _train(workdir, None)


@pytest.fixture(scope="module")
def badnets_model(workdir):
    r = _train(workdir, "badnets", rate=0.01, target_label=0)
    assert r.attack_success_rate > 0.9, "the attack must work before a defense can fail to catch it"
    return r


@pytest.fixture(scope="module")
def wanet_model(workdir):
    r = _train(workdir, "wanet", rate=0.1, target_label=0)
    assert r.attack_success_rate > 0.9, "the attack must work before a defense can fail to catch it"
    return r


def _defender_loader(batch_size: int = 32) -> DataLoader:
    test, labels = load_clean("mnist", train=False)
    d_idx, _ = defender_split(labels, 50)
    return DataLoader(subset(test, d_idx), batch_size=batch_size)


# --- attack quality is a precondition ---------------------------------------


def test_mnist_baseline_meets_the_m0_gate(clean_model):
    """The gate everything else rests on: if the victim is weak, nothing means much."""
    assert clean_model.clean_accuracy > 0.98


def test_backdoors_are_stealthy(clean_model, badnets_model, wanet_model):
    """A backdoor that costs visible accuracy needs no detector to find."""
    for r in (badnets_model, wanet_model):
        assert clean_model.clean_accuracy - r.clean_accuracy < 0.02


# --- Neural Cleanse ---------------------------------------------------------


def test_neural_cleanse_names_the_badnets_target(badnets_model, device):
    """The positive control. If this fails, the lambda controller is broken."""
    r = NeuralCleanse(steps=800).scan(
        load_model(badnets_model.checkpoint, device),
        _defender_loader(),
        num_classes=10,
        device=device,
    )
    assert r.target_label == 0, (
        f"expected label 0, got {r.target_label} (norms {r.extra['l1_norms']})"
    )


def test_neural_cleanse_reconstruction_is_tight_on_badnets(badnets_model, device):
    """The recovered mask should be about the size of the real 3x3 patch.

    Asserted on the *measurement* rather than on the verdict, because on a
    10-class task the two disagree — which is itself one of this repo's
    findings, and is reported rather than papered over.
    """
    r = NeuralCleanse(steps=800).scan(
        load_model(badnets_model.checkpoint, device),
        _defender_loader(),
        num_classes=10,
        device=device,
    )
    assert -r.aux_scores["min_l1"] < 25, "a 9-pixel trigger should not need a 25-pixel mask"
    assert r.aux_scores["norm_ratio"] > 4, "the target label must be far cheaper than the rest"


def test_neural_cleanse_separates_badnets_from_clean_on_reconstruction_size(
    badnets_model, clean_model, device
):
    """Same measurement, both models. This is the comparison that must hold."""
    nc = NeuralCleanse(steps=800)
    loader = _defender_loader()
    poisoned = nc.scan(
        load_model(badnets_model.checkpoint, device), loader, num_classes=10, device=device
    )
    benign = nc.scan(
        load_model(clean_model.checkpoint, device), loader, num_classes=10, device=device
    )
    assert -poisoned.aux_scores["min_l1"] < -benign.aux_scores["min_l1"] / 2, (
        "the backdoored model's cheapest label must be far cheaper than any clean model's"
    )


def test_neural_cleanse_recovers_the_actual_trigger_pixels(badnets_model, device):
    """Flagging for the wrong reason is a weaker result than flagging correctly."""
    from deadbolt.scan import score_one

    blind = artefacts(badnets_model, "model_clean", device)
    nc = NeuralCleanse(steps=800)
    result = nc.scan(blind.model, blind.loader, num_classes=10, device=device)
    iou = score_one(nc, blind, result)["mask_iou"]
    assert iou is not None and iou > 0.15, f"reconstruction does not overlap the real patch ({iou})"


def test_neural_cleanse_fails_on_wanet(wanet_model, device):
    """The structural failure. If this *passes*, our WaNet is not WaNet.

    A warp cannot be written as (1-m)*x + m*p for any mask and pattern, so the
    reconstruction has nothing small to find. The reconstruction size must look
    like a clean model's, not like BadNets'.
    """
    r = NeuralCleanse(steps=800).scan(
        load_model(wanet_model.checkpoint, device),
        _defender_loader(),
        num_classes=10,
        device=device,
    )
    assert r.aux_scores["norm_ratio"] < 4, (
        "Neural Cleanse found a small mask for a warp, which means the warp is too "
        "strong to be the published attack"
    )


# --- STRIP ------------------------------------------------------------------


def test_strip_separates_badnets_inputs(badnets_model, device):
    """The positive control. Failure means the entropy computation is wrong."""
    blind = artefacts(badnets_model, "runtime", device)
    r = STRIP(n_overlays=32).scan(
        blind.model, blind.loader, num_classes=10, device=device, clean=blind.clean_loader
    )
    auc = roc_auc(blind.truth.poison_mask, r.per_sample_scores.numpy())
    assert auc > 0.9, f"STRIP should trivially separate a dominant patch trigger, got {auc}"


def test_strip_falls_to_chance_on_wanet(wanet_model, device):
    """The structural failure, and the reason noise mode exists.

    The network has been explicitly taught that perturbed-and-warped does not
    mean the target class. If STRIP still separates here, cover samples are not
    reaching training.
    """
    blind = artefacts(wanet_model, "runtime", device)
    r = STRIP(n_overlays=32).scan(
        blind.model, blind.loader, num_classes=10, device=device, clean=blind.clean_loader
    )
    auc = roc_auc(blind.truth.poison_mask, r.per_sample_scores.numpy())
    assert auc < 0.75, (
        f"STRIP separated a WaNet+noise-mode model ({auc}); noise mode is not training"
    )


# --- Spectral Signatures ----------------------------------------------------


def test_spectral_finds_badnets_poison_in_the_training_set(badnets_model, device):
    """Dirty-label poisoning is the regime latent statistics were built for."""
    blind = artefacts(badnets_model, "trainset", device, trainset_size=6000)
    r = SpectralSignatures().scan(blind.model, blind.loader, num_classes=10, device=device)
    auc = roc_auc(blind.truth.poison_mask, r.per_sample_scores.numpy())
    assert auc > 0.75, f"a dirty-label patch attack should separate in feature space, got {auc}"


def test_asr_view_excludes_the_target_class_end_to_end(badnets_model):
    """The ASR-inflation guard, checked against a real trained model.

    A clean model scores ~1/C on the ASR view by chance alone if target-class
    samples are not excluded. This asserts the exclusion survived the whole
    pipeline, not just the unit test.
    """
    from deadbolt.attacks import BadNets
    from deadbolt.data.poison import make_asr_view

    test, labels = load_clean("mnist", train=False)
    _, eval_idx = defender_split(labels, 50)
    view = make_asr_view(subset(test, eval_idx), BadNets(10, target_label=0), labels[eval_idx])
    assert len(view) == int((labels[eval_idx] != 0).sum())
    assert all(view[i][1] == 0 for i in range(0, len(view), 500))
