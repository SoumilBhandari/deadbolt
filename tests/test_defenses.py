"""Detector contracts, and the structural failures each one is supposed to have.

Two kinds of test here. The first checks that every detector honours the shared
contract, so the harness can treat them uniformly. The second checks specific
mechanisms — Neural Cleanse's lambda controller, STRIP's entropy — on inputs
where the right answer is known by construction rather than by training a model.

The "does this defense catch that attack" questions live in
`test_reproduction.py`, because answering them honestly requires real models.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from deadbolt.attacks import BadNets
from deadbolt.defenses import (
    DEFENSES,
    STRIP,
    ActivationClustering,
    NeuralCleanse,
    SpectralSignatures,
)
from deadbolt.defenses.base import DetectionResult, Detector
from deadbolt.defenses.features import penultimate_features


@pytest.mark.parametrize("name", sorted(DEFENSES))
def test_declares_a_coherent_threat_model(name):
    d = DEFENSES[name]()
    assert d.access in {"model_clean", "trainset", "runtime", "weights_only"}
    # A detector must claim *something*, or the harness has nothing to score.
    assert d.produces_model_verdict or d.produces_sample_verdict
    if d.recovers_mask:
        assert d.access in {"model_clean", "weights_only"}, "mask recovery needs the weights"


@pytest.mark.parametrize("name", sorted(DEFENSES))
def test_config_is_json_serialisable(name):
    """Every scan row records the detector config; an unserialisable one loses it."""
    import json

    json.dumps(DEFENSES[name]().config())


@pytest.mark.parametrize("name", sorted(DEFENSES))
def test_scan_returns_a_populated_result(name, tiny_model, toy_loader, cpu):
    d = DEFENSES[name]()
    kwargs = {"clean": toy_loader} if d.access == "runtime" else {}
    r = d.scan(tiny_model, toy_loader, num_classes=10, device=cpu, **kwargs)
    assert isinstance(r, DetectionResult)
    assert np.isfinite(r.score), "a non-finite score would be refused by the scorer"
    assert r.runtime_s > 0, "timing is mandatory: cost is a headline comparison"
    if d.identifies_target:
        assert r.target_label is not None and 0 <= r.target_label < 10
    if d.produces_sample_verdict:
        assert r.per_sample_scores is not None
        assert len(r.per_sample_scores) == len(toy_loader.dataset)


def test_strip_refuses_to_run_without_trusted_data():
    """Overlays are STRIP's threat model; silently skipping them is a different method."""
    with pytest.raises(ValueError, match="clean loader"):
        STRIP().scan(nn.Identity(), None, num_classes=10, device=torch.device("cpu"))


def test_strip_scores_are_negated_entropy(tiny_model, toy_loader, cpu):
    """Higher must always mean more suspicious, across every detector."""
    r = STRIP(n_overlays=4).scan(
        tiny_model, toy_loader, num_classes=10, device=cpu, clean=toy_loader
    )
    assert r.extra["mean_entropy"] == pytest.approx(-r.score, abs=1e-4)


def test_strip_flags_a_model_that_ignores_its_input(toy_loader, cpu):
    """The mechanism, isolated: a constant predictor has zero entropy everywhere.

    That is the limiting case of a perfectly dominant trigger, and STRIP must
    report it as maximally suspicious. If this fails, the entropy computation is
    wrong and nothing downstream can be trusted.
    """

    class Constant(nn.Module):
        def forward(self, x):
            out = torch.full((x.shape[0], 10), -20.0)
            out[:, 3] = 20.0
            return out

    r = STRIP(n_overlays=4).scan(
        Constant(), toy_loader, num_classes=10, device=cpu, clean=toy_loader
    )
    assert r.extra["mean_entropy"] < 1e-6


def test_neural_cleanse_recovers_a_planted_shortcut(cpu):
    """The mechanism, isolated from training.

    A hand-built model that outputs class 3 whenever the bottom-right 3x3 corner
    is bright is a backdoor with a known trigger. Neural Cleanse must find a
    small mask for label 3 and a large one for everything else. This tests the
    lambda controller and the reconstruction loop without a single training
    step, so a failure here is unambiguously the detector's.
    """

    class Planted(nn.Module):
        """Every class needs the whole image rewritten — except class 3.

        A nearest-prototype classifier: reaching class j means making the image
        resemble prototype j, which no small mask can do. Class 3 additionally
        responds to the bottom-right 3x3 corner, so it is reachable by changing
        nine pixels. That is a backdoor with a trigger known exactly, and
        Neural Cleanse must find it without a single training step — so a
        failure here is unambiguously the detector's.

        A linear model would not do: with a mask free to set pixels to 1, every
        class ends up cheaply reachable by lighting up its own high-weight
        pixels, and all ten labels get similar norms.
        """

        def __init__(self) -> None:
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.prototypes = torch.rand(10, 3, 16, 16, generator=g)

        def forward(self, x):
            dist = ((x.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2).sum(dim=(2, 3, 4))
            corner = x[:, :, -4:-1, -4:-1].mean(dim=(1, 2, 3))
            boost = torch.zeros_like(dist)
            boost[:, 3] = 40.0 * corner
            return -dist + boost

    images = torch.rand(64, 3, 16, 16) * 0.4  # dim: the corner is not bright by chance
    loader = DataLoader(
        list(zip(images, torch.zeros(64, dtype=torch.long), strict=True)), batch_size=32
    )
    r = NeuralCleanse(steps=400, window=10).scan(Planted(), loader, num_classes=10, device=cpu)

    norms = r.extra["l1_norms"]
    assert r.target_label == 3, f"must name the planted class, got {r.target_label} (norms {norms})"
    assert norms[3] < 0.5 * np.median(norms), "the planted label needs a far smaller mask"
    assert r.aux_scores["norm_ratio"] > 2.0


def test_neural_cleanse_mask_is_binary_and_shaped_for_iou(tiny_model, toy_loader, cpu):
    """(1, H, W), matching Trigger.ground_truth_mask so mask_iou can compare them."""
    r = NeuralCleanse(steps=40).scan(tiny_model, toy_loader, num_classes=10, device=cpu)
    mask = r.recovered_mask
    assert mask.shape == BadNets(10).ground_truth_mask((8, 8)).shape == (1, 8, 8)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_neural_cleanse_records_both_statistics(tiny_model, toy_loader, cpu):
    """score is the paper's; aux_scores keeps the measurement it was derived from."""
    r = NeuralCleanse(steps=40).scan(tiny_model, toy_loader, num_classes=10, device=cpu)
    assert set(r.aux_scores) == {"norm_ratio", "min_l1"}
    assert all(np.isfinite(v) for v in r.aux_scores.values())


def test_spectral_finds_a_planted_subpopulation(cpu):
    """The mechanism, isolated: a class with two feature clusters must separate."""

    class TwoClusters(nn.Module):
        """Feature extractor that maps the first channel to a 1-D signal."""

        def penultimate(self, x):
            return torch.stack([x.mean(dim=(1, 2, 3)) * 10, torch.zeros(x.shape[0])], dim=1)

    n = 200
    x = torch.rand(n, 1, 4, 4) * 0.1
    x[:10] += 0.9  # 5% of the class sits far away
    loader = DataLoader(list(zip(x, torch.zeros(n, dtype=torch.long), strict=True)), batch_size=64)
    r = SpectralSignatures(assumed_rate=0.05).scan(TwoClusters(), loader, num_classes=1, device=cpu)
    scores = r.per_sample_scores
    assert scores[:10].mean() > 10 * scores[10:].mean()
    assert r.score > 5


def test_activation_clustering_ignores_balanced_splits(cpu):
    """A 50/50 split with a great silhouette is a bimodal class, not an implant."""

    class TwoHalves(nn.Module):
        def penultimate(self, x):
            # Two live dimensions: one carrying the split, one pure noise. A
            # constant second dimension would be dropped as dead and the class
            # skipped, which is a different code path from the one under test.
            g = torch.Generator().manual_seed(1)
            return torch.stack(
                [x.mean(dim=(1, 2, 3)) * 50, torch.rand(x.shape[0], generator=g)], dim=1
            )

    n = 200
    x = torch.rand(n, 1, 4, 4) * 0.01
    x[: n // 2] += 1.0  # perfectly separable, but half the class
    loader = DataLoader(list(zip(x, torch.zeros(n, dtype=torch.long), strict=True)), batch_size=64)
    r = ActivationClustering(max_ratio=0.35).scan(TwoHalves(), loader, num_classes=1, device=cpu)
    assert r.extra["minority_ratio"][0] == pytest.approx(0.5, abs=0.02)
    assert not r.is_backdoored, "an even split is class structure, not poison"


def test_features_come_back_on_cpu_in_float64(tiny_model, toy_loader, cpu):
    """MPS has no float64; the cast must happen after the transfer, not before."""
    feats, labels = penultimate_features(tiny_model, toy_loader, cpu)
    assert feats.dtype == torch.float64 and feats.device.type == "cpu"
    assert len(feats) == len(labels) == len(toy_loader.dataset)


def test_features_preserve_dataset_order(tiny_model, cpu):
    """Per-sample scores are matched to the poison mask by position."""
    from tests.conftest import ToyDataset

    ds = ToyDataset(n=64)
    _, labels = penultimate_features(tiny_model, DataLoader(ds, batch_size=8, shuffle=False), cpu)
    assert torch.equal(labels, torch.tensor(ds.labels))


def test_timer_records_even_when_the_body_raises():
    """A crashed scan still cost wall-clock, and the record should say how much."""

    class Boom(Detector):
        name = "boom"

        def scan(self, model, data, **kwargs):
            raise RuntimeError

    holder: dict[str, float] = {}
    with pytest.raises(RuntimeError), Boom().timer(holder):
        raise RuntimeError
    assert "runtime_s" in holder


def test_karm_and_neural_cleanse_share_one_optimiser():
    """K-Arm's claim is about *scheduling*, and is only measurable if the
    optimisation underneath is literally the same code.

    Two separate implementations would differ in initialisation, in the lambda
    controller, in early stopping — and any cost gap between them would be
    unattributable to the scheduler.
    """
    import inspect

    from deadbolt.defenses import karm, neural_cleanse
    from deadbolt.defenses.reconstruct import TriggerOptimizer

    assert "TriggerOptimizer" in inspect.getsource(neural_cleanse)
    assert "TriggerOptimizer" in inspect.getsource(karm)
    assert karm.TriggerOptimizer is neural_cleanse.TriggerOptimizer is TriggerOptimizer


def test_karm_spends_its_whole_budget_and_no_more(tiny_model, toy_loader, cpu):
    """Budget is the unit of comparison against Neural Cleanse; overrunning it
    would make K-Arm look cheap while quietly costing more."""
    from deadbolt.defenses import KArm

    budget = 400
    r = KArm(budget=budget, warmup=10, round_steps=20).scan(
        tiny_model, toy_loader, num_classes=10, device=cpu
    )
    assert sum(r.extra["steps_per_label"]) == budget


def test_karm_concentrates_budget_unevenly(cpu):
    """The whole point: an equal split is Neural Cleanse, not K-Arm."""
    from deadbolt.defenses import KArm

    class Planted(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.prototypes = torch.rand(10, 3, 16, 16, generator=g)

        def forward(self, x):
            dist = ((x.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2).sum(dim=(2, 3, 4))
            corner = x[:, :, -4:-1, -4:-1].mean(dim=(1, 2, 3))
            boost = torch.zeros_like(dist)
            boost[:, 3] = 40.0 * corner
            return -dist + boost

    images = torch.rand(64, 3, 16, 16) * 0.4
    loader = DataLoader(
        list(zip(images, torch.zeros(64, dtype=torch.long), strict=True)), batch_size=32
    )
    r = KArm(budget=1200, warmup=30, round_steps=30).scan(
        Planted(), loader, num_classes=10, device=cpu
    )
    steps = r.extra["steps_per_label"]
    assert max(steps) > 2 * min(steps), "budget must be reallocated, not split evenly"
    assert r.target_label == 3, f"must still name the planted class (steps {steps})"


def test_spectre_beats_spectral_when_poison_hides_across_directions(cpu):
    """The reason SPECTRE exists, isolated from any trained model.

    Spectral Signatures projects onto a single top direction. Poison spread
    across several moderate directions dominates none of them and is missed.
    SPECTRE whitens first and scores with QUE, so evidence spread over
    directions still accumulates.
    """
    from deadbolt.defenses import SPECTRE, SpectralSignatures

    torch.manual_seed(0)
    n, d, n_poison = 400, 8, 20
    # Clean cloud with strongly anisotropic covariance: one direction carries
    # most of the variance, which is what the top singular vector will find.
    scale = torch.tensor([8.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    x = torch.randn(n, d, dtype=torch.float64) * scale
    # Poison sits off-distribution along the *low*-variance directions only.
    x[:n_poison, 2:] += 4.0
    x[:n_poison, :2] = torch.randn(n_poison, 2, dtype=torch.float64) * scale[:2]

    truth = torch.zeros(n, dtype=torch.bool)
    truth[:n_poison] = True

    from deadbolt.metrics import roc_auc

    spectral_auc = roc_auc(truth.numpy(), SpectralSignatures.class_scores(x).numpy())
    spectre_auc = roc_auc(truth.numpy(), SPECTRE(assumed_rate=0.05).que_scores(x).numpy())
    assert spectre_auc > spectral_auc, (
        f"SPECTRE {spectre_auc:.3f} should beat Spectral {spectral_auc:.3f} when the "
        "poison direction is not the dominant one"
    )
    assert spectre_auc > 0.9


def test_spectre_robust_moments_ignore_the_poisoned_tail(cpu):
    """The poison must not define the geometry used to find it."""
    from deadbolt.defenses import SPECTRE

    torch.manual_seed(0)
    x = torch.randn(300, 4, dtype=torch.float64)
    x[:30] += 20.0  # a large, far-away poisoned group

    det = SPECTRE(assumed_rate=0.1)
    mean, _ = det._robust_moments(x)
    naive = x.mean(0)
    clean = x[30:].mean(0)
    assert (mean - clean).abs().max() < (naive - clean).abs().max()
