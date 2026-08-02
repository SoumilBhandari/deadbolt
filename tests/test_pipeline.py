"""Splits, checkpoints, zoo expansion, and the scan harness's data plumbing.

These are the joints between modules, which is where a benchmark quietly breaks:
each piece is individually correct and the composition is not.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from deadbolt.attacks import AdaptiveBlend, BadNets, WaNet
from deadbolt.checkpoints import build_model, load_model, roundtrip_error, save_model
from deadbolt.data.poison import PoisonedDataset, plan_poisoning
from deadbolt.data.splits import defender_split, stratified_indices
from deadbolt.scan import _stratified_prefix
from deadbolt.train import TrainConfig, TrainResult, apply_stealth_filter, build_trigger
from deadbolt.zoo import AttackSweep, ZooSpec, summarise

# --- splits -----------------------------------------------------------------


def test_stratified_split_is_balanced_and_disjoint():
    labels = np.repeat(np.arange(10), 100)
    held, rest = stratified_indices(labels, 5, seed=0)
    assert len(held) == 50
    counts = np.bincount(labels[held], minlength=10)
    assert (counts == 5).all(), "an unbalanced defender set makes some labels reconstruct worse"
    assert set(held).isdisjoint(rest)
    assert len(held) + len(rest) == len(labels)


def test_stratified_split_takes_what_rare_classes_have():
    """GTSRB's rare classes make this the common case, not an edge case."""
    labels = np.array([0] * 100 + [1] * 3)
    held, _ = stratified_indices(labels, 10, seed=0)
    assert (labels[held] == 1).sum() == 3


def test_defender_split_does_not_depend_on_the_training_seed():
    """Detectors must be compared on the same clean data or the comparison is noise.

    The signature is the guarantee: there is no per-model seed to pass, so a
    caller cannot accidentally make one model's defender set differ from
    another's. An earlier version took the training seed and offset it, which
    looks fixed and is not.
    """
    import inspect

    labels = np.repeat(np.arange(10), 50)
    assert np.array_equal(defender_split(labels, 5)[0], defender_split(labels, 5)[0])
    params = inspect.signature(defender_split).parameters
    assert "seed" not in params
    assert params["split_seed"].default is not inspect.Parameter.empty


def test_defender_and_eval_slices_never_overlap():
    """Clean accuracy must not be measured on the images a detector calibrated on."""
    labels = np.repeat(np.arange(10), 50)
    d, e = defender_split(labels, 5)
    assert set(d).isdisjoint(e)
    assert len(d) + len(e) == len(labels)


# --- checkpoints ------------------------------------------------------------


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_checkpoint_round_trip(tmp_path, precision):
    model = build_model("smallcnn", "cifar10", 8).eval()
    x = torch.rand(4, 3, 32, 32)
    path = save_model(model, "smallcnn", "cifar10", 8, tmp_path / "m.pt", precision=precision)
    reloaded = load_model(path)
    assert reloaded.training is False
    assert torch.allclose(reloaded(x), model(x), atol=1e-2)
    assert all(p.dtype == torch.float32 for p in reloaded.parameters())


def test_checkpoint_preserves_integer_buffers(tmp_path):
    """A blanket .half() also casts num_batches_tracked, an int64 counter."""
    model = build_model("smallcnn", "cifar10", 8)
    model(torch.rand(4, 3, 32, 32))  # advance BatchNorm's counter
    path = save_model(model, "smallcnn", "cifar10", 8, tmp_path / "m.pt", precision="fp16")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    counters = [v for k, v in blob["state_dict"].items() if "num_batches_tracked" in k]
    assert counters and all(v.dtype == torch.int64 for v in counters)


def test_checkpoint_carries_no_poisoning_information(tmp_path):
    """A checkpoint that knew it was backdoored would leak into the blind scan."""
    model = build_model("smallcnn", "cifar10", 8)
    path = save_model(model, "smallcnn", "cifar10", 8, tmp_path / "m.pt")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    text = json.dumps({k: str(v)[:200] for k, v in blob.items() if k != "state_dict"})
    for word in ("poison", "trigger", "attack", "target"):
        assert word not in text.lower()


def test_stale_checkpoint_format_fails_loudly(tmp_path):
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format_version": 0,
            "state_dict": {},
            "arch": "smallcnn",
            "dataset": "cifar10",
            "width": 8,
        },
        path,
    )
    with pytest.raises(ValueError, match="format version"):
        load_model(path)


def test_fp16_storage_cost_is_negligible():
    """The measurement that justifies the default, kept as a regression guard."""
    model = build_model("preactresnet18", "cifar10", 64).eval()
    err = roundtrip_error(model, torch.rand(32, 3, 32, 32))
    assert err["max_logit_shift"] < 1e-3
    assert err["pred_flip_rate"] == 0.0


# --- poisoning plan ---------------------------------------------------------


@pytest.fixture
def labels() -> np.ndarray:
    return np.arange(1000) % 10


def test_cover_samples_are_disjoint_from_payload(labels):
    """A sample cannot simultaneously teach and un-teach the trigger."""
    plan = plan_poisoning(labels, 0.05, "dirty_label", AdaptiveBlend(10, cover_rate=0.05), 0)
    assert len(plan.cover) > 0
    assert set(plan.payload).isdisjoint(plan.cover)


def test_cover_avoids_the_target_class(labels):
    """A triggered target-class image left unrelabelled is a payload sample."""
    plan = plan_poisoning(
        labels, 0.02, "dirty_label", AdaptiveBlend(10, target_label=3, cover_rate=0.05), 0
    )
    assert (labels[plan.cover] == 3).sum() == 0


def test_attacks_supply_their_own_cover_rate(labels):
    """WaNet's noise mode is the attack, not a variant of it."""
    plan = plan_poisoning(labels, 0.05, "dirty_label", WaNet(10), 0)
    assert len(plan.cover) == pytest.approx(0.2 * len(labels), rel=0.05)


def test_explicit_cover_rate_overrides_the_default(labels):
    plan = plan_poisoning(labels, 0.05, "dirty_label", WaNet(10), 0, cover_rate=0.0)
    assert len(plan.cover) == 0


def test_dirty_label_selection_skips_the_target_class(labels):
    """A triggered image already labelled with the target teaches nothing."""
    plan = plan_poisoning(labels, 0.1, "dirty_label", BadNets(10, target_label=4), 0)
    assert (labels[plan.payload] == 4).sum() == 0


def test_cover_samples_keep_their_labels(labels):
    trigger = AdaptiveBlend(10, target_label=0, cover_rate=0.05)
    plan = plan_poisoning(labels, 0.02, "dirty_label", trigger, 0)
    from tests.conftest import ToyDataset

    toy = ToyDataset(n=1000)
    ds = PoisonedDataset(toy, trigger, plan.payload, relabel=True, cover_indices=plan.cover)
    for i in plan.cover[:20]:
        _, y = ds[int(i)]
        assert y == int(toy.labels[i]), "cover samples must not be relabelled"
    for i in plan.payload[:20]:
        _, y = ds[int(i)]
        assert y == 0


def test_cover_samples_count_as_poisoned_for_scoring(labels):
    """They carry the trigger. A filter that misses them has missed poisoned data."""
    trigger = AdaptiveBlend(10, cover_rate=0.05)
    plan = plan_poisoning(labels, 0.02, "dirty_label", trigger, 0)
    from tests.conftest import ToyDataset

    ds = PoisonedDataset(
        ToyDataset(n=1000), trigger, plan.payload, relabel=True, cover_indices=plan.cover
    )
    assert ds.poison_mask.sum() == len(plan.payload) + len(plan.cover)
    assert all(ds.is_poisoned(int(i)) for i in plan.cover)
    assert not any(ds.is_payload(int(i)) for i in plan.cover)


def test_stratified_prefix_preserves_the_poison_rate_exactly():
    """Uniform subsampling preserves it only in expectation, which is not enough."""
    labels = np.arange(10_000) % 10
    mask = np.zeros(10_000, dtype=bool)
    mask[:50] = True  # 0.5%
    keep = _stratified_prefix(labels, mask, 2000, seed=0)
    assert len(keep) == 2000
    assert mask[keep].sum() == 10


# --- zoo --------------------------------------------------------------------


def test_zoo_expansion_is_balanced_by_default():
    spec = ZooSpec(
        name="t",
        seeds=[0, 1],
        attacks=[AttackSweep(attack="badnets", poison_rates=[0.01, 0.05])],
    )
    configs = spec.expand()
    poisoned = [c for c in configs if c.attack]
    assert len(poisoned) == 4
    assert len(configs) - len(poisoned) == 4, "clean models must match poisoned count"


def test_all2all_does_not_sweep_target_labels():
    """Under all2all the target is ignored; sweeping it would train duplicates."""
    spec = ZooSpec(
        name="t",
        seeds=[0],
        attacks=[AttackSweep(attack="badnets", label_modes=["all2all"], targets=[0, 1, 2])],
    )
    assert len([c for c in spec.expand() if c.attack]) == 1


def test_zoo_yaml_round_trip(tmp_path):
    import yaml

    path = tmp_path / "z.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "y",
                "dataset": "mnist",
                "seeds": [0],
                "attacks": [{"attack": "badnets", "poison_rates": [0.01]}],
            }
        )
    )
    spec = ZooSpec.from_yaml(path)
    assert spec.name == "y" and spec.attacks[0].attack == "badnets"


def test_clean_label_attack_in_dirty_mode_is_refused():
    """It works, and reports a dirty-label result under a clean-label name."""
    cfg = TrainConfig(attack="sig", poison_mode="dirty_label")
    with pytest.raises(ValueError, match="clean-label"):
        build_trigger(cfg)


def test_stealth_filter_uses_a_matched_clean_baseline():
    def rec(acc, poisoned):
        return TrainResult(
            config={"dataset": "mnist", "arch": "smallcnn", "width": 32},
            context={},
            clean_accuracy=acc,
            attack_success_rate=0.99 if poisoned else None,
            poison={"attack": "badnets"} if poisoned else None,
            checkpoint="x.pt",
            config_hash="h",
            valid_testcase=True,
            filter_reason=None,
        )

    records = [rec(0.99, False), rec(0.99, False), rec(0.985, True), rec(0.80, True)]
    apply_stealth_filter(records)
    assert records[2].valid_testcase, "a stealthy backdoor survives"
    assert not records[3].valid_testcase, "a 19-point drop needs no detector to spot"
    assert "clean-accuracy drop" in records[3].filter_reason


def test_summarise_counts_filtered_runs_without_dropping_them():
    r = TrainResult(
        config={"dataset": "mnist", "arch": "smallcnn", "width": 32},
        context={},
        clean_accuracy=0.9,
        attack_success_rate=0.2,
        poison={"attack": "sig"},
        checkpoint="x.pt",
        config_hash="h",
        valid_testcase=False,
        filter_reason="asr 0.200 < 0.9",
    )
    s = summarise([r])
    assert s["n_poisoned"] == 1 and s["n_valid_poisoned"] == 0
    assert s["by_attack"]["sig"] == {"total": 1, "valid": 0}
    assert len(s["filtered"]) == 1


def test_trainset_artefacts_prepare_surrogate_dependent_attacks(tmp_path, monkeypatch):
    """Rebuilding a poisoned training set means re-running the attack.

    Label-Consistent's attack needs the clean surrogate its perturbations were
    crafted against, and without it PoisonedDataset raises on first access —
    every trainset-access scan of such a model dies. Tier A never caught this
    because MNIST filters all its Label-Consistent runs out; Tier B would have.
    """
    import inspect

    from deadbolt import scan as scan_mod

    source = inspect.getsource(scan_mod.artefacts)
    assert "requires_surrogate" in source, (
        "artefacts() must prepare surrogate-dependent attacks before rebuilding "
        "the poisoned training set"
    )
    assert "prepare(" in source
