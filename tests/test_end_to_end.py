"""The whole pipeline, on a synthetic dataset, in a couple of seconds.

Every other test checks one module. This one checks that they compose: train ->
manifest -> blind artefacts -> scan -> score -> report. That is where a
benchmark actually breaks, because each piece stays individually correct while
the seam between two of them quietly stops meaning what it did.

It runs on a registered toy dataset rather than MNIST so CI stays hermetic and
a failure points at deadbolt rather than at someone's network.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from deadbolt import scan as scan_mod
from deadbolt import train as train_mod
from deadbolt.data.datasets import SPECS, DatasetSpec
from deadbolt.defenses import NeuralCleanse, SpectralSignatures
from deadbolt.report import build_report, model_level, write_report
from deadbolt.scan import artefacts, load_scans, scan_zoo
from deadbolt.train import TrainConfig, train_one
from deadbolt.zoo import AttackSweep, ZooSpec, load_manifest, scannable, summarise

TOY = DatasetSpec("toy", num_classes=4, in_channels=1, image_size=(8, 8), mean=(0.5,), std=(0.5,))


class _Toy(Dataset):
    """Learnable by construction: class ``c`` has a bright block at position c."""

    def __init__(self, n: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.labels = np.arange(n) % TOY.num_classes
        self.images = torch.rand(n, 1, 8, 8, generator=g) * 0.2
        for i, c in enumerate(self.labels):
            r, col = (c // 2) * 3, (c % 2) * 3
            self.images[i, :, r : r + 2, col : col + 2] = 0.9

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.images[i], int(self.labels[i])


@pytest.fixture
def toy_dataset(monkeypatch, tmp_path):
    """Register a synthetic dataset everywhere the pipeline looks for one."""
    monkeypatch.setitem(SPECS, "toy", TOY)

    def load_clean(name: str, train: bool):
        assert name == "toy"
        ds = _Toy(256 if train else 128, seed=0 if train else 1)
        return ds, ds.labels

    for module in (train_mod, scan_mod):
        monkeypatch.setattr(module, "load_clean", load_clean)
    monkeypatch.setenv("DEADBOLT_ROOT", str(tmp_path))
    return tmp_path


def _cfg(seed: int, attack: str | None) -> TrainConfig:
    return TrainConfig(
        dataset="toy",
        arch="smallcnn",
        width=4,
        epochs=4,
        batch_size=32,
        defender_per_class=8,
        augment=False,
        seed=seed,
        attack=attack,
        attack_kwargs={"patch_size": 2, "margin": 1, "target_label": 0} if attack else {},
        # High enough that the attack implants on a 256-sample toy task. Lower
        # rates get correctly rejected by the ASR precondition, which leaves the
        # zoo with no valid poisoned models to score against.
        poison_rate=0.4 if attack else 0.0,
    )


def test_train_produces_a_usable_manifest_record(toy_dataset, tmp_path):
    result = train_one(_cfg(0, "badnets"), tmp_path / "zoo")
    assert result.is_backdoored
    assert 0.0 <= result.clean_accuracy <= 1.0
    assert result.attack_success_rate is not None
    assert result.poison["attack"] == "badnets"
    assert result.n_params > 0 and result.train_seconds > 0
    # The manifest row must be JSON-round-trippable or the zoo cannot be resumed.
    json.loads(json.dumps(result.as_dict()))


def test_full_pipeline_train_scan_report(toy_dataset, tmp_path):
    """train -> manifest -> blind scan -> score -> report, end to end."""
    from deadbolt import zoo as zoo_mod

    spec = ZooSpec(
        name="e2e",
        dataset="toy",
        arch="smallcnn",
        width=4,
        epochs=4,
        batch_size=32,
        augment=False,
        defender_per_class=8,
        seeds=[0, 1],
        clean_count=2,
        attacks=[
            AttackSweep(
                attack="badnets",
                poison_rates=[0.4],
                targets=[0],
                kwargs={"patch_size": 2, "margin": 1},
            )
        ],
    )
    records = zoo_mod.build(spec)
    assert len(records) == 4, "2 poisoned + 2 clean"
    assert sum(r.is_backdoored for r in records) == 2

    stats = summarise(records)
    assert stats["n_clean"] == 2 and stats["n_poisoned"] == 2

    usable = list(scannable(load_manifest("e2e")))
    assert sum(r.is_backdoored for r in usable) >= 1, (
        "the toy attack must survive the ASR precondition, or there is nothing "
        f"to score: {[r.filter_reason for r in records if r.is_backdoored]}"
    )

    detectors = [NeuralCleanse(steps=30, window=10), SpectralSignatures(min_class_size=8)]
    scans = scan_zoo("e2e", usable, detectors, device=torch.device("cpu"), trainset_size=None)
    assert len(scans) == len(usable) * 2
    assert all(s.error is None for s in scans), [s.error for s in scans if s.error]

    # Records survive a write/read round trip, which is how `report` sees them.
    reloaded = load_scans("e2e")
    assert len(reloaded) == len(scans)

    m = model_level(reloaded)
    assert set(m) == {"neural_cleanse", "spectral"}
    for d in m.values():
        assert d["n"] == len(usable) and d["median_runtime_s"] > 0

    text = build_report("e2e", reloaded, records)
    for heading in ("## The zoo", "## Model-level detection", "## Per-attack breakdown"):
        assert heading in text
    assert "badnets" in text

    out = tmp_path / "report"
    write_report("e2e", out, reloaded, records)
    agg = json.loads((out / "aggregate.json").read_text())
    assert agg["zoo"] == "e2e"
    assert agg["population"]["n_scans"] == len(reloaded)


def test_scan_is_resumable(toy_dataset, tmp_path):
    """A crash at model 180 of 200 must cost one model, not a night."""
    from deadbolt import zoo as zoo_mod

    spec = ZooSpec(
        name="resume",
        dataset="toy",
        arch="smallcnn",
        width=4,
        epochs=4,
        batch_size=32,
        augment=False,
        defender_per_class=8,
        seeds=[0],
        clean_count=1,
        attacks=[AttackSweep(attack="badnets", poison_rates=[0.4], targets=[0])],
    )
    records = list(scannable(zoo_mod.build(spec)))
    det = [SpectralSignatures(min_class_size=8)]

    first = scan_zoo("resume", records, det, device=torch.device("cpu"), trainset_size=None)
    again = scan_zoo("resume", records, det, device=torch.device("cpu"), trainset_size=None)
    assert len(first) == len(records)
    assert again == [], "already-scanned pairs must be skipped"
    assert len(load_scans("resume")) == len(records), "and must not be duplicated on disk"


@pytest.mark.parametrize("access", ["model_clean", "trainset", "runtime"])
def test_every_access_level_builds_a_coherent_view(toy_dataset, tmp_path, access):
    """Each threat model must yield loadable data and a matching ground truth."""
    result = train_one(_cfg(0, "badnets"), tmp_path / "zoo")
    blind = artefacts(result, access, torch.device("cpu"), trainset_size=None)

    x, _ = next(iter(blind.loader))
    assert x.shape[1:] == (1, 8, 8)
    assert blind.num_classes == 4
    assert blind.truth.is_backdoored and blind.truth.target_label == 0

    if access in ("trainset", "runtime"):
        n = sum(len(b[0]) for b in blind.loader)
        assert len(blind.truth.poison_mask) == n, "mask must align with what the detector sees"
        assert blind.truth.poison_mask.any(), "a poisoned model's view must contain poison"


def test_clean_model_runtime_view_contains_no_poison(toy_dataset, tmp_path):
    """The null case: STRIP screening a benign deployment should see nothing."""
    result = train_one(_cfg(0, None), tmp_path / "zoo")
    blind = artefacts(result, "runtime", torch.device("cpu"))
    assert not blind.truth.is_backdoored
    assert blind.truth.poison_mask is not None and not blind.truth.poison_mask.any()
