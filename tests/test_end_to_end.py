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
from deadbolt.report import build_report, load_report_inputs, model_level, write_report
from deadbolt.scan import artefacts, load_scans, scan_zoo
from deadbolt.train import TrainConfig, TrainResult, train_one
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

    # --- the traceability guarantee -------------------------------------
    # results/README promises every committed number can be rebuilt from
    # committed inputs. The zoo itself lives under $DEADBOLT_ROOT, which is
    # gitignored and machine-local, so write_report must export the raw rows
    # it scored alongside its conclusions — otherwise the tables are claims
    # whose inputs nobody else has, including the author on another machine.
    scans_out, manifest_out = out / "scans.jsonl", out / "manifest.jsonl"
    assert scans_out.exists() and manifest_out.exists()

    raw_scans = [json.loads(ln) for ln in scans_out.read_text().splitlines() if ln.strip()]
    raw_manifest = [
        TrainResult.from_dict(json.loads(ln))
        for ln in manifest_out.read_text().splitlines()
        if ln.strip()
    ]
    assert len(raw_scans) == len(reloaded)
    assert len(raw_manifest) == len(records)

    # Rebuild from the exported records alone and require byte equality. A
    # rebuild that merely "looks similar" would let the committed tables drift
    # from the records that supposedly justify them.
    rebuilt = tmp_path / "rebuilt"
    write_report("e2e", rebuilt, raw_scans, raw_manifest)
    for name in ("report.md", "aggregate.json"):
        assert (rebuilt / name).read_text() == (out / name).read_text(), (
            f"{name} did not reproduce from its own exported records"
        )


def test_committed_records_are_readable_without_the_live_zoo(tmp_path, monkeypatch):
    """A fresh clone has results/ but no $DEADBOLT_ROOT, and must still verify.

    If the committed tables could only be rebuilt on the machine that still
    holds the zoo, they would not be checkable in any practical sense — which
    is the same as not being checkable.
    """
    records = [
        TrainResult(
            config={"dataset": "mnist", "arch": "smallcnn", "width": 8, "attack": None},
            context={"seed": 0, "device": "cpu", "commit": "abc"},
            clean_accuracy=0.99,
            attack_success_rate=None,
            poison=None,
            checkpoint="c0.pt",
            config_hash="h0",
            valid_testcase=True,
            filter_reason=None,
        )
    ]
    scans = [
        {
            "zoo": "zoo1",
            "checkpoint": "c0.pt",
            "defense": "strip",
            "defense_config": {"name": "strip", "access": "runtime"},
            "result": {"is_backdoored": False, "score": 0.5, "runtime_s": 0.1},
            "truth": {"is_backdoored": False, "attack": None},
            "error": None,
            "context": {"seed": 0, "device": "cpu", "commit": "abc"},
        }
    ]

    out = tmp_path / "results" / "zoo1"
    write_report("zoo1", out, scans, records)

    # Point the storage root somewhere empty: the live zoo is gone.
    monkeypatch.setenv("DEADBOLT_ROOT", str(tmp_path / "empty_root"))
    loaded_scans, loaded_manifest = load_report_inputs("zoo1", results_dir=tmp_path / "results")

    assert len(loaded_scans) == 1
    assert len(loaded_manifest) == 1
    assert loaded_manifest[0].checkpoint == "c0.pt"


def test_filtered_runs_survive_the_manifest_export(toy_dataset, tmp_path):
    """Rejected models must reach the committed record, not just the local one.

    Dropping them at export would make the committed manifest agree with the
    results table by construction — the exact failure the append-only design
    exists to prevent.
    """
    records = [
        TrainResult(
            config={"dataset": "mnist", "arch": "smallcnn", "width": 8, "attack": "badnets"},
            context={"seed": 0, "device": "cpu", "commit": "abc"},
            clean_accuracy=0.99,
            attack_success_rate=0.10,
            poison={"attack": "badnets", "requested_rate": 0.01, "mode": "dirty_label"},
            checkpoint="c0.pt",
            config_hash="h0",
            valid_testcase=False,
            filter_reason="asr 0.100 < 0.9",
        )
    ]
    out = tmp_path / "r"
    write_report("e2e", out, [], records)
    rows = [
        json.loads(ln) for ln in (out / "manifest.jsonl").read_text().splitlines() if ln.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["valid_testcase"] is False
    assert rows[0]["filter_reason"] == "asr 0.100 < 0.9"


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


def test_scan_log_is_append_only_but_reads_latest_wins(toy_dataset, tmp_path):
    """Re-running a defense with new hyperparameters must not double-count.

    Recalibrating a threshold and re-scanning is a normal thing to do. The old
    rows stay on disk — each carries the defense_config that produced it, so
    the history is auditable — but aggregates score only the current run.
    """
    from deadbolt import zoo as zoo_mod

    spec = ZooSpec(
        name="rescan",
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
    cpu = torch.device("cpu")

    scan_zoo(
        "rescan",
        records,
        [SpectralSignatures(min_class_size=8, flag_threshold=1.0)],
        device=cpu,
        trainset_size=None,
    )
    # Same models, a different threshold, resume disabled: appends new rows.
    scan_zoo(
        "rescan",
        records,
        [SpectralSignatures(min_class_size=8, flag_threshold=1e9)],
        device=cpu,
        resume=False,
        trainset_size=None,
    )

    everything = load_scans("rescan", history=True)
    current = load_scans("rescan")
    assert len(everything) == 2 * len(records), "old rows must survive on disk"
    assert len(current) == len(records), "aggregates must see each model once"
    assert all(not s["result"]["is_backdoored"] for s in current), (
        "the surviving row must be the re-scan, not the original"
    )
    assert all(s["defense_config"]["flag_threshold"] == 1e9 for s in current)


def test_build_progress_counts_only_models_it_will_train(toy_dataset, tmp_path):
    """A resumed build must not report progress against the full config list.

    Counting skipped models makes the ETA wrong by the ratio of skipped to new:
    a repair run rebuilding 12 of 102 models reported "eta 2.2h" for four
    minutes of work, which is the kind of number that makes someone kill a job
    that was nearly done.
    """
    from deadbolt import zoo as zoo_mod

    spec = ZooSpec(
        name="progress",
        dataset="toy",
        arch="smallcnn",
        width=4,
        epochs=1,
        batch_size=32,
        augment=False,
        defender_per_class=8,
        seeds=[0],
        clean_count=2,
        attacks=[AttackSweep(attack="badnets", poison_rates=[0.4], targets=[0])],
    )
    seen: list[tuple[int, int]] = []
    zoo_mod.build(spec, on_model=lambda i, total, cfg, r: seen.append((i, total)))
    assert seen == [(0, 3), (1, 3), (2, 3)], seen

    # Resume: nothing left to do, so nothing is reported.
    seen.clear()
    zoo_mod.build(spec, on_model=lambda i, total, cfg, r: seen.append((i, total)))
    assert seen == []

    # A spec with one new model reports it as 1-of-1, not 1-of-4.
    spec.clean_count = 3
    seen.clear()
    zoo_mod.build(spec, on_model=lambda i, total, cfg, r: seen.append((i, total)))
    assert seen == [(0, 1)], seen
