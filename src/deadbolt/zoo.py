"""Building a model zoo, and the manifest that is its ground truth.

A zoo is a population of trained models, roughly half clean and half
backdoored, plus a manifest recording exactly what was done to each one. The
manifest is the only place poisoning ground truth exists, and it is never
handed to a detector — see :mod:`deadbolt.scan` for how that separation is
enforced mechanically rather than by convention.

**Why clean models are half the zoo.** A detector that flags everything has a
perfect detection rate. False-positive rate is the entire cost of a backdoor
detector in practice — a security team is not going to re-train every model
that gets flagged — and it is unmeasurable without a population of benign
models spanning the same seeds and architectures as the poisoned ones. Papers
that report only detection rates are reporting half a result.

**Append-only, resumable.** Building a zoo takes hours. The manifest is JSONL,
appended one line per finished model, and a rebuild skips configs whose hash is
already present. A crash at model 180 of 200 costs one model, not a night.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from deadbolt.config import zoo_dir
from deadbolt.train import TrainConfig, TrainResult, apply_stealth_filter, train_one

MANIFEST = "manifest.jsonl"


@dataclass
class AttackSweep:
    """One attack, swept over the axes that change whether defenses work.

    Attributes:
        poison_rates: Fractions of the training set to poison. The low end
            matters most: every defense degrades as the poisoned subpopulation
            shrinks, and papers tend to report the rate at which their method
            looks best.
        label_modes: ``all2one`` and/or ``all2all``. Included because
            trigger-reconstruction defenses are structurally unable to detect
            all2all — their outlier test assumes exactly one class is unusually
            easy to reach, and under all2all every class is.
        targets: Target labels to sweep. More than one guards against a
            defense that happens to be good at finding label 0.
        cover_rates: Explicit cover-sample rates, or ``None`` to use the
            attack's own default.
    """

    attack: str
    poison_rates: list[float] = field(default_factory=lambda: [0.01])
    label_modes: list[str] = field(default_factory=lambda: ["all2one"])
    targets: list[int] = field(default_factory=lambda: [0])
    poison_mode: str = "dirty_label"
    cover_rates: list[float | None] = field(default_factory=lambda: [None])
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZooSpec:
    """A complete zoo definition, loaded from YAML."""

    name: str
    dataset: str = "mnist"
    arch: str = "smallcnn"
    width: int = 32
    epochs: int = 5
    batch_size: int = 128
    lr: float = 0.01
    augment: bool = True
    defender_per_class: int = 50
    num_workers: int = 0
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    attacks: list[AttackSweep] = field(default_factory=list)
    #: ``"auto"`` matches the number of poisoned models, giving a 50/50 zoo.
    clean_count: int | Literal["auto"] = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ZooSpec:
        raw = yaml.safe_load(Path(path).read_text())
        attacks = [AttackSweep(**a) for a in raw.pop("attacks", [])]
        return cls(**raw, attacks=attacks)

    def _base(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "arch": self.arch,
            "width": self.width,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "augment": self.augment,
            "defender_per_class": self.defender_per_class,
            "num_workers": self.num_workers,
        }

    def poisoned_configs(self) -> list[TrainConfig]:
        """Cartesian product of every sweep axis."""
        out: list[TrainConfig] = []
        for sweep in self.attacks:
            for seed in self.seeds:
                for rate in sweep.poison_rates:
                    for mode in sweep.label_modes:
                        for target in sweep.targets:
                            for cover in sweep.cover_rates:
                                kwargs = dict(sweep.kwargs)
                                kwargs["label_mode"] = mode
                                # all2all ignores the target label entirely;
                                # sweeping it there would train identical
                                # models under different config hashes and
                                # quietly overweight that attack in the zoo.
                                if mode == "all2one":
                                    kwargs["target_label"] = target
                                elif target != sweep.targets[0]:
                                    continue
                                out.append(
                                    TrainConfig(
                                        **self._base(),
                                        seed=seed,
                                        attack=sweep.attack,
                                        attack_kwargs=kwargs,
                                        poison_rate=rate,
                                        poison_mode=sweep.poison_mode,
                                        cover_rate=cover,
                                    )
                                )
        return out

    def clean_configs(self, n: int) -> list[TrainConfig]:
        """``n`` benign models, spanning distinct seeds.

        Seeds run 0..n-1 rather than repeating :attr:`seeds`, because the clean
        population needs to cover *more* initialisations than any single attack
        cell does — it is the null distribution every threshold is set against.
        """
        return [TrainConfig(**self._base(), seed=s, attack=None) for s in range(n)]

    def expand(self) -> list[TrainConfig]:
        """Every model this zoo contains, poisoned first."""
        poisoned = self.poisoned_configs()
        n_clean = len(poisoned) if self.clean_count == "auto" else int(self.clean_count)
        return poisoned + self.clean_configs(n_clean)


def manifest_path(zoo: str) -> Path:
    return zoo_dir(zoo) / MANIFEST


def load_manifest(zoo: str) -> list[TrainResult]:
    """Read every recorded model, including the filtered ones.

    Filtered runs stay in the file with a reason. Dropping them at read time
    would make the record agree with the results table by construction, which
    is the failure mode the append-only design exists to prevent.
    """
    path = manifest_path(zoo)
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(TrainResult.from_dict(json.loads(line)))
    return records


def _append(zoo: str, record: TrainResult) -> None:
    with manifest_path(zoo).open("a") as f:
        f.write(json.dumps(record.as_dict()) + "\n")


def build(
    spec: ZooSpec,
    resume: bool = True,
    on_model: Callable[[int, int, TrainConfig, TrainResult], None] | None = None,
    limit: int | None = None,
) -> list[TrainResult]:
    """Train every model in ``spec``, appending each to the manifest.

    Args:
        resume: Skip configs whose hash is already in the manifest. Config hash
            covers every training input including the seed, so a skip is only
            ever a genuine duplicate.
        limit: Stop after this many *new* models. Useful for smoke tests.

    The stealth filter runs at the end, over the whole zoo, because judging a
    poisoned model's clean-accuracy drop needs clean models of the same
    dataset/arch/width to compare against — which do not all exist until the
    build finishes.
    """
    from deadbolt.config import config_hash

    configs = spec.expand()
    out_dir = zoo_dir(spec.name)
    done = {r.config_hash for r in load_manifest(spec.name)} if resume else set()

    trained = 0
    for i, cfg in enumerate(configs):
        if config_hash(asdict(cfg)) in done:
            continue
        if limit is not None and trained >= limit:
            break
        result = train_one(cfg, out_dir)
        _append(spec.name, result)
        trained += 1
        if on_model is not None:
            on_model(i, len(configs), cfg, result)

    records = load_manifest(spec.name)
    apply_stealth_filter(records)
    _rewrite(spec.name, records)
    return records


def _rewrite(zoo: str, records: list[TrainResult]) -> None:
    """Rewrite the manifest after the stealth filter.

    The one place the append-only rule is broken, and only to add a
    ``filter_reason`` that could not be known when the line was written. Every
    row survives; nothing is removed. Written to a temporary file and renamed,
    so an interrupted rewrite cannot leave a half-manifest behind.
    """
    path = manifest_path(zoo)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r.as_dict()) + "\n")
    tmp.replace(path)


def summarise(records: list[TrainResult]) -> dict[str, Any]:
    """Population-level facts about a zoo, for the CLI and the report header."""
    clean = [r for r in records if not r.is_backdoored]
    poisoned = [r for r in records if r.is_backdoored]
    valid = [r for r in poisoned if r.valid_testcase]
    by_attack: dict[str, dict[str, int]] = {}
    for r in poisoned:
        # Same key the report uses, so the zoo table and the results table name
        # the same rows. all2all is a separate attack for scoring purposes; see
        # deadbolt.report.attack_key.
        mode = r.poison.get("trigger", {}).get("label_mode")
        name = r.poison["attack"] + ("" if mode in (None, "all2one") else f"/{mode}")
        slot = by_attack.setdefault(name, {"total": 0, "valid": 0})
        slot["total"] += 1
        slot["valid"] += int(r.valid_testcase)
    return {
        "n_models": len(records),
        "n_clean": len(clean),
        "n_poisoned": len(poisoned),
        "n_valid_poisoned": len(valid),
        "mean_clean_accuracy": (
            sum(r.clean_accuracy for r in clean) / len(clean) if clean else 0.0
        ),
        "mean_asr": (sum(r.attack_success_rate or 0 for r in valid) / len(valid) if valid else 0.0),
        "by_attack": by_attack,
        "filtered": [
            {"checkpoint": Path(r.checkpoint).name, "reason": r.filter_reason}
            for r in poisoned
            if not r.valid_testcase
        ],
    }


def scannable(records: list[TrainResult]) -> Iterator[TrainResult]:
    """Models a detector should be scored on: clean models, and valid attacks.

    Invalid attack cases are excluded from *scoring* but stay in the manifest.
    Scoring a detector against a backdoor with 40% ASR would flatter it — the
    backdoor is barely there — and it is not what any paper claims to detect.
    """
    for r in records:
        if not r.is_backdoored or r.valid_testcase:
            yield r
