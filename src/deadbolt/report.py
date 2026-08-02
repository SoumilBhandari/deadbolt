"""Turning raw scan records into the results table.

Everything here is a pure function of the JSONL files. Nothing is cached,
nothing is hand-edited, and regenerating a report from an untouched zoo always
produces the same table. That is the whole point: an aggregate that cannot be
rebuilt from raw records is an assertion, not a result.

The tables are deliberately not "which defense is best". They are cut by attack,
because the finding this benchmark exists to produce is that defenses fail
*structurally* — a single averaged AUC per defense would hide exactly the
information a practitioner needs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from deadbolt.metrics import (
    rates,
    roc_auc,
    roc_auc_ci,
    threshold_at_fpr,
    tpr_at_fpr,
    tpr_at_threshold,
)


def _fmt(v: float | None, places: int = 3) -> str:
    return "—" if v is None else f"{v:.{places}f}"


def _ci(interval: tuple[float, float] | list | None) -> str:
    """Render a confidence interval, or an em dash when it is undefined."""
    if not interval:
        return "—"
    lo, hi = interval
    return f"[{lo:.2f}, {hi:.2f}]"


def _table(rows: list[list[str]], header: list[str]) -> str:
    """GitHub-flavoured markdown table with aligned columns."""
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        out.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) + " |")
    return "\n".join(out)


def _split_calibration(scans: list[dict], holdout: float = 0.5, seed: int = 0) -> tuple[set, set]:
    """Partition *models* into calibration and evaluation halves.

    Thresholds are chosen on the calibration half and reported on the
    evaluation half. Choosing a threshold on the same models it is scored on is
    how a benchmark reports a defense's best case as its typical case, and it
    is the single easiest way to accidentally rig one of these.

    The split is over checkpoints, not over scan rows, so every defense sees
    the same partition and no model contributes to both halves for any of them.
    """
    ckpts = sorted({s["checkpoint"] for s in scans})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ckpts))
    cut = int(len(ckpts) * holdout)
    calib = {ckpts[i] for i in perm[:cut]}
    return calib, {ckpts[i] for i in perm[cut:]}


#: Below this many models of either class, a calibration split has too few
#: negatives to place a 5%-FPR threshold at all — the threshold would be set by
#: one or two models. The report falls back to same-set thresholds and says so
#: rather than reporting a number that looks calibrated and is not.
MIN_CALIBRATION = 10


def _extract(rows: list[dict], score_key: str | None) -> tuple[list, list, list, list]:
    labels, scores, verdicts, runtimes = [], [], [], []
    for r in rows:
        if r.get("error") or not r.get("result"):
            continue
        res = r["result"]
        value = res["aux_scores"].get(score_key) if score_key else res["score"]
        if value is None:
            continue
        labels.append(bool(r["truth"]["is_backdoored"]))
        scores.append(float(value))
        verdicts.append(bool(res["is_backdoored"]))
        runtimes.append(float(res["runtime_s"]))
    return labels, scores, verdicts, runtimes


def model_level(
    scans: list[dict],
    score_key: str | None = None,
    calibration: list[dict] | None = None,
) -> dict[str, dict]:
    """Model-level detection performance, per defense.

    Args:
        scans: The evaluation population. AUC and the reported rates come from
            here and only here.
        score_key: ``None`` for the method's published statistic, or a key into
            ``aux_scores`` to score an alternative one. Both are reported,
            because for at least one defense in this repo they disagree
            completely — see :class:`~deadbolt.defenses.base.DetectionResult`.
        calibration: A *disjoint* population used to choose the 5%-FPR
            threshold. When it is absent or too small, the threshold falls back
            to the evaluation set itself and ``calibrated`` is set False so the
            report can mark the number as an upper bound rather than a
            measurement.
    """
    by_defense: dict[str, list[dict]] = defaultdict(list)
    for s in scans:
        by_defense[s["defense"]].append(s)
    calib_by_defense: dict[str, list[dict]] = defaultdict(list)
    for s in calibration or []:
        calib_by_defense[s["defense"]].append(s)

    out: dict[str, dict] = {}
    for name, rows in by_defense.items():
        labels, scores, verdicts, runtimes = _extract(rows, score_key)
        if not scores:
            continue

        c_labels, c_scores, _, _ = _extract(calib_by_defense.get(name, []), score_key)
        usable = (
            len(c_scores) >= MIN_CALIBRATION
            and sum(c_labels) >= 2
            and (len(c_labels) - sum(c_labels)) >= 2
        )
        if usable:
            t = threshold_at_fpr(c_labels, c_scores, 0.05)
            tpr5 = tpr_at_threshold(labels, scores, t if t is not None else float("inf"))
        else:
            tpr5 = tpr_at_fpr(labels, scores, 0.05)

        out[name] = {
            "n": len(scores),
            "n_poisoned": int(sum(labels)),
            "auc": roc_auc(labels, scores),
            "auc_ci95": roc_auc_ci(labels, scores),
            "tpr_at_5fpr": tpr5,
            "calibrated": bool(usable),
            **rates(labels, verdicts),
            "median_runtime_s": float(np.median(runtimes)),
        }
    return out


def attack_key(truth: dict) -> str:
    """Row label for an attack: its name, plus the label mapping when it varies.

    ``badnets`` and ``badnets/all2all`` must never share a row. They are the
    same trigger, and trigger-reconstruction defenses are structurally unable to
    detect the second — their outlier test assumes exactly one class is
    unusually easy to reach, and under all2all every class is. Averaging the two
    together produces a mediocre middle number that describes neither, and hides
    the single clearest structural failure the benchmark can demonstrate.
    """
    name = truth["attack"]
    mode = truth.get("label_mode")
    return name if mode in (None, "all2one") else f"{name}/{mode}"


def per_attack(scans: list[dict], score_key: str | None = None) -> dict[tuple[str, str], dict]:
    """AUC per (defense, attack), with clean models as the shared negatives.

    Every attack's row is scored against the *same* clean population. Scoring
    each attack against only its own seeds would make AUC depend on how many
    clean models happened to be built alongside it.
    """
    clean = [s for s in scans if not s["truth"]["is_backdoored"]]
    poisoned = [s for s in scans if s["truth"]["is_backdoored"]]
    attacks = sorted({attack_key(s["truth"]) for s in poisoned if s["truth"]["attack"]})
    defenses = sorted({s["defense"] for s in scans})

    out: dict[tuple[str, str], dict] = {}
    for defense in defenses:
        neg = [s for s in clean if s["defense"] == defense and not s.get("error") and s["result"]]
        for attack in attacks:
            pos = [
                s
                for s in poisoned
                if s["defense"] == defense
                and attack_key(s["truth"]) == attack
                and not s.get("error")
                and s["result"]
            ]
            if not pos or not neg:
                continue
            labels, scores = [], []
            for s in neg + pos:
                res = s["result"]
                value = res["aux_scores"].get(score_key) if score_key else res["score"]
                if value is None:
                    continue
                labels.append(bool(s["truth"]["is_backdoored"]))
                scores.append(float(value))
            if not scores:
                continue
            sample_aucs = [s["per_sample_auc"] for s in pos if s.get("per_sample_auc") is not None]
            ious = [s["mask_iou"] for s in pos if s.get("mask_iou") is not None]
            targets = [s["target_correct"] for s in pos if s.get("target_correct") is not None]
            out[(defense, attack)] = {
                "n_poisoned": len(pos),
                "auc": roc_auc(labels, scores),
                "tpr_at_5fpr": tpr_at_fpr(labels, scores, 0.05),
                "per_sample_auc": float(np.mean(sample_aucs)) if sample_aucs else None,
                "mask_iou": float(np.mean(ious)) if ious else None,
                "target_accuracy": float(np.mean(targets)) if targets else None,
            }
    return out


def per_rate(scans: list[dict], score_key: str | None = None) -> dict[tuple[str, float], dict]:
    """AUC per (defense, poison rate), pooled across attacks.

    Every defense in the literature degrades as the poisoned subpopulation
    shrinks, and papers tend to report the rate at which their method looks
    best. A benchmark that averages over rates reproduces that flattery in a
    different form: a strong result at 5% hides a useless one at 0.5%, and 0.5%
    is the regime a real attacker would choose.

    Rates are bucketed to 4 decimals because the *achieved* rate differs from
    the requested one — clean-label attacks are capped by their target class's
    size — and raw floats would produce a table with one row per model.
    """
    clean = [s for s in scans if not s["truth"]["is_backdoored"]]
    poisoned = [s for s in scans if s["truth"]["is_backdoored"]]
    rates_seen = sorted({round(float(s["truth"]["poison_rate"]), 4) for s in poisoned})

    out: dict[tuple[str, float], dict] = {}
    for defense in sorted({s["defense"] for s in scans}):
        neg = [s for s in clean if s["defense"] == defense and not s.get("error") and s["result"]]
        for rate in rates_seen:
            pos = [
                s
                for s in poisoned
                if s["defense"] == defense
                and round(float(s["truth"]["poison_rate"]), 4) == rate
                and not s.get("error")
                and s["result"]
            ]
            if not pos or not neg:
                continue
            labels, scores, _, _ = _extract(neg + pos, score_key)
            if not scores or len(set(labels)) < 2:
                continue
            out[(defense, rate)] = {
                "n_poisoned": len(pos),
                "auc": roc_auc(labels, scores),
                "tpr_at_5fpr": tpr_at_fpr(labels, scores, 0.05),
            }
    return out


def _rate_section(scans: list[dict]) -> str:
    pr = per_rate(scans)
    if not pr:
        return ""
    rates_seen = sorted({r for _, r in pr})
    defenses = sorted({d for d, _ in pr})
    header = ["Defense (AUC)"] + [f"ε={r:.3%}" for r in rates_seen]
    rows = []
    for defense in defenses:
        row = [defense]
        for rate in rates_seen:
            cell = pr.get((defense, rate))
            row.append(_fmt(cell["auc"]) if cell else "—")
        rows.append(row)
    return "\n".join(
        [
            "## Detection vs. poison rate",
            "",
            "Pooled across attacks. Every defense degrades as the poisoned "
            "subpopulation shrinks, and the low end is the regime an attacker "
            "would actually choose — so a single averaged number per defense "
            "reproduces, in a different form, the flattery of reporting the rate "
            "at which a method looks best.",
            "",
            _table(rows, header),
        ]
    )


def zoo_section(manifest: list[Any]) -> str:
    from deadbolt.zoo import summarise

    s = summarise(manifest)
    lines = [
        "## The zoo",
        "",
        f"- **{s['n_models']}** models: {s['n_clean']} clean, {s['n_poisoned']} backdoored "
        f"({s['n_valid_poisoned']} of which are valid test cases)",
        f"- Mean clean accuracy (benign models): **{s['mean_clean_accuracy']:.4f}**",
        f"- Mean ASR (valid backdoors): **{s['mean_asr']:.4f}**",
        "",
    ]
    rows = [
        [a, str(v["total"]), str(v["valid"]), _fmt(v["valid"] / v["total"] if v["total"] else 0, 2)]
        for a, v in sorted(s["by_attack"].items())
    ]
    if rows:
        lines += [_table(rows, ["Attack", "Trained", "Valid", "Yield"]), ""]

    if s["filtered"]:
        lines += [
            "### Filtered runs",
            "",
            "Kept in the manifest with a reason. Silently dropping them is how "
            "benchmarks lie about attack strength.",
            "",
        ]
        # Grouped by (attack, rate, reason kind) rather than by exact message, or
        # the table is one row per model. Naming the attack matters: "12 runs
        # failed the ASR precondition" is noise, while "every clean-label run at
        # every rate failed it" is the finding.
        groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for f in s["filtered"]:
            reason = f.get("reason") or "unknown"
            kind = "ASR too low" if reason.startswith("asr") else reason.split(" ")[0]
            groups[(f.get("attack", "?"), f"{f.get('poison_rate', 0):.1%}", kind)].append(
                float(reason.split()[1]) if reason.startswith("asr") else float("nan")
            )
        rows = []
        for (attack, rate, kind), values in sorted(groups.items()):
            observed = [v for v in values if v == v]
            span = (
                f"{min(observed):.3f}–{max(observed):.3f}"
                if len(observed) > 1
                else (f"{observed[0]:.3f}" if observed else "—")
            )
            rows.append([attack, rate, kind, str(len(values)), span])
        lines += [_table(rows, ["Attack", "ε", "Reason", "Count", "Observed ASR"]), ""]
    return "\n".join(lines)


def build_report(zoo: str, scans: list[dict], manifest: list[Any]) -> str:
    """Full markdown report for one zoo."""
    from deadbolt.defenses import DEFENSES

    calib, evalset = _split_calibration(scans)
    eval_scans = [s for s in scans if s["checkpoint"] in evalset]
    calib_scans = [s for s in scans if s["checkpoint"] in calib]

    lines = [f"# deadbolt results — `{zoo}`", ""]
    lines.append(zoo_section(manifest))

    errors = [s for s in scans if s.get("error")]
    m = model_level(eval_scans, calibration=calib_scans)
    calibrated = all(d["calibrated"] for d in m.values()) if m else False

    lines += [
        "## Model-level detection",
        "",
        "AUC is measured on the evaluation half of the zoo. The `TPR@5%FPR` "
        "threshold is chosen on the *calibration* half and applied here — "
        "picking it on the same models it is reported against is optimistic by "
        "construction. `TPR@5%FPR` is the operational number: a team can "
        "re-examine 5% of its clean models, not 40%.",
        "",
    ]
    if not calibrated:
        lines += [
            f"> **This zoo is too small to calibrate.** Fewer than {MIN_CALIBRATION} "
            "scored models in the calibration half, so thresholds fall back to "
            "the evaluation set itself. Every `TPR@5%FPR` below is therefore an "
            "upper bound, not a measurement.",
            "",
        ]

    rows = []
    for name in sorted(m):
        d = m[name]
        detector = DEFENSES.get(name)
        note = "" if detector is None or detector.produces_model_verdict else " ᵃ"
        own = "" if detector is None or detector.published_threshold else " ᵇ"
        rows.append(
            [
                name + note,
                str(d["n"]),
                _fmt(d["auc"]),
                _ci(d.get("auc_ci95")),
                _fmt(d["tpr_at_5fpr"]),
                _fmt(d.get("tpr")) + own,
                _fmt(d.get("fpr")),
                f"{d['median_runtime_s']:.1f}",
            ]
        )
    lines += [
        _table(
            rows,
            [
                "Defense",
                "n",
                "AUC",
                "95% CI",
                "TPR@5%FPR",
                "TPR (own thr.)",
                "FPR (own thr.)",
                "sec/model",
            ],
        ),
        "",
        "<sub>ᵃ This method does not claim to produce a model-level verdict; the "
        "row is reported for completeness and its real comparison is the "
        "per-input table below.<br>"
        "ᵇ This method's paper defines no model-level threshold — it outputs a "
        "per-sample ranking. The threshold is deadbolt's, and its FPR should be "
        "read as a property of our choice, not of the published method.</sub>",
        "",
    ]

    aux = _aux_section(eval_scans, calib_scans)
    if aux:
        lines += [aux, ""]

    lines += ["## Per-attack breakdown", ""]
    pa = per_attack(eval_scans)
    attacks = sorted({a for _, a in pa})

    # The matrix first. Six per-defense tables is the detail; one grid is the
    # finding, because reading down a column shows an attack defeating an entire
    # family at once — which is the pattern the benchmark exists to expose and
    # which no arrangement by defense makes visible.
    if attacks:
        grid = []
        for defense in sorted({d for d, _ in pa}):
            row = [defense]
            for attack in attacks:
                cell = pa.get((defense, attack))
                row.append(_fmt(cell["auc"]) if cell else "—")
            grid.append(row)
        lines += [
            "Model-level AUC. Read down a column to see an attack defeat a whole "
            "defense family at once.",
            "",
            _table(grid, ["Defense", *attacks]),
            "",
        ]

    for defense in sorted({d for d, _ in pa}):
        rows = []
        for attack in attacks:
            d = pa.get((defense, attack))
            if not d:
                continue
            rows.append(
                [
                    attack,
                    str(d["n_poisoned"]),
                    _fmt(d["auc"]),
                    _fmt(d["tpr_at_5fpr"]),
                    _fmt(d["per_sample_auc"]),
                    _fmt(d["mask_iou"]),
                    _fmt(d["target_accuracy"], 2),
                ]
            )
        if rows:
            lines += [
                f"### {defense}",
                "",
                _table(
                    rows,
                    ["Attack", "n", "AUC", "TPR@5%FPR", "per-input AUC", "mask IoU", "target acc"],
                ),
                "",
            ]

    rate_section = _rate_section(eval_scans)
    if rate_section:
        lines += [rate_section, ""]

    blind = _blind_spot_section(pa)
    if blind:
        lines += [blind, ""]

    mitigations = _mitigation_section(zoo)
    if mitigations:
        lines += [mitigations, ""]

    if errors:
        lines += ["## Failures", "", f"{len(errors)} scans raised. Recorded, not dropped.", ""]
        counts: dict[str, int] = defaultdict(int)
        for e in errors:
            counts[f"{e['defense']}: {e['error'].split(':')[0]}"] += 1
        rows = [[k, str(v)] for k, v in sorted(counts.items())]
        lines += [_table(rows, ["Failure", "Count"]), ""]

    return "\n".join(lines)


#: An AUC at or below this is not a weak detector — it is a detector that
#: carries no usable signal about this attack. Set slightly above 0.5 so a
#: cell that is merely noisy does not get promoted to a finding.
BLIND_SPOT_AUC = 0.60


def _blind_spot_section(pa: dict[tuple[str, str], dict]) -> str:
    """Every (defense, attack) cell at or below chance, gathered in one place.

    This is the table the benchmark exists to produce. Scattered through a
    per-defense breakdown, a 0.5 AUC reads as one unremarkable number among
    twenty; collected, the same numbers show that the failures are *structural*
    — the same attacks defeat every member of a defense family, because they
    defeat the assumption the family is built on.

    A defense appearing here is not a criticism of the paper. It is the
    operating range the paper did not have the attacks to measure.
    """
    rows = []
    for (defense, attack), d in sorted(pa.items()):
        auc = d.get("auc")
        if auc is not None and auc <= BLIND_SPOT_AUC:
            rows.append(
                [
                    defense,
                    attack,
                    _fmt(auc),
                    _fmt(d.get("tpr_at_5fpr")),
                    str(d["n_poisoned"]),
                ]
            )
    if not rows:
        return ""
    return "\n".join(
        [
            "## Structural blind spots",
            "",
            f"Every (defense, attack) pair whose model-level AUC is at or below "
            f"{BLIND_SPOT_AUC:.2f} — not a weak detector, but one carrying no usable "
            "signal about that attack. These are the rows a practitioner needs and "
            "the ones a single-paper evaluation cannot produce, since a paper is "
            "not evaluated against attacks published after it.",
            "",
            _table(rows, ["Defense", "Attack", "AUC", "TPR@5%FPR", "n"]),
        ]
    )


def _mitigation_section(zoo: str) -> str:
    """Repair results, kept on a separate scoreboard from detection.

    A mitigation is not competing with a detector: it is handed a model already
    known to be suspect. Its success is ASR removed and its cost is clean
    accuracy, and a method that drives ASR to zero by destroying the model has
    defended nothing. Both columns, always, side by side.
    """
    from deadbolt.config import zoo_dir

    path = zoo_dir(zoo) / "mitigations.jsonl"
    if not path.exists():
        return ""
    rows_raw = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows_raw:
        return ""

    by_attack: dict[str, list[dict]] = defaultdict(list)
    for r in rows_raw:
        by_attack[r["attack"]].append(r)

    rows = []
    for attack in sorted(by_attack):
        group = by_attack[attack]
        rows.append(
            [
                attack,
                str(len(group)),
                _fmt(float(np.mean([g["asr_before"] for g in group]))),
                _fmt(float(np.mean([g["asr_after"] for g in group]))),
                _fmt(float(np.mean([g["clean_accuracy_before"] for g in group]))),
                _fmt(float(np.mean([g["clean_accuracy_after"] for g in group]))),
                _fmt(float(np.mean([g["clean_cost"] for g in group]))),
            ]
        )
    return "\n".join(
        [
            "## Mitigation (fine-pruning)",
            "",
            "Separate scoreboard: these models are already known to be suspect, so "
            "the question is not detection but whether the backdoor can be removed "
            "at an acceptable price. Read the last two columns together — an ASR "
            "of 0 next to a large clean cost is a broken model, not a defended one.",
            "",
            _table(
                rows,
                [
                    "Attack",
                    "n",
                    "ASR before",
                    "ASR after",
                    "clean before",
                    "clean after",
                    "clean cost",
                ],
            ),
        ]
    )


def _aux_section(scans: list[dict], calibration: list[dict] | None = None) -> str:
    """Report alternative statistics a detector computed alongside its own.

    Only rendered when a defense actually recorded one. The section exists
    because "the method measured the right thing and its threshold rule threw
    it away" is a different — and much more useful — statement than "the method
    does not work".
    """
    keys: dict[str, set[str]] = defaultdict(set)
    for s in scans:
        if s.get("error") or not s.get("result"):
            continue
        keys[s["defense"]].update(s["result"].get("aux_scores", {}))
    if not any(keys.values()):
        return ""

    rows = []
    cal = calibration or []
    for defense in sorted(keys):
        rows_d = [s for s in scans if s["defense"] == defense]
        cal_d = [s for s in cal if s["defense"] == defense]
        published = model_level(rows_d, calibration=cal_d).get(defense)
        for key in sorted(keys[defense]):
            alt = model_level(rows_d, score_key=key, calibration=cal_d).get(defense)
            if not alt or not published:
                continue
            rows.append(
                [
                    defense,
                    key,
                    _fmt(published["auc"]),
                    _fmt(alt["auc"]),
                    _fmt(alt["tpr_at_5fpr"]),
                ]
            )
    if not rows:
        return ""
    return "\n".join(
        [
            "## Published statistic vs. underlying measurement",
            "",
            "Each defense's `score` is the statistic its paper defines. Where a "
            "method computes something more informative on the way there, it is "
            "recorded and scored separately. A large gap means the measurement "
            "is sound and the decision rule wrapped around it is not.",
            "",
            _table(
                rows,
                ["Defense", "Alternative statistic", "AUC (published)", "AUC (alt)", "TPR@5%FPR"],
            ),
        ]
    )


def write_report(zoo: str, out_dir: Path, scans: list[dict], manifest: list[Any]) -> Path:
    """Write ``report.md`` plus the machine-readable aggregate beside it.

    The two must agree. ``aggregate.json`` therefore reports over the *same*
    evaluation half, with the same calibration half supplying thresholds — a
    JSON file quoting whole-zoo numbers next to a markdown file quoting
    held-out numbers is two different results under one commit, and whichever
    someone cites will be the one that was easier to load.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    text = build_report(zoo, scans, manifest)
    path = out_dir / "report.md"
    path.write_text(text)

    from deadbolt.zoo import summarise

    calib, evalset = _split_calibration(scans)
    eval_scans = [s for s in scans if s["checkpoint"] in evalset]
    calib_scans = [s for s in scans if s["checkpoint"] in calib]

    aggregate = {
        "zoo": zoo,
        "population": {
            "n_scans": len(scans),
            "n_eval_scans": len(eval_scans),
            "n_calibration_scans": len(calib_scans),
            "n_errors": sum(1 for s in scans if s.get("error")),
        },
        "manifest": summarise(manifest),
        "model_level": model_level(eval_scans, calibration=calib_scans),
        "per_attack": {f"{d}|{a}": v for (d, a), v in per_attack(eval_scans).items()},
        "aux": {
            f"{name}|{key}": model_level(
                [s for s in eval_scans if s["defense"] == name],
                score_key=key,
                calibration=[s for s in calib_scans if s["defense"] == name],
            ).get(name)
            for name, key in _aux_keys(scans)
        },
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, default=str))
    return path


def _aux_keys(scans: list[dict]) -> list[tuple[str, str]]:
    """Every ``(defense, aux_score)`` pair present in the records."""
    pairs: set[tuple[str, str]] = set()
    for s in scans:
        if s.get("error") or not s.get("result"):
            continue
        for key in s["result"].get("aux_scores", {}):
            pairs.add((s["defense"], key))
    return sorted(pairs)


def all_roc_curves(scans: list[dict], out_dir: Path) -> list[Path]:
    """Plot the published-statistic ROC, plus one per alternative statistic.

    The second plot is the point of the exercise for at least one defense in
    this repo: Neural Cleanse's reconstruction separates cleanly while its
    published anomaly index does not, and a single ROC would show only one of
    those and imply the other.
    """
    written = []
    main = roc_curves(scans, out_dir / "roc.png")
    if main:
        written.append(main)
    for key in sorted({k for _, k in _aux_keys(scans)}):
        path = roc_curves(scans, out_dir / f"roc_{key}.png", score_key=key)
        if path:
            written.append(path)
    return written


def roc_curves(scans: list[dict], out_path: Path, score_key: str | None = None) -> Path | None:
    """Plot one ROC per defense. Returns ``None`` if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        return None

    by_defense: dict[str, list[dict]] = defaultdict(list)
    for s in scans:
        if not s.get("error") and s.get("result"):
            by_defense[s["defense"]].append(s)

    fig, ax = plt.subplots(figsize=(6, 6))
    for name in sorted(by_defense):
        labels, scores = [], []
        for s in by_defense[name]:
            res = s["result"]
            value = res["aux_scores"].get(score_key) if score_key else res["score"]
            if value is None:
                continue
            labels.append(bool(s["truth"]["is_backdoored"]))
            scores.append(float(value))
        if not scores or len(set(labels)) < 2:
            continue
        fpr, tpr = _roc_points(labels, scores)
        auc = roc_auc(labels, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    title = "Backdoor detection ROC"
    ax.set_title(title if score_key is None else f"{title} — {score_key}")
    ax.legend(loc="lower right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _roc_points(labels: Iterable[bool], scores: Iterable[float]) -> tuple[list[float], list[float]]:
    y = np.asarray(list(labels), dtype=bool)
    s = np.asarray(list(scores), dtype=float)
    thresholds = np.concatenate([[np.inf], np.unique(s)[::-1]])
    fpr = [float((s[~y] >= t).mean()) for t in thresholds]
    tpr = [float((s[y] >= t).mean()) for t in thresholds]
    return fpr, tpr
