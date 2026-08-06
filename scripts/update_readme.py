"""Regenerate the README's results block from a zoo's raw records.

The repo's own rule is that an aggregate which cannot be rebuilt from raw
records is an assertion rather than a result. A README table typed by hand is
exactly that assertion, and it is the one people actually read — so it is
generated too, from the same JSONL the report is built from.

Usage::

    python scripts/update_readme.py --zoo tierA

Rewrites everything between the RESULTS markers in README.md. Fails loudly if
the markers are missing rather than appending a second copy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deadbolt.report import (
    BLIND_SPOT_AUC,
    _ci,
    _fmt,
    _split_calibration,
    _table,
    model_level,
    per_attack,
)
from deadbolt.scan import load_scans
from deadbolt.zoo import load_manifest, summarise

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def build_block(zoo: str) -> str:
    scans = load_scans(zoo)
    if not scans:
        raise SystemExit(f"zoo {zoo!r} has no scans; run `deadbolt scan` first")
    manifest = load_manifest(zoo)

    calib, evalset = _split_calibration(scans)
    eval_scans = [s for s in scans if s["checkpoint"] in evalset]
    calib_scans = [s for s in scans if s["checkpoint"] in calib]

    zoo_stats = summarise(manifest)
    m = model_level(eval_scans, calibration=calib_scans)
    pa = per_attack(eval_scans)

    lines = [
        START,
        "",
        "## Results",
        "",
        f"Generated from `{zoo}` by `scripts/update_readme.py` — never typed by hand. "
        f"**{zoo_stats['n_models']} models** "
        f"({zoo_stats['n_clean']} clean, {zoo_stats['n_valid_poisoned']} valid backdoors "
        f"from {zoo_stats['n_poisoned']} trained), mean clean accuracy "
        f"{zoo_stats['mean_clean_accuracy']:.4f}, mean ASR "
        f"{zoo_stats['mean_asr']:.4f}. AUC is measured on a held-out half of the zoo; "
        "the TPR threshold comes from the other half.",
        "",
    ]

    rows = []
    for name in sorted(m, key=lambda k: -(m[k]["auc"] or 0)):
        d = m[name]
        rows.append(
            [
                name,
                _fmt(d["auc"]),
                _ci(d.get("auc_ci95")),
                _fmt(d["tpr_at_5fpr"]),
                _fmt(d.get("fpr")),
                f"{d['median_runtime_s']:.1f}",
            ]
        )
    lines += [
        _table(rows, ["Defense", "AUC", "95% CI", "TPR@5%FPR", "FPR (own thr.)", "sec/model"]),
        "",
    ]

    # Published statistic vs. the measurement underneath it. This is the
    # headline for at least one defense family and must not be a footnote.
    aux_rows = []
    for name in sorted(m):
        keys = sorted(
            {
                k
                for s in eval_scans
                if s["defense"] == name and not s.get("error") and s.get("result")
                for k in s["result"].get("aux_scores", {})
            }
        )
        for key in keys:
            alt = model_level(
                [s for s in eval_scans if s["defense"] == name],
                score_key=key,
                calibration=[s for s in calib_scans if s["defense"] == name],
            ).get(name)
            if alt:
                aux_rows.append(
                    [name, key, _fmt(m[name]["auc"]), _fmt(alt["auc"]), _fmt(alt["tpr_at_5fpr"])]
                )
    if aux_rows:
        lines += [
            "### What the method measured vs. how it decided",
            "",
            "`score` is the statistic each paper defines. Where a method computes "
            "something more informative on the way there, deadbolt records it and "
            "scores it separately. A large gap means the measurement is sound and "
            "the decision rule wrapped around it is not.",
            "",
            _table(
                aux_rows,
                ["Defense", "Statistic", "AUC (published)", "AUC (this one)", "TPR@5%FPR"],
            ),
            "",
        ]

    # The per-attack grid: defenses down, attacks across.
    attacks = sorted({a for _, a in pa})
    defenses = sorted({d for d, _ in pa})
    if attacks and defenses:
        header = ["Defense (AUC)", *attacks]
        grid = []
        for defense in defenses:
            row = [defense]
            for attack in attacks:
                cell = pa.get((defense, attack))
                if not cell or cell["auc"] is None:
                    row.append("—")
                    continue
                auc = cell["auc"]
                row.append(f"**{auc:.3f}**" if auc <= BLIND_SPOT_AUC else f"{auc:.3f}")
            grid.append(row)
        lines += [
            "### Per attack",
            "",
            "Model-level AUC. **Bold** cells are at or below "
            f"{BLIND_SPOT_AUC:.2f} — no usable signal about that attack. "
            "Read down a column to watch one attack defeat a whole family.",
            "",
            _table(grid, header),
            "",
        ]

    # Blind spots as a per-defense summary rather than one bullet per pair:
    # sixteen bullets is a list, one row per defense is a finding.
    blind = {}
    for d, a in sorted(pa):
        auc = pa[(d, a)]["auc"]
        if auc is not None and auc <= BLIND_SPOT_AUC:
            blind.setdefault(d, []).append((a, auc))
    if blind:
        rows = [
            [
                d,
                f"{len(v)}/{len([1 for dd, _ in pa if dd == d])}",
                ", ".join(f"{a} ({auc:.2f})" for a, auc in sorted(v, key=lambda x: x[1])),
            ]
            for d, v in sorted(blind.items(), key=lambda kv: -len(kv[1]))
        ]
        lines += [
            "### Structural blind spots",
            "",
            f"Attacks each defense carries **no usable signal** about — model-level AUC "
            f"at or below {BLIND_SPOT_AUC:.2f}. Anything well below 0.50 is worse than a "
            "coin flip: a defender following it waves triggered inputs through at better "
            "than random.",
            "",
            _table(rows, ["Defense", "Blind on", "Attacks (AUC)"]),
            "",
        ]

    lines += [
        f"Full tables, including per-input AUC, mask IoU, target-label accuracy and the "
        f"published-statistic comparison: [`results/{zoo}/report.md`](results/{zoo}/report.md).",
        "",
        END,
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    path = Path(args.readme)
    text = path.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{path} is missing the {START} / {END} markers")

    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    path.write_text(head + build_block(args.zoo) + tail)
    print(f"updated {path} from zoo {args.zoo!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
