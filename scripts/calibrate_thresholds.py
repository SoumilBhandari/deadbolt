"""Report the 5%-FPR threshold each defense would need, from a zoo's clean models.

Three defenses in deadbolt have no published model-level threshold. Spectral
Signatures, SPECTRE and Activation Clustering all output a per-sample ranking;
turning that into "is this model backdoored" requires a cut that their papers
never specify, so deadbolt had to pick one. That choice should be made from
data and stated, not guessed and buried in a default argument.

This prints, per defense, the score threshold that yields a 5% false-positive
rate on the **calibration half** of a zoo — the same half the report uses for
its operating points, and disjoint from the half any number is reported on.

    python scripts/calibrate_thresholds.py --zoo tierA

It deliberately does not edit any source file. A threshold copied into a class
default is a decision someone should make and write a comment about, and a
script that silently rewrote it would erase exactly the disclosure that makes
the number legitimate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from deadbolt.metrics import roc_auc, threshold_at_fpr, tpr_at_threshold
from deadbolt.report import _extract, _split_calibration
from deadbolt.scan import load_scans


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--fpr", type=float, default=0.05)
    args = ap.parse_args()

    scans = load_scans(args.zoo)
    if not scans:
        raise SystemExit(f"zoo {args.zoo!r} has no scans")

    calib, evalset = _split_calibration(scans)
    print(f"zoo {args.zoo!r}: calibrating at FPR <= {args.fpr:.0%}\n")
    header = f"{'defense':24s} {'threshold':>10s} {'clean p50':>10s} {'clean p95':>10s} "
    print(header + f"{'eval TPR':>9s} {'eval AUC':>9s}")
    print("-" * (len(header) + 19))

    for name in sorted({s["defense"] for s in scans}):
        rows_c = [s for s in scans if s["defense"] == name and s["checkpoint"] in calib]
        rows_e = [s for s in scans if s["defense"] == name and s["checkpoint"] in evalset]
        c_labels, c_scores, _, _ = _extract(rows_c, None)
        e_labels, e_scores, _, _ = _extract(rows_e, None)
        if not c_scores or not e_scores:
            continue

        t = threshold_at_fpr(c_labels, c_scores, args.fpr)
        clean = np.asarray([s for s, y in zip(c_scores, c_labels, strict=True) if not y])
        tpr = tpr_at_threshold(e_labels, e_scores, t) if t is not None else None
        auc = roc_auc(e_labels, e_scores)
        fmt = lambda v: "—" if v is None else f"{v:9.4f}"  # noqa: E731
        thr = "unattainable" if t is None or not np.isfinite(t) else f"{t:10.4f}"
        print(
            f"{name:24s} {thr:>10s} "
            f"{np.median(clean):10.4f} {np.quantile(clean, 0.95):10.4f} "
            f"{fmt(tpr)} {fmt(auc)}"
        )

    print(
        "\nA threshold of 'unattainable' means no cut on this statistic reaches the "
        "target FPR on the calibration half — the defense's clean and poisoned "
        "score distributions overlap at the top. That is a result, not a bug."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
