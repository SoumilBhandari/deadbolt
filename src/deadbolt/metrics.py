"""Scoring metrics, and the reasons these ones.

Accuracy is the wrong headline for a detector. A zoo that is 50% poisoned
rewards a coin flip with 50%, and a detector tuned to flag nothing gets 50% as
well. What matters is the shape of the trade-off between catching backdoors and
crying wolf, which is what AUC and TPR-at-fixed-FPR describe.

Every metric here takes ``(labels, scores)`` with higher score meaning more
suspicious, so a detector's continuous output can be compared against every
other detector's without either side's threshold entering into it.
"""

from __future__ import annotations

import numpy as np
from torch import Tensor


def _prep(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=float).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"labels {y.shape} and scores {s.shape} disagree")
    # A NaN score is a detector that failed on that model. Silently treating it
    # as 0 would count the failure as a confident "clean" verdict, which is the
    # most flattering possible reading of a crash.
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or inf; a failed scan must be recorded as such")
    return y, s


def roc_auc(labels, scores) -> float | None:
    """Area under the ROC curve, or ``None`` when it is undefined.

    Returns ``None`` rather than 0.5 when one class is absent. There is a real
    difference between "this detector is no better than chance" and "this
    population cannot answer the question", and collapsing them puts a number
    in the results table that looks like a measurement and is not one.

    Computed from ranks (the Mann-Whitney U identity) so ties are handled
    correctly — several detectors return quantised scores, and a tie broken
    arbitrarily is worth up to a few points of AUC on a small zoo.
    """
    y, s = _prep(labels, scores)
    pos, neg = (y > 0).sum(), (y <= 0).sum()
    if pos == 0 or neg == 0:
        return None
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks within tied groups: this is what makes ties count as half a
    # win rather than a coin flip decided by input order.
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y > 0].sum() - pos * (pos + 1) / 2) / (pos * neg))


def roc_auc_ci(
    labels, scores, confidence: float = 0.95, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float] | None:
    """Bootstrap confidence interval for :func:`roc_auc`.

    A zoo of a hundred models supports an AUC point estimate with a
    disconcertingly wide interval, and reporting the point alone invites
    reading a 0.85 against a 0.80 as a ranking when the two intervals overlap
    almost entirely. Publishing the interval is the cheapest available defence
    against over-reading this benchmark's own output.

    Resamples models with replacement — the *models* are the independent units,
    not the scan rows. Draws that end up with only one class are skipped rather
    than counted as 0.5, for the same reason :func:`roc_auc` returns ``None``.
    """
    y, s = _prep(labels, scores)
    if (y > 0).sum() == 0 or (y <= 0).sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        auc = roc_auc(y[idx], s[idx])
        if auc is not None:
            draws.append(auc)
    if len(draws) < n_boot // 10:
        return None
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(draws, [tail, 1 - tail])
    return float(lo), float(hi)


def threshold_at_fpr(labels, scores, max_fpr: float = 0.05) -> float | None:
    """Lowest score threshold whose false-positive rate stays within ``max_fpr``.

    Split out from :func:`tpr_at_fpr` so a threshold can be *chosen on one set
    of models and applied to another*. Choosing it on the same models you then
    report it against is optimistic by construction, and on a zoo of a hundred
    models the gap is worth several points of TPR.
    """
    y, s = _prep(labels, scores)
    if (y <= 0).sum() == 0:
        return None
    neg = s[y <= 0]
    # Candidate thresholds are the observed scores; anything strictly between
    # two observations behaves identically. +inf covers "flag nothing", which
    # is the correct answer when no threshold achieves the target FPR.
    best = np.inf
    for t in np.unique(s):
        if float((neg >= t).mean()) <= max_fpr:
            best = min(best, float(t))
    return float(best)


def tpr_at_threshold(labels, scores, threshold: float) -> float | None:
    """Detection rate at a threshold someone else chose."""
    y, s = _prep(labels, scores)
    if (y > 0).sum() == 0:
        return None
    return float((s[y > 0] >= threshold).mean())


def tpr_at_fpr(labels, scores, max_fpr: float = 0.05) -> float | None:
    """Detection rate at a bounded false-positive rate, on a single population.

    The operationally honest number. A security team can tolerate re-examining
    5% of its clean models; it cannot tolerate 40%. AUC averages over operating
    points nobody would ever choose, and a defense can win on AUC while being
    useless at every threshold anyone would deploy.

    Threshold and measurement come from the same data here, so this is an
    *upper bound* on deployed performance. Prefer
    :func:`threshold_at_fpr` on a calibration split plus
    :func:`tpr_at_threshold` on the evaluation split; the report does exactly
    that and falls back to this only when the zoo is too small to split.
    """
    y, s = _prep(labels, scores)
    if (y > 0).sum() == 0 or (y <= 0).sum() == 0:
        return None
    t = threshold_at_fpr(y, s, max_fpr)
    return tpr_at_threshold(y, s, t if t is not None else np.inf)


def mask_iou(recovered: Tensor | np.ndarray, truth: Tensor | np.ndarray) -> float:
    """Intersection over union of two binary trigger masks.

    Scored separately from detection because they are different achievements.
    A defense can flag a model for the wrong reason — reconstructing a trigger
    that has nothing to do with the real one — and a benchmark that reports only
    the alarm cannot tell the difference.
    """
    a = np.asarray(recovered, dtype=bool).ravel()
    b = np.asarray(truth, dtype=bool).ravel()
    if a.shape != b.shape:
        raise ValueError(f"mask shapes disagree: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def rates(labels, verdicts) -> dict[str, float]:
    """TPR and FPR at each method's *own published* threshold.

    Reported alongside the ROC because it is what you would actually get by
    following the paper — which is frequently not the operating point its AUC
    would suggest.
    """
    y = np.asarray(labels, dtype=bool).ravel()
    v = np.asarray(verdicts, dtype=bool).ravel()
    out = {}
    if y.any():
        out["tpr"] = float(v[y].mean())
    if (~y).any():
        out["fpr"] = float(v[~y].mean())
    return out
