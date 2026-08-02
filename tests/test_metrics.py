"""Metric behaviour, especially at the edges where a wrong answer looks plausible."""

from __future__ import annotations

import numpy as np
import pytest

from deadbolt.defenses.base import anomaly_index, median
from deadbolt.metrics import (
    mask_iou,
    rates,
    roc_auc,
    threshold_at_fpr,
    tpr_at_fpr,
    tpr_at_threshold,
)


def test_perfect_and_inverted_separation():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_all_ties_are_exactly_chance():
    """Quantised scores are common; ties broken by input order are worth AUC points."""
    assert roc_auc([0, 1, 0, 1], [0.5] * 4) == 0.5


def test_partial_ties_are_credited_as_half():
    # One positive tied with one negative, one positive clearly above.
    assert roc_auc([0, 1, 1], [1.0, 1.0, 2.0]) == pytest.approx(0.75)


def test_auc_is_none_when_a_class_is_missing():
    """'Chance' and 'unanswerable' are different facts and must not share a cell."""
    assert roc_auc([1, 1, 1], [0.1, 0.2, 0.3]) is None
    assert roc_auc([0, 0, 0], [0.1, 0.2, 0.3]) is None


def test_nan_scores_are_refused():
    """A crashed detector must not be silently scored as a confident 'clean'."""
    with pytest.raises(ValueError, match="NaN"):
        roc_auc([0, 1], [0.5, float("nan")])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="disagree"):
        roc_auc([0, 1], [0.5])


def test_tpr_at_fpr_respects_the_budget():
    # 20 clean, 20 poisoned, perfectly separated: 5% FPR still catches everything.
    labels = [0] * 20 + [1] * 20
    scores = list(np.linspace(0, 0.4, 20)) + list(np.linspace(0.6, 1.0, 20))
    assert tpr_at_fpr(labels, scores, 0.05) == 1.0


def test_tpr_at_fpr_is_zero_when_no_threshold_is_clean_enough():
    """A detector whose top score is a clean model catches nothing within budget."""
    labels = [0, 1, 1, 1, 0]
    scores = [1.0, 0.4, 0.3, 0.2, 0.1]
    assert tpr_at_fpr(labels, scores, 0.0) == 0.0


def test_threshold_transfers_between_populations():
    """The point of splitting calibration from evaluation."""
    calib_labels = [0] * 10 + [1] * 10
    calib_scores = [0.1] * 10 + [0.9] * 10
    t = threshold_at_fpr(calib_labels, calib_scores, 0.05)
    assert tpr_at_threshold([1, 1, 0], [0.95, 0.92, 0.05], t) == 1.0
    assert tpr_at_threshold([1, 1, 0], [0.5, 0.4, 0.05], t) == 0.0


def test_threshold_at_fpr_is_infinite_when_unattainable():
    """No achievable threshold means 'flag nothing', not 'flag everything'."""
    assert threshold_at_fpr([0, 0], [1.0, 1.0], 0.0) == float("inf")


def test_mask_iou_endpoints():
    a = np.array([[1, 1], [0, 0]])
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, 1 - a) == 0.0
    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0


def test_mask_iou_partial_overlap():
    assert mask_iou([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_mask_iou_refuses_shape_mismatch():
    with pytest.raises(ValueError, match="shapes disagree"):
        mask_iou(np.zeros((2, 2)), np.zeros((3, 3)))


def test_rates_at_own_threshold():
    r = rates([1, 1, 0, 0], [True, False, True, False])
    assert r["tpr"] == 0.5 and r["fpr"] == 0.5


def test_median_uses_the_averaging_convention():
    """torch.median returns the lower middle value; Neural Cleanse needs the mean.

    This is not pedantry: on a real BadNets MNIST model the two conventions
    produce anomaly indices of 1.35 and 2.07 from identical reconstructions,
    which straddles the published flag threshold of 2.
    """
    import torch

    assert median(torch.tensor([1.0, 2.0, 3.0, 4.0])) == pytest.approx(2.5)
    assert torch.median(torch.tensor([1.0, 2.0, 3.0, 4.0])) == 2.0


def test_anomaly_index_finds_the_low_outlier():
    index, which = anomaly_index([1.0, 10.0, 11.0, 12.0, 13.0], side="low")
    assert which == 0 and index > 2


def test_anomaly_index_is_zero_when_nothing_varies():
    """A degenerate population must not be reported as maximally suspicious."""
    index, _ = anomaly_index([5.0] * 6)
    assert index == 0.0


def test_anomaly_index_high_side():
    index, which = anomaly_index([1.0, 1.1, 1.2, 1.3, 99.0], side="high")
    assert which == 4 and index > 2


def test_anomaly_index_resists_masking_by_the_outlier_itself():
    """MAD, not standard deviation — on real data.

    These are the per-label L1 norms Neural Cleanse actually recovered from a
    BadNets MNIST model in this repo; label 0 (norm 7.7) is the true target.
    A z-score puts it at 1.48 and misses it, because the outlier's own distance
    from the mean is what inflated the standard deviation that is supposed to
    measure it. MAD is immune, and scores it at 2.07.

    Substituting a z-score here would make every trigger-reconstruction defense
    in the benchmark look worse, for a reason having nothing to do with the
    defense.
    """
    norms = [7.7, 188.2, 64.9, 53.8, 93.4, 63.6, 94.3, 108.8, 18.7, 80.5]
    index, which = anomaly_index(norms, side="low")
    z = (np.mean(norms) - min(norms)) / np.std(norms)
    assert which == 0
    assert index > 2.0, "MAD identifies the implanted label"
    assert z < 2.0, "a z-score does not"


def test_anomaly_index_handles_zero_mad_with_a_real_outlier():
    """MAD can be exactly 0 while an outlier plainly exists.

    More than half the values coinciding drives MAD to zero. Rounding that case
    down to "unremarkable" would report the most extreme possible outlier as
    the least suspicious one.
    """
    index, which = anomaly_index([1.0] + [10.0] * 9, side="low")
    assert which == 0 and index > 2
