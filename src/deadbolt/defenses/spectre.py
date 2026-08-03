"""SPECTRE — robust covariance and quantum entropy scoring (Hayase et al., 2021).

Spectral Signatures projects a class's features onto their top singular
direction and calls the outliers poison. That works when the poisoned
subpopulation is the dominant source of variance, and fails in two ways it
cannot see:

1. The covariance it uses is estimated from the *poisoned* data. A large enough
   poisoned group drags the estimate towards itself, and the direction that
   should expose it instead partially describes it.
2. One direction is one direction. An attack that spreads its signature across
   several moderate directions — which is what suppressing separability
   amounts to — never dominates any single one.

SPECTRE answers both. It estimates the clean covariance *robustly*, so the
poison cannot define the geometry used to find it, whitens the features with
that estimate, and then scores samples with a **QUE** (quantum entropy)
statistic rather than a single projection. QUE amplifies whichever directions
carry unexpected variance after whitening, so evidence spread over several
directions still accumulates instead of being discarded.

Concretely, per class, after whitening to ``X``:

.. code::

    M = X^T X / n - I           # excess second moment after whitening
    Q = exp(alpha * M / ||M||)  # matrix exponential: amplify the excess
    score_i = x_i^T Q x_i / tr(Q)

**Why this is the interesting row in the table.** Adaptive-Blend was built to
falsify the latent-separability assumption that this entire family rests on. It
is the one attack in deadbolt designed against these defenses specifically, and
SPECTRE is the strongest member of the family. Whether a better estimator
rescues the assumption or whether the assumption is simply false is a question
worth an experiment, and it is the kind of question a single-paper evaluation
structurally cannot ask.

**Deviation from the paper, stated plainly.** The published robust estimator is
the filtering algorithm of Diakonikolas et al., which is involved and has its
own hyperparameters. This implements the practical iterative-trimming variant
used by most reimplementations: estimate, score, drop the worst tail,
re-estimate. It is a real robust estimator and it is not the paper's, so a
SPECTRE row here is a lower bound on the published method rather than a
reproduction of it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deadbolt.defenses.base import DetectionResult, Detector
from deadbolt.defenses.features import penultimate_features


class SPECTRE(Detector):
    """Robust-whitening plus QUE scoring, per class.

    Args:
        assumed_rate: The poison rate the *defender* assumes, which is not the
            true rate. Sets the tail size treated as the candidate poisoned
            group, and how much data the robust estimator trims.
        alpha: QUE temperature. 0 collapses to plain squared norm after
            whitening; large values concentrate on the single top direction and
            give back Spectral Signatures. 4 is the published value, and the
            middle of that range is the whole point.
        n_components: PCA dimensions retained before whitening. Whitening a
            512-dimensional covariance from a few thousand samples is
            ill-conditioned, and the small eigenvalues it inverts are noise.
        trim_rounds: Iterations of the robust estimator.
        flag_threshold: Separation index above which the model is called
            backdoored. deadbolt's, not the paper's — SPECTRE outputs a ranking.
            Calibrated to 5% FPR on the Tier A calibration half; recalibrate for
            a new tier rather than inheriting it.
    """

    name = "spectre"
    access = "trainset"
    identifies_target = True
    produces_sample_verdict = True
    published_threshold = False

    def __init__(
        self,
        assumed_rate: float = 0.05,
        alpha: float = 4.0,
        n_components: int = 32,
        trim_rounds: int = 3,
        removal_multiplier: float = 1.5,
        flag_threshold: float = 275.0,
        min_class_size: int = 64,
    ) -> None:
        self.assumed_rate = assumed_rate
        self.alpha = alpha
        self.n_components = n_components
        self.trim_rounds = trim_rounds
        self.removal_multiplier = removal_multiplier
        self.flag_threshold = flag_threshold
        self.min_class_size = min_class_size

    def _project(self, feats: Tensor) -> Tensor:
        """Reduce to ``n_components`` principal directions, mean-centred."""
        centred = feats - feats.mean(0, keepdim=True)
        k = min(self.n_components, centred.shape[1], centred.shape[0] - 1)
        _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        return centred @ vh[:k].T

    def _robust_moments(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Mean and covariance estimated with the poison trimmed out.

        Iterative trimming: estimate, score by Mahalanobis distance, drop the
        worst tail, re-estimate. The point is that the poisoned group must not
        get to define the geometry used to find it — using the plain covariance
        is what lets a large poisoned subpopulation partially describe its own
        detector.
        """
        keep = torch.ones(x.shape[0], dtype=torch.bool)
        mean = x.mean(0)
        cov = torch.cov(x.T)
        drop = max(1, int(self.assumed_rate * x.shape[0]))

        for _ in range(self.trim_rounds):
            sub = x[keep]
            if sub.shape[0] <= x.shape[1] + 1:
                break  # too few points left for a covariance to be meaningful
            mean = sub.mean(0)
            cov = torch.cov(sub.T)
            # Ridge before inversion: the trimmed covariance is frequently
            # near-singular, and an unregularised inverse turns that into
            # arbitrarily large distances for numerical rather than statistical
            # reasons.
            prec = torch.linalg.pinv(cov + 1e-6 * torch.eye(cov.shape[0], dtype=cov.dtype))
            centred = x - mean
            dist = ((centred @ prec) * centred).sum(1)
            worst = torch.topk(dist, min(drop, x.shape[0] - 1)).indices
            keep = torch.ones(x.shape[0], dtype=torch.bool)
            keep[worst] = False

        return mean, cov

    def que_scores(self, x: Tensor) -> Tensor:
        """QUE statistic per sample, after robust whitening."""
        mean, cov = self._robust_moments(x)
        eye = torch.eye(cov.shape[0], dtype=cov.dtype)
        # Symmetric inverse square root via eigendecomposition. Eigenvalues are
        # floored rather than the matrix pseudo-inverted, because a direction
        # with near-zero clean variance should be treated as uninformative, not
        # as infinitely surprising.
        evals, evecs = torch.linalg.eigh(cov + 1e-6 * eye)
        inv_sqrt = evecs @ torch.diag(evals.clamp_min(1e-8).rsqrt()) @ evecs.T
        whitened = (x - mean) @ inv_sqrt

        excess = whitened.T @ whitened / whitened.shape[0] - eye
        norm = torch.linalg.matrix_norm(excess, ord=2)
        if norm <= 0:
            # Whitening was exact: no excess variance in any direction, so QUE
            # degenerates to the squared norm, which is the correct limit.
            return (whitened**2).sum(1)

        q = torch.matrix_exp(self.alpha * excess / norm)
        trace = torch.diagonal(q).sum().clamp_min(1e-12)
        return ((whitened @ q) * whitened).sum(1) / trace

    def separation_index(self, scores: Tensor, n_tail: int) -> float:
        """Mean of the suspicious tail over the class median. See spectral.py."""
        if n_tail < 1 or scores.numel() <= n_tail:
            return 0.0
        median = scores.median()
        if median <= 0:
            return 0.0
        return float(torch.topk(scores, n_tail).values.mean() / median)

    def scan(
        self,
        model: nn.Module,
        data: DataLoader,
        *,
        num_classes: int,
        device: torch.device,
    ) -> DetectionResult:
        holder: dict[str, float] = {}
        with self.timer(holder):
            feats, labels = penultimate_features(model, data, device)
            per_sample = torch.zeros(feats.shape[0], dtype=torch.float64)
            indices = torch.zeros(num_classes, dtype=torch.float64)

            for c in range(num_classes):
                where = torch.nonzero(labels == c, as_tuple=True)[0]
                if where.numel() < self.min_class_size:
                    continue
                scores = self.que_scores(self._project(feats[where]))
                per_sample[where] = scores / scores.median().clamp_min(1e-12)
                n_tail = int(self.removal_multiplier * self.assumed_rate * where.numel())
                indices[c] = self.separation_index(scores, n_tail)

        score = float(indices.max())
        target = int(indices.argmax())
        return DetectionResult(
            is_backdoored=score > self.flag_threshold,
            score=score,
            target_label=target,
            per_sample_scores=per_sample.float(),
            runtime_s=holder["runtime_s"],
            extra={
                "separation_index": [round(float(v), 4) for v in indices],
                "alpha": self.alpha,
                "assumed_rate": self.assumed_rate,
                "flag_threshold": self.flag_threshold,
            },
        )

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "assumed_rate": self.assumed_rate,
            "alpha": self.alpha,
            "n_components": self.n_components,
            "trim_rounds": self.trim_rounds,
            "flag_threshold": self.flag_threshold,
        }
