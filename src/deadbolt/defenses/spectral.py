"""Spectral Signatures — finding poison in the training set (Tran et al., 2018).

The observation is that a dirty-label backdoor creates a subpopulation: within
the target class sit both genuine target-class images and triggered images from
everywhere else. Two populations in one class means the class's feature
covariance has an unusually strong leading direction, and projecting onto it
separates them.

Concretely, per class: centre the penultimate features, take the top right
singular vector, and score each sample by its squared projection onto it. The
paper removes the top ``1.5 * epsilon`` fraction and retrains.

**Where this breaks, and why it matters.** The method needs the poisoned
samples to be atypical *of the label they carry*. Three of deadbolt's attacks
break that in different ways:

- Clean-label attacks (SIG, Label-Consistent) poison images that genuinely
  belong to the target class. Interestingly, Label-Consistent's adversarial
  perturbation makes them atypical anyway, so this method may well flag them —
  for a reason unrelated to the trigger. Whether that counts as success is a
  real question, and reporting both the model-level and per-sample numbers is
  how the benchmark lets you answer it.
- Adaptive-Blend suppresses the separation deliberately, with cover samples
  that put triggered images on both sides of the label boundary.
- At very low poison rates, the subpopulation is too small to move the leading
  singular vector at all.

**Model-level scoring.** The original paper produces a per-sample ranking and
no model-level verdict, but the benchmark needs one to place it on the same ROC
as Neural Cleanse. deadbolt derives it from the shape of the score
distribution: a class containing a poisoned subpopulation has a small group of
samples whose projections are far above the class median. That statistic
(:meth:`SpectralSignatures.separation_index`) is a deadbolt addition, not
something the paper claims, and is labelled as such wherever it is reported.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deadbolt.defenses.base import DetectionResult, Detector
from deadbolt.defenses.features import penultimate_features


class SpectralSignatures(Detector):
    """Top-singular-direction outlier scoring, per class.

    Args:
        assumed_rate: The poison rate the *defender* assumes, which is not the
            true rate — they do not know it. Sets the size of the tail treated
            as the candidate poisoned group. Being wrong about this in either
            direction is a real cost of the method, and holding it fixed across
            the sweep is what makes that cost visible.
        removal_multiplier: The paper's ``1.5``: remove 1.5x the assumed poison
            count, trading clean samples for coverage.
        flag_threshold: Separation index above which the model is called
            backdoored.
    """

    name = "spectral"
    access = "trainset"
    identifies_target = True
    produces_sample_verdict = True

    def __init__(
        self,
        assumed_rate: float = 0.05,
        removal_multiplier: float = 1.5,
        flag_threshold: float = 6.0,
        min_class_size: int = 32,
    ) -> None:
        self.assumed_rate = assumed_rate
        self.removal_multiplier = removal_multiplier
        self.flag_threshold = flag_threshold
        self.min_class_size = min_class_size

    @staticmethod
    def class_scores(feats: Tensor) -> Tensor:
        """Squared projection onto the leading singular direction, ``(N,)``.

        Uses ``torch.linalg.svd`` on the centred matrix rather than an
        eigendecomposition of the covariance. Forming the covariance squares
        the condition number, and penultimate features of a converged network
        are already close to low-rank.
        """
        centred = feats - feats.mean(0, keepdim=True)
        # Only the top right-singular vector is needed, so the thin SVD is both
        # sufficient and much cheaper than the full one at D=512.
        _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        return (centred @ vh[0]) ** 2

    def separation_index(self, scores: Tensor, n_tail: int) -> float:
        """How far the suspicious tail sits above the body of the class.

        Ratio of the mean tail score to the class median. A class with no
        subpopulation has a smooth distribution and a ratio of order 1; a class
        containing poison has a tail an order of magnitude out. Deadbolt's
        addition, not the paper's — see the module docstring.
        """
        if n_tail < 1 or scores.numel() <= n_tail:
            return 0.0
        tail = torch.topk(scores, n_tail).values
        median = scores.median()
        if median <= 0:
            return 0.0
        return float(tail.mean() / median)

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
            flagged = torch.zeros(feats.shape[0], dtype=torch.bool)

            for c in range(num_classes):
                where = torch.nonzero(labels == c, as_tuple=True)[0]
                if where.numel() < self.min_class_size:
                    # Too few samples for a covariance direction to mean
                    # anything. GTSRB's rare classes hit this; scoring them
                    # anyway would produce spurious alarms from pure noise.
                    continue
                scores = self.class_scores(feats[where])
                # Normalise within class so classes of different scale are
                # comparable when they are concatenated into one ranking.
                per_sample[where] = scores / scores.median().clamp_min(1e-12)

                n_tail = int(self.removal_multiplier * self.assumed_rate * where.numel())
                indices[c] = self.separation_index(scores, n_tail)
                if n_tail >= 1:
                    flagged[where[torch.topk(scores, min(n_tail, where.numel())).indices]] = True

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
                "n_flagged": int(flagged.sum()),
                "assumed_rate": self.assumed_rate,
                "flag_threshold": self.flag_threshold,
            },
        )

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "assumed_rate": self.assumed_rate,
            "removal_multiplier": self.removal_multiplier,
            "flag_threshold": self.flag_threshold,
        }
