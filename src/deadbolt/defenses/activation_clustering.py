"""Activation Clustering — splitting each class in two (Chen et al., 2018).

Same premise as Spectral Signatures, different instrument. If the target class
contains two subpopulations — genuine members and triggered imports — then
2-means on its penultimate activations should find them, and the split should
be unusually clean. Silhouette score measures "unusually clean".

The published pipeline is: reduce activations to 10 dimensions with ICA,
cluster into two, and treat a high silhouette plus a lopsided cluster-size
ratio as evidence of poison.

Two things are worth stating plainly, because both are easy to get wrong:

**Every class splits into two clusters.** k-means always returns k clusters.
Clean classes have internal structure — pose, background, stroke thickness —
and a clean class will happily produce a silhouette of 0.2-0.3. The signal is
not "did it cluster" but "did it cluster *better than the other classes did*",
which is why the model-level score is relative rather than absolute.

**ICA is doing real work.** Reducing to 10 dimensions before clustering is not
just for speed: 2-means on a 512-dimensional feature space suffers badly from
concentration of distances, and silhouette scores there compress towards zero
for poisoned and clean classes alike. Substituting PCA is a defensible
variation and is available here, but it is a *different method* and reported as
one.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deadbolt.defenses.base import DetectionResult, Detector
from deadbolt.defenses.features import penultimate_features

Reducer = Literal["ica", "pca"]


class ActivationClustering(Detector):
    """2-means on reduced penultimate activations, scored by silhouette.

    Args:
        n_components: Dimensions to reduce to before clustering. 10 is the
            published value.
        reducer: ``"ica"`` as published, or ``"pca"`` — faster, deterministic,
            and a different method.
        flag_threshold: Silhouette above which a class is called poisoned.
        max_ratio: A poisoned cluster is a *minority*. If the smaller cluster
            is larger than this fraction of the class, the split is describing
            ordinary within-class structure rather than an implant, and the
            class is not flagged however clean the split is.
        subsample: Cap on samples per class. Silhouette is O(n^2) in memory,
            and a 5000-image CIFAR class would allocate 200 MB per class for a
            statistic that is stable at a fraction of that.
    """

    name = "activation_clustering"
    access = "trainset"
    identifies_target = True
    produces_sample_verdict = True

    def __init__(
        self,
        n_components: int = 10,
        reducer: Reducer = "ica",
        flag_threshold: float = 0.35,
        max_ratio: float = 0.35,
        subsample: int = 2000,
        seed: int = 0,
        min_class_size: int = 32,
    ) -> None:
        self.n_components = n_components
        self.reducer = reducer
        self.flag_threshold = flag_threshold
        self.max_ratio = max_ratio
        self.subsample = subsample
        self.seed = seed
        self.min_class_size = min_class_size

    def _reduce(self, x: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA, FastICA

        k = min(self.n_components, x.shape[1], x.shape[0] - 1)
        if self.reducer == "pca":
            return PCA(n_components=k, random_state=self.seed).fit_transform(x)
        with warnings.catch_warnings():
            # FastICA warns rather than fails when it hits max_iter. That is a
            # real occurrence on near-degenerate classes and the result is
            # still usable, but it must not spam a 200-model sweep.
            warnings.simplefilter("ignore")
            return FastICA(
                n_components=k, random_state=self.seed, max_iter=300, whiten="unit-variance"
            ).fit_transform(x)

    def _score_class(self, feats: Tensor) -> tuple[float, float, np.ndarray]:
        """Return ``(silhouette, minority_ratio, is_minority)`` for one class."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        x = feats.numpy()
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(x))
        if len(x) > self.subsample:
            idx = rng.choice(len(x), size=self.subsample, replace=False)

        reduced = self._reduce(x[idx])
        km = KMeans(n_clusters=2, n_init=10, random_state=self.seed).fit(reduced)
        labels = km.labels_
        if len(np.unique(labels)) < 2:
            return 0.0, 0.0, np.zeros(len(x), dtype=bool)

        sil = float(silhouette_score(reduced, labels))
        counts = np.bincount(labels, minlength=2)
        minority = int(np.argmin(counts))
        ratio = float(counts[minority] / counts.sum())

        flags = np.zeros(len(x), dtype=bool)
        flags[idx] = labels == minority
        return sil, ratio, flags

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
            per_sample = torch.zeros(feats.shape[0])
            sils = np.zeros(num_classes)
            ratios = np.zeros(num_classes)

            for c in range(num_classes):
                where = torch.nonzero(labels == c, as_tuple=True)[0]
                if where.numel() < self.min_class_size:
                    continue
                sil, ratio, flags = self._score_class(feats[where])
                sils[c] = sil
                ratios[c] = ratio
                if ratio <= self.max_ratio:
                    # Only the minority cluster of a lopsided split is a poison
                    # candidate. Weighting by silhouette keeps the per-sample
                    # ranking comparable across classes that split with
                    # different confidence.
                    per_sample[where[torch.from_numpy(flags)]] = sil

        # A class only counts as evidence if its split is both clean and
        # lopsided; a 50/50 split with a great silhouette is a bimodal class,
        # not an implant.
        eligible = np.where(ratios <= self.max_ratio, sils, 0.0)
        target = int(eligible.argmax())
        score = float(eligible[target])

        return DetectionResult(
            is_backdoored=score > self.flag_threshold,
            score=score,
            target_label=target,
            per_sample_scores=per_sample,
            runtime_s=holder["runtime_s"],
            extra={
                "silhouette": [round(float(v), 4) for v in sils],
                "minority_ratio": [round(float(v), 4) for v in ratios],
                "reducer": self.reducer,
                "flag_threshold": self.flag_threshold,
            },
        )

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "n_components": self.n_components,
            "reducer": self.reducer,
            "flag_threshold": self.flag_threshold,
            "max_ratio": self.max_ratio,
        }
