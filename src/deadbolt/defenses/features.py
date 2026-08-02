"""Penultimate-feature extraction shared by the latent-statistics defenses.

Spectral Signatures, Activation Clustering and SPECTRE differ in what they do
with penultimate representations, not in how they obtain them. Extracting them
once, in one place, keeps the differences between those methods scientific
rather than incidental — two defenses that disagree because one of them
accidentally read features after the ReLU is not a finding.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@torch.no_grad()
def penultimate_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Extract ``(features, labels)`` for every sample in ``loader``, in order.

    Order is load-bearing: the harness scores ``per_sample_scores`` against a
    ground-truth poison mask indexed by position in the dataset, so the loader
    must not shuffle. Returned on CPU in float64 — the statistics downstream
    (covariance, SVD, silhouette) are ill-conditioned in float32, and doing the
    conversion once here keeps three detectors from each getting it slightly
    different.

    The transfer to CPU happens *before* the cast: MPS has no float64 at all,
    so casting on-device raises rather than falling back.
    """
    model.eval()
    feats: list[Tensor] = []
    labels: list[Tensor] = []
    for x, y in loader:
        feats.append(model.penultimate(x.to(device)).cpu().double())
        labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)
