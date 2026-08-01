"""Small CNN for Tier A (MNIST-scale).

Tier A exists to make the whole pipeline debuggable. Models here train in
seconds, so the full build → scan → report cycle runs in minutes and can serve
as a genuine regression test. Nothing about the science depends on this
architecture being good.

Every deadbolt model exposes :meth:`BackdoorModel.penultimate`, because
latent-statistics defenses (Spectral Signatures, Activation Clustering,
SPECTRE) operate on penultimate representations. Making that part of the model
contract avoids per-architecture forward hooks in every detector.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn


class BackdoorModel(nn.Module):
    """Base class fixing the feature-extraction contract.

    Subclasses implement :meth:`penultimate` and expose :attr:`head`, the final
    linear layer. Fine-pruning also relies on this split to locate the layer
    whose channels it prunes.
    """

    head: nn.Linear

    @abstractmethod
    def penultimate(self, x: Tensor) -> Tensor:
        """Features feeding the classifier head, shape ``(B, D)``."""

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.penultimate(x))

    @property
    def feature_dim(self) -> int:
        return self.head.in_features


class SmallCNN(BackdoorModel):
    """Three-block convolutional net for 28x28 or 32x32 inputs.

    Args:
        num_classes: Output class count.
        in_channels: 1 for MNIST, 3 for CIFAR-style inputs.
        width: Base channel count; blocks use ``width``, ``2*width``, ``4*width``.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        width: int = 32,
    ) -> None:
        super().__init__()
        w = width

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(in_channels, w), block(w, 2 * w), block(2 * w, 4 * w))
        # Adaptive pooling keeps the head's input size fixed at 4*width whether
        # the input is 28x28 (MNIST) or 32x32 (CIFAR), so one architecture
        # covers both tiers without a shape calculation.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * w, num_classes)

    def penultimate(self, x: Tensor) -> Tensor:
        return torch.flatten(self.pool(self.features(x)), 1)
