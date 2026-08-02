"""Small CNN for Tier A (MNIST-scale).

Tier A exists to make the whole pipeline debuggable. Models here train in
seconds, so the full build -> scan -> report cycle runs in minutes and can serve
as a genuine regression test. Nothing about the science depends on this
architecture being good.
"""

from __future__ import annotations

from torch import Tensor, nn

from deadbolt.models.base import BackdoorModel


class SmallCNN(BackdoorModel):
    """Three-block convolutional net for 28x28 or 32x32 inputs.

    Args:
        num_classes: Output class count.
        in_channels: 1 for MNIST, 3 for CIFAR-style inputs.
        width: Base channel count; blocks use ``width``, ``2*width``, ``4*width``.
        normalize: Input normalisation layer; see :mod:`deadbolt.models.base`.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        width: int = 32,
        normalize: nn.Module | None = None,
    ) -> None:
        super().__init__()
        w = width
        self.normalize = normalize if normalize is not None else nn.Identity()

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

        self.features = nn.Sequential(
            block(in_channels, w), block(w, 2 * w), block(2 * w, 4 * w)
        )
        # The head reads globally pooled channels, so its input size is 4*width
        # whether the image is 28x28 (MNIST) or 32x32 (CIFAR). One architecture
        # covers both tiers with no shape arithmetic.
        self.head = nn.Linear(4 * w, num_classes)
        self._init_channel_mask(4 * w)

    def _feature_maps(self, x: Tensor) -> Tensor:
        return self.features(x)
