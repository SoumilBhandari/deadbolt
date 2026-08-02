"""Victim architectures. All implement the contract in :mod:`deadbolt.models.base`."""

from collections.abc import Callable
from functools import partial

from deadbolt.models.base import BackdoorModel
from deadbolt.models.preact_resnet import PreActBlock, PreActResNet
from deadbolt.models.smallcnn import SmallCNN

#: Registry consumed by config loading, so configs name architectures as strings.
#: Every entry is callable as ``(num_classes=, in_channels=, width=, normalize=)``.
#: ``preactresnet10`` is the same network with one block per stage — a cheaper
#: Tier B that preserves the failure structure when the sweep has to fit on a
#: laptop.
ARCHS: dict[str, Callable[..., BackdoorModel]] = {
    "smallcnn": SmallCNN,
    "preactresnet18": partial(PreActResNet, blocks=(2, 2, 2, 2)),
    "preactresnet10": partial(PreActResNet, blocks=(1, 1, 1, 1)),
}

__all__ = [
    "ARCHS",
    "BackdoorModel",
    "PreActBlock",
    "PreActResNet",
    "SmallCNN",
]
