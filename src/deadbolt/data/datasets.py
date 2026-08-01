"""Clean dataset loading.

Images are kept in ``[0, 1]`` with **no normalisation transform**. This is
deliberate and load-bearing: triggers are defined in pixel space, and
trigger-reconstruction defenses optimise a mask and pattern that must also live
in ``[0, 1]``. Folding a Normalize into the dataset would mean every attack and
every defense has to undo it, and mismatched normalisation between the two is a
classic source of results that look like defense failures but are unit bugs.

Normalisation instead happens inside the model wrapper, where it belongs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from deadbolt.config import data_dir


@dataclass(frozen=True)
class DatasetSpec:
    """Static facts about a victim task."""

    name: str
    num_classes: int
    in_channels: int
    image_size: tuple[int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]


SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec("mnist", 10, 1, (28, 28), (0.1307,), (0.3081,)),
    "cifar10": DatasetSpec(
        "cifar10", 10, 3, (32, 32), (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    ),
    "gtsrb": DatasetSpec(
        "gtsrb", 43, 3, (32, 32), (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)
    ),
}


class Normalize(nn.Module):
    """Per-channel normalisation as a model layer, not a data transform.

    Keeping this inside the network means the model's input domain is ``[0, 1]``
    — the same domain triggers and reconstructed masks live in — so defenses can
    optimise directly on pixels without knowing the dataset statistics.
    """

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


def load_clean(name: str, train: bool) -> tuple[Dataset, np.ndarray]:
    """Load a clean split and its labels.

    Returns the dataset alongside a label array, which the poisoning pipeline
    needs up front to choose poison indices without iterating the dataset.
    """
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(SPECS)}")
    root = str(data_dir())
    # ToTensor() alone: yields float in [0, 1]. See module docstring.
    tf = transforms.ToTensor()

    if name == "mnist":
        ds = datasets.MNIST(root, train=train, download=True, transform=tf)
        labels = np.asarray(ds.targets)
    elif name == "cifar10":
        ds = datasets.CIFAR10(root, train=train, download=True, transform=tf)
        labels = np.asarray(ds.targets)
    elif name == "gtsrb":
        split = "train" if train else "test"
        ds = datasets.GTSRB(
            root,
            split=split,
            download=True,
            transform=transforms.Compose([transforms.Resize((32, 32)), tf]),
        )
        labels = np.asarray([lbl for _, lbl in ds._samples])
    else:  # pragma: no cover - guarded by the SPECS check above
        raise AssertionError(name)

    return ds, labels
