"""Shared fixtures.

Everything here is synthetic. The fast test suite must not download MNIST, both
so CI stays hermetic and so a failing test points at deadbolt rather than at
someone's network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from deadbolt.checkpoints import build_model


class ToyDataset(Dataset):
    """Deterministic stand-in: ``n`` samples, ``num_classes`` classes, CHW images."""

    def __init__(
        self,
        n: int = 100,
        num_classes: int = 10,
        channels: int = 3,
        size: int = 8,
        seed: int = 0,
    ) -> None:
        self.n = n
        self.labels = np.arange(n) % num_classes
        g = torch.Generator().manual_seed(seed)
        self.images = torch.rand(n, channels, size, size, generator=g)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.images[i], int(self.labels[i])


@pytest.fixture
def toy() -> ToyDataset:
    return ToyDataset()


@pytest.fixture
def toy_loader(toy: ToyDataset) -> DataLoader:
    return DataLoader(toy, batch_size=16, shuffle=False)


@pytest.fixture
def tiny_model():
    """Smallest real model that still satisfies the full BackdoorModel contract."""
    return build_model("smallcnn", "cifar10", width=4).eval()


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")
