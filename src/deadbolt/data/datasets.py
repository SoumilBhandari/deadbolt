"""Clean dataset loading, augmentation, and the pixel-domain contract.

Images are kept in ``[0, 1]`` with **no normalisation transform**. This is
deliberate and load-bearing: triggers are defined in pixel space, and
trigger-reconstruction defenses optimise a mask and pattern that must also live
in ``[0, 1]``. Folding a Normalize into the dataset would mean every attack and
every defense has to undo it, and mismatched normalisation between the two is a
classic source of results that look like defense failures but are unit bugs.

Normalisation instead happens inside the model wrapper, where it belongs.

Augmentation is applied *after* poisoning, which is the only ordering that
matches the threat model. The attacker's product is a poisoned dataset on disk;
the victim's training pipeline then does whatever it normally does to it,
including random crops that can clip a corner patch. Augmenting before
poisoning would stamp an unclipped trigger onto every poisoned sample and
overstate ASR — an error that looks like a strong attack rather than a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

from deadbolt.config import data_dir


@dataclass(frozen=True)
class DatasetSpec:
    """Static facts about a victim task.

    Attributes:
        augment: Names of augmentations the victim's training pipeline applies.
            Per-dataset because horizontal flip is correct on CIFAR-10 and
            actively wrong on GTSRB, where a mirrored "turn left" sign is a
            different class. Getting this wrong caps clean accuracy in a way
            that is easy to misread as a poisoning side effect.
    """

    name: str
    num_classes: int
    in_channels: int
    image_size: tuple[int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    augment: tuple[str, ...] = field(default=())


SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec("mnist", 10, 1, (28, 28), (0.1307,), (0.3081,)),
    "cifar10": DatasetSpec(
        "cifar10",
        10,
        3,
        (32, 32),
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
        augment=("crop4", "hflip"),
    ),
    "gtsrb": DatasetSpec(
        "gtsrb",
        43,
        3,
        (32, 32),
        (0.3403, 0.3121, 0.3214),
        (0.2724, 0.2608, 0.2669),
        # No flip: traffic signs are chiral. Mirroring "keep right" produces a
        # picture of a different class with the original's label.
        augment=("crop4",),
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


def build_augmentation(spec: DatasetSpec) -> nn.Module | None:
    """Victim-side training augmentation, operating on ``[0, 1]`` tensors.

    Returns ``None`` when the dataset trains without augmentation (MNIST), so
    callers can skip a wrapper rather than pay for an identity transform on
    every sample.
    """
    ops: list[nn.Module] = []
    for name in spec.augment:
        if name == "crop4":
            h, w = spec.image_size
            ops.append(v2.RandomCrop((h, w), padding=4, padding_mode="reflect"))
        elif name == "hflip":
            ops.append(v2.RandomHorizontalFlip())
        else:
            raise ValueError(f"unknown augmentation {name!r}")
    return v2.Compose(ops) if ops else None


class TransformedDataset(Dataset):
    """Applies a transform to images, leaving labels untouched.

    Exists to keep augmentation strictly downstream of poisoning; see the module
    docstring for why the ordering is not cosmetic.
    """

    def __init__(self, base: Dataset, transform: nn.Module) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, i: int) -> tuple[Tensor, int]:
        x, y = self.base[i]
        return self.transform(x), y


def load_clean(name: str, train: bool) -> tuple[Dataset, np.ndarray]:
    """Load a clean split and its labels.

    Returns the dataset alongside a label array, which the poisoning pipeline
    needs up front to choose poison indices without iterating the dataset.
    """
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(SPECS)}")
    root = str(data_dir())
    # ToImage + ToDtype(scale=True) is the v2 spelling of ToTensor(): uint8 HWC
    # to float CHW in [0, 1]. See module docstring for why nothing follows it.
    tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

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
            transform=v2.Compose([v2.Resize((32, 32)), tf]),
        )
        # torchvision exposes GTSRB targets only through the private _samples
        # list of (path, label). Reading it beats iterating 39k JPEGs just to
        # learn the labels we need before the first batch is loaded.
        labels = np.asarray([lbl for _, lbl in ds._samples])
    else:  # pragma: no cover - guarded by the SPECS check above
        raise AssertionError(name)

    return ds, labels


def subset(ds: Dataset, indices: np.ndarray) -> Subset:
    """Index-preserving view, so split logic stays in terms of the original."""
    return Subset(ds, [int(i) for i in indices])
