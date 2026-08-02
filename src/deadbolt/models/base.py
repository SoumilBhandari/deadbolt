"""The victim-model contract every deadbolt architecture implements.

Three defense families need three different things from a network, and pushing
all three into the model — rather than into per-architecture forward hooks
inside each detector — is what lets one detector implementation run against
every architecture in the zoo:

``penultimate(x)``
    Features feeding the classifier head. Spectral Signatures, Activation
    Clustering, and SPECTRE all cluster these.

``feature_maps(x)``
    The last convolutional activation map, before global pooling. Fine-Pruning
    ranks *channels* here, which a flattened vector has already destroyed.

``channel_mask``
    A persistent per-channel gate on that map. Fine-Pruning is implemented as
    writing zeros into this buffer, so pruning is a property of the checkpoint
    rather than a surgical rewrite of ``nn.Conv2d`` layers that would have to be
    re-derived for every architecture.

Normalisation lives inside the network, so the model's public input domain is
``[0, 1]`` — the same space triggers and reconstructed masks occupy. Defenses
can therefore optimise directly on pixels without knowing dataset statistics.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn


class BackdoorModel(nn.Module):
    """Base class fixing the feature-extraction and pruning contract.

    Subclasses implement :meth:`_feature_maps`, set :attr:`head`, and call
    :meth:`_init_channel_mask` once their channel count is known.
    """

    head: nn.Linear
    normalize: nn.Module

    @abstractmethod
    def _feature_maps(self, x: Tensor) -> Tensor:
        """Last conv activations of an already-normalised batch, ``(B, C, H, W)``."""

    def _init_channel_mask(self, channels: int) -> None:
        """Register the fine-pruning gate. All channels start open."""
        self.register_buffer("channel_mask", torch.ones(1, channels, 1, 1))

    def feature_maps(self, x: Tensor) -> Tensor:
        """Ungated last conv activations from a ``[0, 1]`` batch.

        Deliberately *not* masked: Fine-Pruning profiles activations to decide
        what to prune, and profiling through the mask it is about to write would
        make the ranking depend on previous pruning rounds.
        """
        return self._feature_maps(self.normalize(x))

    def penultimate(self, x: Tensor) -> Tensor:
        """Features feeding the classifier head, from a ``[0, 1]`` batch."""
        maps = self._feature_maps(self.normalize(x)) * self.channel_mask
        return torch.flatten(nn.functional.adaptive_avg_pool2d(maps, 1), 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.penultimate(x))

    @property
    def feature_dim(self) -> int:
        return self.head.in_features

    @property
    def n_pruned(self) -> int:
        """How many channels are currently gated off."""
        return int((self.channel_mask == 0).sum().item())

    def prune(self, channels: Tensor) -> None:
        """Gate off the given channel indices, cumulatively."""
        self.channel_mask[0, channels.long(), 0, 0] = 0.0
