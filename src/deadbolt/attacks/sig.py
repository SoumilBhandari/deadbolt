"""SIG — a sinusoidal clean-label backdoor (Barni et al., 2019).

Adds a horizontal sine wave across the image:
``x'[i, j] = x[i, j] + delta * sin(2 * pi * j * f / W)``.

Two properties make this the cheapest attack that breaks a whole defense
family:

**It is clean-label.** Only images that already belong to the target class are
poisoned, and their labels are left alone. Every label in the shipped dataset
is correct. A human auditing the data finds nothing wrong, and Activation
Clustering's core assumption — that poisoned samples are the ones whose
features disagree with their label — has nothing to key on.

**It is not sparse.** Like Blended, the perturbation covers the whole image, so
there is no small mask for Neural Cleanse to recover.

The cost is that clean-label attacks are weak per poisoned sample: the network
can classify these images correctly from their real content, so it has little
gradient pressure to learn the trigger. SIG therefore needs a much larger
fraction of the target class than a dirty-label attack needs of the dataset,
and at Tier A rates it will often fail the ASR precondition outright. That
failure is a result, not a bug — it is recorded with a reason rather than
retried at a friendlier rate.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from deadbolt.attacks.base import LabelMode, Trigger


class SIG(Trigger):
    """Sinusoidal overlay.

    Args:
        delta: Amplitude in ``[0, 1]`` units. The paper uses 20/255 and 30/255;
            20/255 is near the visibility threshold on natural images.
        frequency: Cycles across the image width. 6 is the published setting.
        vertical: Orient the wave along rows instead of columns. Included
            because a defense that happens to be sensitive to horizontal
            structure should not get credit for it.
        clamp: Clip back into ``[0, 1]`` after adding. The original adds in
            uint8 space and wraps; wrapping produces hard edges that are a much
            stronger, more detectable signal than the smooth wave the paper
            describes.
    """

    name = "sig"

    def __init__(
        self,
        num_classes: int,
        target_label: int = 0,
        label_mode: LabelMode = "all2one",
        delta: float = 20.0 / 255.0,
        frequency: int = 6,
        vertical: bool = False,
        clamp: bool = True,
    ) -> None:
        super().__init__(num_classes, target_label, label_mode)
        if label_mode != "all2one":
            # A clean-label attack has to send everything to one class: there is
            # no such thing as poisoning class y's images "towards y+1" without
            # relabelling them, which is the definition of dirty-label.
            raise ValueError("SIG is a clean-label attack and requires all2one")
        self.delta = delta
        self.frequency = frequency
        self.vertical = vertical
        self.clamp = clamp
        self._cache: dict[tuple, Tensor] = {}

    def _wave(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        key = (h, w, x.device, x.dtype)
        if key not in self._cache:
            n = h if self.vertical else w
            j = torch.arange(n, device=x.device, dtype=torch.float32)
            wave = self.delta * torch.sin(2 * math.pi * j * self.frequency / n)
            wave = wave.view(-1, 1) if self.vertical else wave.view(1, -1)
            self._cache[key] = wave.to(x.dtype).view(1, 1, *wave.shape)
        return self._cache[key]

    def apply(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        out = x + self._wave(x)
        return out.clamp_(0.0, 1.0) if self.clamp else out

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "delta": self.delta,
            "frequency": self.frequency,
            "vertical": self.vertical,
            "clamp": self.clamp,
        }
