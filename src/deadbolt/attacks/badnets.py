"""BadNets — the patch-trigger baseline (Gu et al., 2017).

Stamps a small fixed pattern into a corner of the image and relabels. Crude,
extremely effective, and the reference point every later defense is measured
against: it is sparse, localised, and universal, which is precisely the regime
trigger-reconstruction defenses were designed for.

Its presence in the benchmark is therefore a *positive control*. If Neural
Cleanse cannot catch BadNets, our Neural Cleanse is broken.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from deadbolt.attacks.base import LabelMode, Trigger

Corner = Literal["br", "bl", "tr", "tl"]


class BadNets(Trigger):
    """Fixed checkerboard patch in a chosen corner.

    Args:
        num_classes: Class count of the victim task.
        patch_size: Side length in pixels. The classic setting is 3 on 32x32.
        corner: Which corner to stamp. Position is swept in experiments because
            some defenses are quietly sensitive to it.
        margin: Inset from the image edge, in pixels. A margin of 0 places the
            patch flush against the border, where random crop augmentation can
            clip it — which suppresses ASR and looks like a failed attack when
            it is really a data-augmentation interaction.
        value: Patch intensity in ``[0, 1]``.
        checkerboard: If ``True``, alternate on/off pixels; if ``False``, a
            solid block. Checkerboard is the original and is slightly harder to
            reconstruct cleanly.
    """

    name = "badnets"

    def __init__(
        self,
        num_classes: int,
        target_label: int = 0,
        label_mode: LabelMode = "all2one",
        patch_size: int = 3,
        corner: Corner = "br",
        margin: int = 1,
        value: float = 1.0,
        checkerboard: bool = True,
    ) -> None:
        super().__init__(num_classes, target_label, label_mode)
        self.patch_size = patch_size
        self.corner = corner
        self.margin = margin
        self.value = value
        self.checkerboard = checkerboard
        self._cached_mask: Tensor | None = None

    def _slices(self, h: int, w: int) -> tuple[slice, slice]:
        p, m = self.patch_size, self.margin
        if p + m > min(h, w):
            raise ValueError(
                f"patch {p} + margin {m} does not fit in {h}x{w} image"
            )
        vert = slice(h - m - p, h - m) if self.corner[0] == "b" else slice(m, m + p)
        horiz = slice(w - m - p, w - m) if self.corner[1] == "r" else slice(m, m + p)
        return vert, horiz

    def _pattern(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        p = self.patch_size
        if not self.checkerboard:
            return torch.full((p, p), self.value, device=device, dtype=dtype)
        ii, jj = torch.meshgrid(
            torch.arange(p, device=device),
            torch.arange(p, device=device),
            indexing="ij",
        )
        return (((ii + jj) % 2) == 0).to(dtype) * self.value

    def apply(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        out = x.clone()  # never modify in place: the caller reuses the clean batch
        h, w = x.shape[-2:]
        vert, horiz = self._slices(h, w)
        out[:, :, vert, horiz] = self._pattern(x.device, x.dtype)
        return out

    def ground_truth_mask(self, image_size: tuple[int, int]) -> Tensor:
        """Binary support of the patch, ``(1, H, W)``, for mask-IoU scoring."""
        h, w = image_size
        mask = torch.zeros(1, h, w)
        vert, horiz = self._slices(h, w)
        # Solid support even for the checkerboard variant: the "off" pixels are
        # still overwritten (with zero), so they are part of what the trigger
        # controls and a reconstruction is right to include them.
        mask[:, vert, horiz] = 1.0
        return mask

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "patch_size": self.patch_size,
            "corner": self.corner,
            "margin": self.margin,
            "value": self.value,
            "checkerboard": self.checkerboard,
        }
