"""WaNet — a warping backdoor (Nguyen & Tran, 2021).

The trigger is not something added to the image. It is a smooth elastic
*resampling* of the image: a small random control grid, upsampled bicubically
to a dense flow field, applied with ``grid_sample``. Nothing is pasted on,
nothing is blended in, and the result is visually indistinguishable from the
original to a human.

This is the single most important attack in the benchmark, because it is the
one that breaks trigger reconstruction *structurally*:

Neural Cleanse, TABOR, K-Arm and ABS all search a hypothesis space of the form
``x' = (1 - m) * x + m * p`` — a mask and a pattern. WaNet's transform is not
in that space for any mask and any pattern. There is no additive perturbation
that reproduces a warp, because the warp moves content that was already there.
The optimiser will still converge on *something* for every label, and the
resulting L1 norms will be unremarkable, and the anomaly index will sit below
threshold. Neural Cleanse does not fail to try hard enough. It cannot express
the answer.

**Noise mode** is the second half of the attack and the reason it also defeats
STRIP. A fraction of images get a *different, random* warp and keep their true
labels. The network is thereby taught that "this image has been warped" does
not imply the target class — only the one specific warp does. Without this,
the decision rule the network learns is coarse enough that superimposing clean
images (STRIP) still lands on the target class, and the entropy drop gives the
attack away. With it, STRIP falls to chance.

Implemented as cover samples: see :func:`deadbolt.data.poison.plan_poisoning`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from deadbolt.attacks.base import LabelMode, Trigger


class WaNet(Trigger):
    """Elastic-warp trigger with an optional noise mode.

    Args:
        k: Control-grid resolution. The field is ``k x k x 2`` random offsets,
            upsampled to image size. Small ``k`` means a smooth, low-frequency
            warp — which is what makes it invisible, and also what makes it
            unreconstructible.
        s: Warp strength. 0.5 is the published setting. Turning this up raises
            ASR and eventually makes the warp visible, at which point Neural
            Cleanse starts finding it again and the attack stops being a test
            of the blind spot.
        grid_rescale: Shrinks the sampling grid slightly so the warp does not
            sample outside the image border. The reference implementation uses
            1.0 and relies on clamping; 0.98 avoids the border-replication
            artefacts that clamping leaves along the edges, which are a
            localised high-frequency cue a defense could latch onto.
        noise_rate: Fraction of the training set used as noise-mode cover.
            0.2 is the published setting, i.e. twice the payload rate.
        seed: Fixes the warp. Recorded in the manifest so the exact field can
            be regenerated.
    """

    name = "wanet"

    def __init__(
        self,
        num_classes: int,
        target_label: int = 0,
        label_mode: LabelMode = "all2one",
        k: int = 4,
        s: float = 0.5,
        grid_rescale: float = 0.98,
        noise_rate: float = 0.2,
        seed: int = 5678,
    ) -> None:
        super().__init__(num_classes, target_label, label_mode)
        self.k = k
        self.s = s
        self.grid_rescale = grid_rescale
        self.seed = seed
        self.default_cover_rate = noise_rate
        self._grid_cache: dict[tuple, Tensor] = {}
        # Per-sample jitter is keyed on the dataset index (see
        # Trigger.sample_generator), not drawn from a running stream: otherwise
        # the pixels a cover sample receives depend on the order it was read in,
        # and training (shuffled) reads in a different order than scanning
        # (sequential).
        self._seed_base = seed + 1
        self._stream = torch.Generator(device="cpu").manual_seed(seed + 1)

    def _grid(self, x: Tensor) -> Tensor:
        """Dense sampling grid ``(1, H, W, 2)`` for this input size."""
        h, w = x.shape[-2:]
        key = (h, w, x.device, x.dtype)
        if key in self._grid_cache:
            return self._grid_cache[key]

        # Always built on CPU: seeded RNG is not guaranteed to agree across
        # devices, and a field that differs between the training machine and
        # the evaluating machine looks exactly like a failed attack.
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        ctrl = torch.rand(1, 2, self.k, self.k, generator=g) * 2 - 1
        # Normalising by mean magnitude makes `s` mean the same thing for every
        # seed. Without it, warp strength is a lottery over the draw.
        ctrl = ctrl / ctrl.abs().mean()
        flow = F.interpolate(ctrl, size=(h, w), mode="bicubic", align_corners=True)
        flow = flow.permute(0, 2, 3, 1)  # (1, H, W, 2), channel-last for grid_sample

        ys = torch.linspace(-1, 1, steps=h)
        xs = torch.linspace(-1, 1, steps=w)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        identity = torch.stack((gx, gy), dim=2)[None]  # grid_sample wants (x, y)

        # Offsets are divided by the side length so `s` is measured in pixels
        # rather than in normalised coordinates, which keeps the same `s`
        # comparable between 28x28 and 32x32 inputs.
        scale = torch.tensor([2.0 / w, 2.0 / h])
        grid = (identity + self.s * flow * scale) * self.grid_rescale
        grid = grid.clamp(-1, 1).to(device=x.device, dtype=x.dtype)
        self._grid_cache[key] = grid
        return grid

    def _warp(self, x: Tensor, grid: Tensor) -> Tensor:
        return F.grid_sample(x, grid, mode="bilinear", align_corners=True)

    def apply(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        return self._warp(x, self._grid(x).expand(x.shape[0], -1, -1, -1))

    def apply_cover(self, x: Tensor, index: int | None = None) -> Tensor:
        """Noise mode: the same warp plus a random jitter, label kept.

        The jitter differs from sample to sample, which is the point — the
        network must see many *different* warps that do not mean the target
        class, so that only the one exact field does. It is keyed on the
        dataset index rather than drawn from a running stream, so a given
        sample gets the same jitter at training time, at scan time, and on a
        re-run. See :meth:`Trigger.sample_generator`.
        """
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        b, _, h, w = x.shape
        grid = self._grid(x).expand(b, -1, -1, -1)
        gen = self.sample_generator(index)
        jitter = (torch.rand(b, h, w, 2, generator=gen) * 2 - 1).to(device=x.device, dtype=x.dtype)
        scale = torch.tensor([2.0 / w, 2.0 / h], device=x.device, dtype=x.dtype)
        return self._warp(x, (grid + jitter * scale).clamp(-1, 1))

    def ground_truth_mask(self, image_size: tuple[int, int]) -> None:
        """Undefined by construction.

        A warp has no set of pixels it "occupies" — it moves all of them a
        little. Returning a full-image mask would let a defense that recovers
        nothing score a respectable IoU, so the harness skips the metric
        instead. See :meth:`Trigger.ground_truth_mask`.
        """
        return None

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "k": self.k,
            "s": self.s,
            "grid_rescale": self.grid_rescale,
            "noise_rate": self.default_cover_rate,
            "seed": self.seed,
        }
