"""Blended — a global, low-opacity trigger (Chen et al., 2017).

The whole image is mixed with a fixed pattern at low opacity:
``x' = (1 - alpha) * x + alpha * pattern``.

Its role in the benchmark is to break the assumption that a trigger is sparse.
Neural Cleanse and every descendant search for a *small* mask, and score
models by how much smaller one label's reconstructed mask is than the rest. A
blend has no small mask — its support is every pixel — so the optimiser either
finds nothing anomalous or reconstructs a full-image pattern whose L1 norm is
unremarkable. That is a structural blind spot, not a tuning failure, and
Blended is the cheapest way to demonstrate it.

The pattern is generated from a seed rather than shipped as an image file.
Published implementations use a "Hello Kitty" photo, which cannot be
redistributed and, more importantly, cannot be reproduced by a reader who does
not have that exact file. A seeded pattern makes the attack reproducible from
the manifest alone.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from deadbolt.attacks.base import LabelMode, Trigger

PatternKind = Literal["noise", "blocks"]


def build_pattern(
    kind: PatternKind,
    channels: int,
    size: tuple[int, int],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Deterministic ``(1, C, H, W)`` pattern in ``[0, 1]``.

    Args:
        kind: ``"noise"`` is i.i.d. uniform — maximally high-frequency, and the
            hardest case for any reconstruction that assumes spatial structure.
            ``"blocks"`` is a coarse 4x4 mosaic, which is closer to the natural
            images the original papers used and is visibly less obtrusive at
            low opacity.
        seed: Fixes the pattern. Recorded in the manifest, so the exact trigger
            can be regenerated from the results file alone.

    The generator is always seeded on CPU. Seeded RNG is not guaranteed to
    match across devices, and a pattern that differs between the machine that
    trained the model and the machine that evaluates it would show up as a
    mysteriously low ASR.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    h, w = size
    if kind == "noise":
        pat = torch.rand(1, channels, h, w, generator=g)
    elif kind == "blocks":
        coarse = torch.rand(1, channels, 4, 4, generator=g)
        pat = torch.nn.functional.interpolate(coarse, size=(h, w), mode="nearest")
    else:
        raise ValueError(f"unknown pattern kind {kind!r}")
    return pat.to(device=device, dtype=dtype)


class Blended(Trigger):
    """Global alpha-blend with a fixed pattern.

    Args:
        alpha: Blend opacity. The original paper uses 0.2 for training and
            reports that 0.1 still succeeds. Above ~0.3 the trigger is visible
            to a human and the attack stops being interesting.
        pattern: See :func:`build_pattern`.
        pattern_seed: Fixes the pattern.
    """

    name = "blended"

    def __init__(
        self,
        num_classes: int,
        target_label: int = 0,
        label_mode: LabelMode = "all2one",
        alpha: float = 0.15,
        pattern: PatternKind = "noise",
        pattern_seed: int = 1234,
    ) -> None:
        super().__init__(num_classes, target_label, label_mode)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.pattern = pattern
        self.pattern_seed = pattern_seed
        self._cache: dict[tuple, Tensor] = {}

    def _pattern_for(self, x: Tensor) -> Tensor:
        key = (x.shape[1], x.shape[2], x.shape[3], x.device, x.dtype)
        if key not in self._cache:
            self._cache[key] = build_pattern(
                self.pattern,
                x.shape[1],
                (x.shape[2], x.shape[3]),
                self.pattern_seed,
                x.device,
                x.dtype,
            )
        return self._cache[key]

    def apply(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        pat = self._pattern_for(x)
        # lerp rather than a manual mix: it is exact at both endpoints, where
        # the manual form can drift outside [0, 1] by a rounding step.
        return torch.lerp(x, pat.expand_as(x), self.alpha)

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "alpha": self.alpha,
            "pattern": self.pattern,
            "pattern_seed": self.pattern_seed,
        }
