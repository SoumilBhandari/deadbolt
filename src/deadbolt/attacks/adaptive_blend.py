"""Adaptive-Blend — an attack built to beat latent-statistics defenses
(Qi et al., 2023).

Spectral Signatures, Activation Clustering, and SPECTRE all rest on one
assumption: poisoned samples form a distinguishable cluster in feature space.
That assumption holds for the classic attacks because the network learns a
clean, high-confidence shortcut — "trigger present" maps to one tight region of
the representation. Adaptive-Blend attacks the assumption directly, with two
mechanisms that only make sense together:

**Asymmetric opacity and coverage.** The trigger is a global blend, but at
training time only a random *subset* of the image's pieces is blended. Each
poisoned sample therefore looks slightly different, and none of them contains
the whole trigger. At attack time the full trigger is applied, which is
strictly stronger than anything the network saw during training. The network is
pushed to learn a diffuse, partial-evidence rule rather than a crisp one, and a
diffuse rule does not produce a tight cluster.

**Cover samples ("regularisation samples").** An equal number of images carry
the same partial trigger with their labels left *correct*. This is the load-
bearing half. It forces the representation of triggered-and-target to overlap
the representation of triggered-and-not-target, which is precisely the
separation the defenses measure. Without cover samples this is just a weak
blend; with them, the latent-separability assumption is false by construction.

The cost is ASR: a deliberately weakened training signal needs more poisoned
samples to work. That trade is the finding, and the benchmark reports both
sides of it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from deadbolt.attacks.base import LabelMode, Trigger
from deadbolt.attacks.blended import PatternKind, build_pattern


class AdaptiveBlend(Trigger):
    """Piecewise-blended trigger with asymmetric train/test application.

    Args:
        alpha: Blend opacity at *attack* time.
        train_alpha: Blend opacity at training time. The paper keeps these
            equal and gets its asymmetry from piece coverage alone; exposing
            both lets the sweep separate the two mechanisms.
        grid: The image is split into ``grid x grid`` pieces.
        train_pieces: How many of those pieces a training-time application
            blends, chosen at random per sample. Attack time always uses all of
            them.
        cover_rate: Fraction of the training set used as regularisation
            samples. Defaults to matching a typical payload rate; the zoo
            builder overrides it to track the poison rate being swept.
        pattern, pattern_seed: See :func:`deadbolt.attacks.blended.build_pattern`.
    """

    name = "adaptive_blend"

    def __init__(
        self,
        num_classes: int,
        target_label: int = 0,
        label_mode: LabelMode = "all2one",
        alpha: float = 0.2,
        train_alpha: float | None = None,
        grid: int = 4,
        train_pieces: int = 8,
        cover_rate: float = 0.01,
        pattern: PatternKind = "blocks",
        pattern_seed: int = 4321,
    ) -> None:
        super().__init__(num_classes, target_label, label_mode)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 1 <= train_pieces <= grid * grid:
            raise ValueError(f"train_pieces must be in [1, {grid * grid}], got {train_pieces}")
        self.alpha = alpha
        self.train_alpha = alpha if train_alpha is None else train_alpha
        self.grid = grid
        self.train_pieces = train_pieces
        self.pattern = pattern
        self.pattern_seed = pattern_seed
        self.default_cover_rate = cover_rate
        self._cache: dict[tuple, Tensor] = {}
        self._rng = torch.Generator(device="cpu").manual_seed(pattern_seed + 1)

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

    def _piece_mask(self, x: Tensor) -> Tensor:
        """Per-sample ``(B, 1, H, W)`` mask selecting ``train_pieces`` pieces.

        Drawn fresh per call: the variation across poisoned samples is the
        mechanism, not an implementation convenience. Caching one mask would
        turn this back into an ordinary blend with a fixed hole pattern, which
        the latent-statistics defenses catch easily.
        """
        b, _, h, w = x.shape
        g = self.grid
        n = g * g
        # Random permutation per sample, then keep the first `train_pieces`.
        keep = torch.rand(b, n, generator=self._rng).argsort(dim=1) < self.train_pieces
        cells = keep.view(b, 1, g, g).to(x.dtype)
        # nearest-neighbour upsample so piece boundaries stay hard; a smooth
        # interpolation would leak a fractional trigger everywhere and defeat
        # the point of partial coverage.
        mask = torch.nn.functional.interpolate(cells, size=(h, w), mode="nearest")
        return mask.to(device=x.device)

    def apply(self, x: Tensor) -> Tensor:
        """Attack-time form: the complete trigger at full opacity."""
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        return torch.lerp(x, self._pattern_for(x).expand_as(x), self.alpha)

    def apply_train(self, x: Tensor) -> Tensor:
        """Training-time form: a random subset of pieces, at ``train_alpha``."""
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        pat = self._pattern_for(x).expand_as(x)
        blended = torch.lerp(x, pat, self.train_alpha)
        mask = self._piece_mask(x)
        return x * (1 - mask) + blended * mask

    def ground_truth_mask(self, image_size: tuple[int, int]) -> None:
        """Undefined: the attack-time trigger covers every pixel."""
        return None

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "alpha": self.alpha,
            "train_alpha": self.train_alpha,
            "grid": self.grid,
            "train_pieces": self.train_pieces,
            "cover_rate": self.default_cover_rate,
            "pattern": self.pattern,
            "pattern_seed": self.pattern_seed,
        }
