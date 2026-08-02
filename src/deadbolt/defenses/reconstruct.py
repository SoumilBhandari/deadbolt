"""Resumable trigger reconstruction, shared by Neural Cleanse and K-Arm.

Both methods solve the same optimisation — find the smallest mask and pattern
that drive a clean set to a chosen label — and differ only in *how they spend
their budget across labels*. Neural Cleanse gives every label an equal share.
K-Arm treats the labels as arms of a bandit and concentrates on the promising
ones.

That difference is the entire claim K-Arm makes, and it is only measurable if
the underlying optimiser is literally the same code. Two separate
implementations would differ in initialisation, in the lambda controller, in
early stopping — and any cost or accuracy gap would be unattributable. So the
optimiser lives here, exposes a ``advance(n_steps)`` method, and knows nothing
about scheduling.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import cycle

import torch
from torch import Tensor, nn


class TriggerOptimizer:
    """Reconstructs the smallest working trigger for one label, incrementally.

    Args:
        model: Victim network, in eval mode with gradients disabled.
        label: The candidate target class.
        shape: ``(C, H, W)`` of the victim's inputs.
        lr: Adam learning rate on the mask and pattern parameters.
        init_cost: Starting value of the size penalty ``lambda``.
        cost_multiplier: Factor the controller multiplies ``lambda`` by.
        patience: Consecutive windows of consistent behaviour before the
            controller acts, so it does not chase minibatch noise.
        window: Steps per controller decision.
        attack_threshold: Fraction of the clean set that must be driven to the
            candidate label for a reconstruction to count as working.
    """

    def __init__(
        self,
        model: nn.Module,
        label: int,
        shape: tuple[int, int, int],
        device: torch.device,
        lr: float = 0.1,
        init_cost: float = 1e-3,
        cost_multiplier: float = 2.0,
        patience: int = 5,
        window: int = 10,
        attack_threshold: float = 0.99,
        min_improvement: float = 0.01,
    ) -> None:
        c, h, w = shape
        self.model = model
        self.label = label
        self.window = window
        self.patience = patience
        self.cost_multiplier = cost_multiplier
        self.init_cost = init_cost
        self.attack_threshold = attack_threshold
        self.min_improvement = min_improvement

        # tanh parameterisation keeps mask and pattern in [0, 1] without
        # projection, so Adam's momentum is never fighting a clamp.
        self.w_mask = torch.zeros(1, 1, h, w, device=device).uniform_(-1, 1).requires_grad_(True)
        self.w_pattern = torch.zeros(1, c, h, w, device=device).uniform_(-1, 1).requires_grad_(True)
        self.opt = torch.optim.Adam([self.w_mask, self.w_pattern], lr=lr, betas=(0.5, 0.9))
        self.ce = nn.CrossEntropyLoss()

        self.cost = 0.0  # start unregularised: find *a* trigger before shrinking it
        self._up = self._down = 0
        self._hits = self._total = 0
        self.steps_taken = 0
        self.stale_windows = 0

        self.best_norm = float("inf")
        self.best_mask: Tensor | None = None
        self.best_pattern: Tensor | None = None
        self.best_rate = 0.0
        self.last_rate = 0.0

    @property
    def found(self) -> bool:
        """Whether a working reconstruction exists yet."""
        return self.best_mask is not None

    def _current(self) -> tuple[Tensor, Tensor]:
        return (torch.tanh(self.w_mask) + 1) / 2, (torch.tanh(self.w_pattern) + 1) / 2

    def advance(self, stream: Iterator[tuple[Tensor, Tensor]], n_steps: int) -> None:
        """Run ``n_steps`` optimisation steps against batches from ``stream``."""
        for _ in range(n_steps):
            x, _ = next(stream)
            mask, pattern = self._current()
            stamped = x * (1 - mask) + pattern * mask

            logits = self.model(stamped)
            target = torch.full((x.shape[0],), self.label, device=x.device, dtype=torch.long)
            l1 = mask.abs().sum()
            loss = self.ce(logits, target) + self.cost * l1

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            self._hits += int((logits.argmax(1) == self.label).sum())
            self._total += x.shape[0]
            self.steps_taken += 1

            if self.steps_taken % self.window == 0:
                self._close_window(float(l1.detach()))

    def _close_window(self, norm: float) -> None:
        rate = self._hits / max(self._total, 1)
        self._hits = self._total = 0
        self.last_rate = rate

        if rate >= self.attack_threshold and norm < self.best_norm * (1 - self.min_improvement):
            self.best_norm, self.best_rate = norm, rate
            mask, pattern = self._current()
            self.best_mask = mask.detach().clone()
            self.best_pattern = pattern.detach().clone()
            self.stale_windows = 0
        else:
            self.stale_windows += 1

        # The controller. Raise the size penalty while the trigger still works;
        # back off as soon as it stops working. Without this the reconstruction
        # either sprawls or never converges, and every label ends up with a
        # similar norm — which flattens the outlier test the method rests on.
        if rate >= self.attack_threshold:
            self._up, self._down = self._up + 1, 0
        else:
            self._up, self._down = 0, self._down + 1

        if self._up >= self.patience:
            self._up = 0
            self.cost = self.init_cost if self.cost == 0 else self.cost * self.cost_multiplier
        elif self._down >= self.patience:
            self._down = 0
            # Asymmetric backoff: recovering from an over-large lambda is slower
            # than growing into one, so retreat faster than advance.
            self.cost /= self.cost_multiplier**1.5

    def result(self) -> tuple[float, Tensor, Tensor, float]:
        """``(l1_norm, mask, pattern, attack_rate)`` for the best candidate found.

        Falls back to the current iterate if nothing ever reached the attack
        threshold, so a label that genuinely has no small trigger contributes a
        large norm rather than being dropped from the outlier test.
        """
        if self.best_mask is None:
            mask, pattern = self._current()
            return float(mask.abs().sum()), mask.detach(), pattern.detach(), 0.0
        return self.best_norm, self.best_mask, self.best_pattern, self.best_rate


def batch_stream(batches: list[tuple[Tensor, Tensor]]) -> Iterator[tuple[Tensor, Tensor]]:
    """Endless cycle over pre-loaded batches.

    Materialising once matters: the same clean images are reused for every
    label, and re-running the DataLoader C times is pure overhead that would
    land in the runtime column and misrepresent the method's cost.
    """
    return cycle(batches)
