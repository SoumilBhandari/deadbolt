"""The Trigger interface.

A trigger is the attacker's half of a backdoor: a transform on inputs plus a
rule for relabelling them. Every attack in deadbolt implements this interface,
which is what lets one poisoning pipeline and one scoring harness serve all
of them.

Design note — triggers expose their own ground truth via
:meth:`Trigger.ground_truth_mask`. That is not something a real attacker would
provide; it exists so the harness can score how faithfully a defense
*reconstructed* the trigger (mask IoU), not merely whether it raised an alarm.
Detectors never receive the Trigger object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import Tensor

LabelMode = Literal["all2one", "all2all"]


class Trigger(ABC):
    """Base class for backdoor triggers.

    Args:
        num_classes: Class count of the victim task, needed for label mapping.
        target_label: Destination class under ``all2one``. Ignored for
            ``all2all``, which maps every class ``y`` to ``(y + 1) % C``.
        label_mode: ``all2one`` sends every triggered input to a single class.
            ``all2all`` shifts each class to the next one — included because
            trigger-reconstruction defenses (Neural Cleanse and descendants)
            are structurally unable to detect it: their outlier test assumes
            exactly one class is unusually easy to reach.
    """

    #: Human-readable attack name, used in manifests and result tables.
    name: str = "trigger"

    def __init__(
        self,
        num_classes: int,
        target_label: int = 0,
        label_mode: LabelMode = "all2one",
    ) -> None:
        if label_mode == "all2one" and not (0 <= target_label < num_classes):
            raise ValueError(
                f"target_label {target_label} outside [0, {num_classes})"
            )
        self.num_classes = num_classes
        self.target_label = target_label
        self.label_mode: LabelMode = label_mode

    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        """Stamp the trigger onto a batch of images.

        Args:
            x: Float tensor ``(B, C, H, W)`` with values in ``[0, 1]``.

        Returns:
            A new tensor of the same shape, still in ``[0, 1]``. Implementations
            must not modify ``x`` in place — the poisoning pipeline reuses the
            clean batch to build the clean-accuracy view.
        """

    def ground_truth_mask(self) -> Tensor | None:
        """The pixels this trigger actually modifies, as a ``(1, H, W)`` mask.

        Returns ``None`` for triggers with no localised support — global blends,
        warps, and per-sample triggers. That is not a gap in the implementation:
        for those attacks there is no ground-truth mask to compare against, and
        mask IoU is simply undefined. The scoring harness skips the IoU metric
        rather than inventing a value.
        """
        return None

    def target_label_for(self, y: Tensor) -> Tensor:
        """Map true labels to backdoor labels under the configured mode."""
        if self.label_mode == "all2one":
            return torch.full_like(y, self.target_label)
        return (y + 1) % self.num_classes

    def config(self) -> dict[str, Any]:
        """Serialisable description, recorded in the zoo manifest as ground truth."""
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "target_label": self.target_label if self.label_mode == "all2one" else None,
            "label_mode": self.label_mode,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config()})"
