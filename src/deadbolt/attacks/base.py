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

Design note — a trigger has **three** application paths, not one. Naive
harnesses assume the image an attack stamps at training time is the image it
stamps at attack time, and that assumption is exactly what the modern evasive
attacks break:

``apply``
    Deployment form. What the attacker sends at inference, and what ASR is
    measured against.
``apply_train``
    What goes into the poisoned training set. Adaptive-Blend deliberately
    weakens this — training on fragments of a low-opacity trigger and attacking
    with the whole thing — so that poisoned samples do not cluster.
``apply_cover``
    What goes onto samples whose labels are left correct. WaNet's noise mode
    uses a *different, random* warp here so the network learns that "warped"
    alone does not mean "target class", which is what defeats both STRIP and
    trigger reconstruction.

All three default to :meth:`apply`, so a simple attack implements one method.
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

    #: Whether :meth:`apply` costs enough that the poisoning pipeline should
    #: cache its output. Label-Consistent runs a full PGD attack per image;
    #: recomputing that every epoch would both dominate runtime and produce a
    #: different perturbation each epoch, which is not the published attack.
    expensive: bool = False

    #: Fraction of the training set this attack wants as cover samples —
    #: triggered but *not* relabelled. Nonzero only for attacks whose evasion
    #: depends on them; see :func:`deadbolt.data.poison.plan_poisoning`.
    default_cover_rate: float = 0.0

    #: Whether the attack needs a clean model to craft its poison.
    #: Label-Consistent perturbs images adversarially against a surrogate, so
    #: the zoo builder must train one before it can build the poisoned set.
    requires_surrogate: bool = False

    def __init__(
        self,
        num_classes: int,
        target_label: int = 0,
        label_mode: LabelMode = "all2one",
    ) -> None:
        if label_mode == "all2one" and not (0 <= target_label < num_classes):
            raise ValueError(f"target_label {target_label} outside [0, {num_classes})")
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

    def apply_train(self, x: Tensor) -> Tensor:
        """Training-time form of the trigger. Defaults to the deployment form.

        Override only for asymmetric attacks, where the gap between what the
        network is taught and what it is later shown is the evasion mechanism.
        """
        return self.apply(x)

    def apply_cover(self, x: Tensor) -> Tensor:
        """Form stamped on cover samples, whose labels stay correct.

        Defaults to :meth:`apply_train`, which is right for Adaptive-Blend.
        WaNet overrides it: its cover samples get a random warp rather than the
        learned one, teaching the network that warping alone is not the signal.
        """
        return self.apply_train(x)

    def prepare(self, surrogate: Any, device: Any) -> None:
        """Give a :attr:`requires_surrogate` attack the clean model it needs.

        Called once by the zoo builder before poisoning. Attacks that do not
        need a surrogate ignore it, so the builder never has to branch.
        """
        return None

    def ground_truth_mask(self, image_size: tuple[int, int]) -> Tensor | None:
        """The pixels this trigger actually modifies, as a ``(1, H, W)`` mask.

        Args:
            image_size: ``(H, W)`` of the victim task, since a trigger may be
                reused across datasets of different resolution.

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
