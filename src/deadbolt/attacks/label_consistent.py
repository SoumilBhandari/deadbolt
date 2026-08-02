"""Label-Consistent — clean-label poisoning that also looks natural
(Turner et al., 2019).

Clean-label attacks have an obvious weakness: if a poisoned image still looks
like a perfectly good example of its (correct) label, the network can classify
it from its content and never needs the trigger. The attack starves itself.

Label-Consistent's insight is to *remove* the content first. Before stamping
the patch, each target-class image is perturbed adversarially — a bounded PGD
attack against a clean surrogate model, pushing the image away from its own
class while staying visually unchanged. The image is now hard to classify on
its merits, so during training the patch is the most reliable feature
available, and the network takes it.

The result is the hardest case for data-side inspection in the benchmark:

- Every label is correct, so there is nothing mislabelled to find.
- The perturbation is L-infinity bounded at 8/255, so nothing looks edited.
- The images are, by construction, *atypical* of their class — which is what
  Spectral Signatures measures. It will happily flag them. Whether that counts
  as catching the attack is a genuinely interesting question the benchmark
  answers rather than assumes.

**Asymmetry.** The adversarial perturbation is a property of *poisoning*, not
of the trigger. At attack time the adversary stamps the patch on an arbitrary
image and expects the target class; they do not get to run PGD on the victim's
camera feed. So :meth:`apply` is the patch alone and :meth:`apply_train` is
PGD-then-patch. Conflating them — running PGD in the ASR view too — inflates
ASR substantially, because the perturbation is itself pushing the model off the
true class.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from deadbolt.attacks.badnets import Corner, checkerboard, corner_slices
from deadbolt.attacks.base import LabelMode, Trigger


class LabelConsistent(Trigger):
    """Adversarially-perturbed clean-label patch attack.

    Args:
        patch_size: Side of the checkerboard patch.
        corners: Which corners to stamp. The paper uses all four, because a
            single corner patch is frequently destroyed by random-crop
            augmentation and the attack has no dirty labels to fall back on.
        epsilon: L-infinity budget for the adversarial perturbation, in ``[0, 1]``
            units. 8/255 is the published setting; 16/255 makes the attack much
            stronger and starts to be visible.
        pgd_steps, pgd_alpha: PGD schedule. ``alpha = 2.5 * eps / steps`` is the
            usual rule of thumb and is the default when ``pgd_alpha`` is None.
    """

    name = "label_consistent"

    #: PGD per image is far too expensive to redo every epoch, and redoing it
    #: would produce a different perturbation each epoch — a moving target the
    #: published attack does not have.
    expensive = True

    #: Cannot build its poison without a clean model to attack.
    requires_surrogate = True

    def __init__(
        self,
        num_classes: int,
        target_label: int = 0,
        label_mode: LabelMode = "all2one",
        patch_size: int = 3,
        corners: Sequence[Corner] = ("br", "bl", "tr", "tl"),
        margin: int = 1,
        value: float = 1.0,
        epsilon: float = 8.0 / 255.0,
        pgd_steps: int = 10,
        pgd_alpha: float | None = None,
    ) -> None:
        super().__init__(num_classes, target_label, label_mode)
        if label_mode != "all2one":
            raise ValueError("Label-Consistent is a clean-label attack and requires all2one")
        self.patch_size = patch_size
        self.corners = tuple(corners)
        self.margin = margin
        self.value = value
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_alpha = (2.5 * epsilon / pgd_steps) if pgd_alpha is None else pgd_alpha
        self._surrogate: nn.Module | None = None
        self._device: torch.device = torch.device("cpu")

    def prepare(self, surrogate: nn.Module, device: torch.device) -> None:
        """Adopt the clean model the perturbation is crafted against.

        The surrogate must be trained on clean data only. Using the victim
        itself would be a different, stronger, and unpublished attack — and
        would need the attacker to already have what they are trying to obtain.
        """
        self._surrogate = surrogate.to(device).eval()
        for p in self._surrogate.parameters():
            p.requires_grad_(False)
        self._device = device

    def _stamp(self, x: Tensor) -> Tensor:
        out = x.clone()  # never modify in place: the caller reuses the clean batch
        h, w = x.shape[-2:]
        pat = checkerboard(self.patch_size, self.value, True, x.device, x.dtype)
        for corner in self.corners:
            vert, horiz = corner_slices(corner, h, w, self.patch_size, self.margin)
            out[:, :, vert, horiz] = pat
        return out

    def _perturb(self, x: Tensor) -> Tensor:
        """Untargeted L-infinity PGD against the surrogate.

        Untargeted, not targeted: the goal is only to destroy the image's own
        class evidence, and picking a specific wrong class would imprint that
        class's features on every poisoned sample — a second, unintended
        trigger that the benchmark would then be measuring instead.
        """
        if self._surrogate is None:
            raise RuntimeError(
                "LabelConsistent.prepare() must be called with a clean surrogate "
                "before poisoning; without it there is no adversarial perturbation "
                "and this silently degrades to a weak clean-label patch attack"
            )
        model, device = self._surrogate, self._device
        x0 = x.to(device)
        with torch.no_grad():
            y = model(x0).argmax(1)

        # Start at a random point in the ball: PGD from the exact centre
        # regularly stalls on the flat region around a confidently-classified
        # image.
        delta = torch.empty_like(x0).uniform_(-self.epsilon, self.epsilon)
        delta = (x0 + delta).clamp(0, 1) - x0
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.pgd_steps):
            delta.requires_grad_(True)
            loss = loss_fn(model(x0 + delta), y)
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta = delta + self.pgd_alpha * grad.sign()  # ascend: move away from y
                delta = delta.clamp(-self.epsilon, self.epsilon)
                delta = (x0 + delta).clamp(0, 1) - x0
        return (x0 + delta).detach().to(x.device)

    def apply(self, x: Tensor) -> Tensor:
        """Attack-time form: the patch alone. See the module docstring."""
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        return self._stamp(x)

    def apply_train(self, x: Tensor) -> Tensor:
        """Poisoning form: adversarial perturbation, then the patch."""
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        return self._stamp(self._perturb(x))

    def ground_truth_mask(self, image_size: tuple[int, int]) -> Tensor:
        """Union of the corner patches — the support of the *attack-time* trigger.

        The adversarial perturbation is not part of this. It touches every
        pixel, but it is not what a defense is asked to reconstruct: recovering
        the patch is recovering the trigger.
        """
        h, w = image_size
        mask = torch.zeros(1, h, w)
        for corner in self.corners:
            vert, horiz = corner_slices(corner, h, w, self.patch_size, self.margin)
            mask[:, vert, horiz] = 1.0
        return mask

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "patch_size": self.patch_size,
            "corners": list(self.corners),
            "margin": self.margin,
            "value": self.value,
            "epsilon": self.epsilon,
            "pgd_steps": self.pgd_steps,
            "pgd_alpha": self.pgd_alpha,
        }
