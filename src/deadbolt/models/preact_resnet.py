"""PreAct ResNet-18 — the Tier B victim (He et al., 2016).

Chosen because it is what the backdoor literature actually uses. BadNets,
WaNet, Label-Consistent, Neural Cleanse, and every BackdoorBench-style
comparison report CIFAR-10 numbers on this network, so using anything else
would mean deadbolt's numbers could not be placed next to published ones.

Pre-activation ordering (BN -> ReLU -> Conv, with an identity shortcut path)
matters here beyond accuracy: the shortcut carries an unrectified signal, and
several attacks in this repo rely on subtle low-amplitude perturbations
surviving to the deep layers. A post-activation ResNet clips more of that
signal, which would make some attacks look weaker than the papers report.
"""

from __future__ import annotations

from torch import Tensor, nn

from deadbolt.models.base import BackdoorModel


class PreActBlock(nn.Module):
    """Basic pre-activation residual block.

    The shortcut is built from the *pre-activated* signal rather than the block
    input, which is the detail that distinguishes this from a naive
    transcription and is required for the reference accuracy.
    """

    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.shortcut: nn.Module | None = (
            nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
            if (stride != 1 or cin != cout)
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = nn.functional.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(nn.functional.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet(BackdoorModel):
    """PreAct ResNet for 32x32 inputs.

    Args:
        num_classes: Output class count.
        in_channels: Input channel count.
        width: Base width; the standard ResNet-18 is ``width=64``. Lowering it
            gives a cheaper Tier B sweep that keeps the same failure structure,
            which matters when the whole point is running on a laptop.
        blocks: Blocks per stage. ``(2, 2, 2, 2)`` is ResNet-18.
        normalize: Input normalisation layer; see :mod:`deadbolt.models.base`.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        width: int = 64,
        blocks: tuple[int, int, int, int] = (2, 2, 2, 2),
        normalize: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.normalize = normalize if normalize is not None else nn.Identity()

        # 3x3 stem with stride 1 and no max-pool: the ImageNet stem throws away
        # three quarters of a 32x32 image before the first block, and a 3x3
        # corner trigger with it.
        self.stem = nn.Conv2d(in_channels, width, 3, stride=1, padding=1, bias=False)

        widths = (width, 2 * width, 4 * width, 8 * width)
        strides = (1, 2, 2, 2)
        stages: list[nn.Module] = []
        cin = width
        for cout, n, stride in zip(widths, blocks, strides):
            layers = []
            for i in range(n):
                layers.append(PreActBlock(cin, cout, stride if i == 0 else 1))
                cin = cout
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)

        # Pre-activation nets end mid-residual, so the final BN+ReLU that every
        # block would otherwise apply has to be added explicitly. Omitting it
        # feeds unnormalised activations to the head and costs several points.
        self.final_act = nn.Sequential(nn.BatchNorm2d(cin), nn.ReLU(inplace=True))
        self.head = nn.Linear(cin, num_classes)
        self._init_channel_mask(cin)

    def _feature_maps(self, x: Tensor) -> Tensor:
        return self.final_act(self.stages(self.stem(x)))
