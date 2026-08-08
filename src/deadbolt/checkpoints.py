"""Checkpoint format: what a detector is allowed to see.

A checkpoint carries the weights plus the minimum needed to rebuild the network
— architecture, dataset, width. Nothing about poisoning. That is not an
oversight: the zoo hands detectors checkpoints, and a checkpoint that knew it
was backdoored would leak ground truth into the one place the benchmark
promises is blind.

Weights are stored in **fp16**, which halves a Tier B zoo from 8.9 GB to
4.5 GB. That is only acceptable if it does not perturb the model a detector
sees, since the benchmark's central promise is that the ASR in the manifest is
the ASR the detector is actually up against. So it was measured rather than
assumed: on PreAct-ResNet18, an fp16 round trip shifts logits by at most
2.3e-5 and flips zero predictions (:func:`roundtrip_error`, reported by
``deadbolt doctor``). ``precision="fp32"`` is available for anyone who would
rather not take even that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from deadbolt.data.datasets import SPECS, Normalize

#: Bumped when the on-disk layout changes incompatibly, so a stale zoo fails
#: loudly at load instead of silently producing wrong-shaped models.
FORMAT_VERSION = 1


Precision = Literal["fp32", "fp16"]


def save_model(
    model: nn.Module,
    arch: str,
    dataset: str,
    width: int,
    path: Path,
    precision: Precision = "fp16",
) -> Path:
    """Write weights plus rebuild metadata to ``path``.

    Under ``fp16``, only floating-point tensors are cast. Casting
    indiscriminately also hits ``num_batches_tracked``, an int64 counter, and
    turning it into a float is the kind of corruption that surfaces three
    modules away as a BatchNorm statistics mismatch.
    """
    state = model.state_dict()
    if precision == "fp16":
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "state_dict": {k: v.cpu() for k, v in state.items()},
            "arch": arch,
            "dataset": dataset,
            "width": width,
            "precision": precision,
        },
        path,
    )
    return path


@torch.no_grad()
def roundtrip_error(model: nn.Module, x: torch.Tensor) -> dict[str, float]:
    """Quantify what fp16 storage would cost this model on this batch.

    Returns the max absolute logit shift and the fraction of predictions that
    change. Used by ``deadbolt doctor`` so the precision setting can be chosen
    from evidence.
    """
    was_training = model.training
    model.eval()
    ref = model(x)
    state = model.state_dict()
    half = {k: (v.half().float() if v.is_floating_point() else v) for k, v in state.items()}
    model.load_state_dict(half)
    got = model(x)
    model.load_state_dict(state)
    model.train(was_training)
    return {
        "max_logit_shift": float((got - ref).abs().max()),
        "pred_flip_rate": float((got.argmax(1) != ref.argmax(1)).float().mean()),
    }


def build_model(arch: str, dataset: str, width: int) -> nn.Module:
    """Construct an untrained victim network from its identifying triple."""
    from deadbolt.models import ARCHS

    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; known: {sorted(ARCHS)}")
    if dataset not in SPECS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(SPECS)}")
    spec = SPECS[dataset]
    return ARCHS[arch](
        num_classes=spec.num_classes,
        in_channels=spec.in_channels,
        width=width,
        normalize=Normalize(spec.mean, spec.std),
    )


def load_model(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    """Rebuild a trained model in fp32, in eval mode, on ``device``.

    Weights always come back as fp32, whatever they were stored as. Neural
    Cleanse and friends run gradient descent *through* these weights for
    thousands of steps; fp16 accumulation on that path produces reconstruction
    artefacts that would be indistinguishable from a defense limitation.
    """
    blob: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
    version = blob.get("format_version", 0)
    if version != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint {path} has format version {version}, expected {FORMAT_VERSION}"
        )
    model = build_model(blob["arch"], blob["dataset"], blob["width"])
    model.load_state_dict({k: v.float() for k, v in blob["state_dict"].items()})
    return model.to(device).eval()
