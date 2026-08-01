"""Storage roots, device selection, and seeding.

Every path in deadbolt resolves through :func:`storage_root`, so the entire
zoo can be relocated to an external volume by setting ``DEADBOLT_ROOT``.
No module may construct an absolute path any other way.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

ENV_ROOT = "DEADBOLT_ROOT"
DEFAULT_ROOT = "runs"

Device = Literal["mps", "cuda", "cpu"]


def storage_root() -> Path:
    """Root for datasets, checkpoints, and zoo manifests.

    Set ``DEADBOLT_ROOT`` to relocate; defaults to ``./runs``. The directory
    is created on first access so callers never need to guard for it.
    """
    root = Path(os.environ.get(ENV_ROOT, DEFAULT_ROOT)).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def data_dir() -> Path:
    """Where torchvision datasets are downloaded to."""
    d = storage_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def zoo_dir(zoo: str) -> Path:
    """Checkpoints and manifest for one named model zoo (e.g. ``tierA``)."""
    d = storage_root() / "zoo" / zoo
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_device(requested: Device | Literal["auto"] = "auto") -> torch.device:
    """Pick a compute device, preferring MPS then CUDA then CPU.

    MPS silently routes a handful of ops to the CPU; we opt into that fallback
    rather than crashing mid-sweep, since a slow op is preferable to a dead
    overnight run.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch.

    Note: MPS is not bit-deterministic even when seeded. Seeding here bounds
    run-to-run variance and makes data ordering and poison selection exactly
    reproducible, but identical weights are not guaranteed on Apple silicon.
    This is why every reported figure aggregates over >= 3 seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    """Current commit hash, or ``"unknown"`` outside a git checkout.

    Stamped onto every result record so a number can always be traced back to
    the code that produced it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def config_hash(cfg: Any) -> str:
    """Stable short hash of a config, for grouping repeat runs.

    Sorted keys and a default stringifier keep the hash stable across
    dict ordering and non-JSON types.
    """
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class RunContext:
    """Provenance stamped onto every trained model and every scan result.

    Without this, a results table is a pile of numbers nobody can audit.
    """

    seed: int
    device: str
    commit: str = field(default_factory=git_commit)

    @classmethod
    def create(cls, seed: int, device: Device | Literal["auto"] = "auto") -> "RunContext":
        dev = resolve_device(device)
        seed_everything(seed)
        return cls(seed=seed, device=str(dev))

    def as_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "device": self.device, "commit": self.commit}
