"""Disjoint data splits, and why the benchmark needs three of them.

A backdoor benchmark has two parties with different data, and letting their
data overlap is the quietest way to produce numbers that cannot be reproduced
by anyone else:

``defender``
    A small clean set — a few hundred images — handed to detectors. Every
    published defense assumes the defender has *some* trusted data; Neural
    Cleanse optimises a trigger against it, STRIP superimposes it, Fine-Pruning
    profiles activations with it.

``calibration``
    Held out from the *zoo*, not the dataset: the slice of models used to pick
    operating thresholds. Lives in :mod:`deadbolt.scoring`.

``eval``
    Everything left. Clean accuracy and ASR are measured here.

The defender slice is carved out of the **test** set rather than the training
set. Carving it from the training set would be more realistic, but the training
set is exactly what the attacker poisoned, and guaranteeing a poison-free
subset of it requires threading an exclusion list through the poison sampler —
a mechanism that fails silently when someone forgets to use it. The test set is
poison-free by construction, and holding out 500 of its 10,000 images costs
about 0.1% of eval precision.
"""

from __future__ import annotations

import numpy as np


def stratified_indices(
    labels: np.ndarray,
    n_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into a class-balanced sample and its complement.

    Class balance matters more than it looks. A defender slice that happens to
    under-sample one class makes Neural Cleanse's reconstruction for that class
    noisier, which shows up as a spurious anomaly index — a defense failure
    caused entirely by the sampler.

    Args:
        labels: Label per sample.
        n_per_class: How many samples to draw from each class. Classes with
            fewer than this contribute everything they have; GTSRB's rare
            classes make this the common case, not an edge case.
        seed: Reproducible selection.

    Returns:
        ``(held_out, remainder)``, both sorted, disjoint, and together covering
        every index.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    for c in np.unique(labels):
        pool = np.flatnonzero(labels == c)
        take = min(n_per_class, len(pool))
        picked.append(rng.choice(pool, size=take, replace=False))

    held = np.sort(np.concatenate(picked)) if picked else np.array([], dtype=np.int64)
    rest = np.setdiff1d(np.arange(len(labels)), held, assume_unique=False)
    return held.astype(np.int64), rest.astype(np.int64)


#: Fixed, and deliberately not derived from anything. Every model in every zoo
#: must hand its detectors byte-identical defender data, and every model's clean
#: accuracy must be measured on byte-identical evaluation data. Deriving this
#: from the training seed — even through an offset — makes both vary per model,
#: and the resulting sampling noise lands inside the comparison the split exists
#: to make fair.
DEFENDER_SPLIT_SEED = 10_000


def defender_split(
    labels: np.ndarray,
    n_per_class: int,
    split_seed: int = DEFENDER_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Test-set split into ``(defender, eval)``, identical for every model.

    Args:
        split_seed: Only vary this to measure how much a result depends on
            which clean images the defender happened to get. It is not a
            per-model knob; see :data:`DEFENDER_SPLIT_SEED`.
    """
    return stratified_indices(labels, n_per_class, seed=split_seed)
