"""Backdoor detectors, across all threat models."""

from deadbolt.defenses.activation_clustering import ActivationClustering
from deadbolt.defenses.base import Access, DetectionResult, Detector, anomaly_index
from deadbolt.defenses.karm import KArm
from deadbolt.defenses.neural_cleanse import NeuralCleanse
from deadbolt.defenses.spectral import SpectralSignatures
from deadbolt.defenses.spectre import SPECTRE
from deadbolt.defenses.strip import STRIP

#: Registry consumed by the CLI, so `--defense neural_cleanse,strip` works.
DEFENSES: dict[str, type[Detector]] = {
    d.name: d
    for d in (NeuralCleanse, KArm, STRIP, SpectralSignatures, SPECTRE, ActivationClustering)
}

__all__ = [
    "DEFENSES",
    "SPECTRE",
    "STRIP",
    "Access",
    "ActivationClustering",
    "DetectionResult",
    "Detector",
    "KArm",
    "NeuralCleanse",
    "SpectralSignatures",
    "anomaly_index",
]
