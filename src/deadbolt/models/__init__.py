"""Victim architectures. All expose a penultimate-feature contract."""

from deadbolt.models.smallcnn import BackdoorModel, SmallCNN

ARCHS: dict[str, type[BackdoorModel]] = {
    "smallcnn": SmallCNN,
}

__all__ = ["BackdoorModel", "SmallCNN", "ARCHS"]
