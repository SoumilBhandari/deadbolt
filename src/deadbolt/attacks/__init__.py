"""Backdoor attacks: trigger implementations."""

from deadbolt.attacks.badnets import BadNets
from deadbolt.attacks.base import LabelMode, Trigger

#: Registry consumed by config loading, so configs name attacks as strings.
ATTACKS: dict[str, type[Trigger]] = {
    BadNets.name: BadNets,
}

__all__ = ["Trigger", "LabelMode", "BadNets", "ATTACKS"]
