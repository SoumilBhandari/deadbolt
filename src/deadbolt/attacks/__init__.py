"""Backdoor attacks: trigger implementations."""

from deadbolt.attacks.adaptive_blend import AdaptiveBlend
from deadbolt.attacks.badnets import BadNets
from deadbolt.attacks.base import LabelMode, Trigger
from deadbolt.attacks.blended import Blended
from deadbolt.attacks.label_consistent import LabelConsistent
from deadbolt.attacks.sig import SIG
from deadbolt.attacks.wanet import WaNet

#: Registry consumed by config loading, so configs name attacks as strings.
ATTACKS: dict[str, type[Trigger]] = {
    t.name: t for t in (BadNets, Blended, SIG, WaNet, LabelConsistent, AdaptiveBlend)
}

#: Attacks that are only meaningful in clean-label mode. The zoo builder uses
#: this to reject a config pairing them with dirty-label poisoning, which would
#: silently produce a different — and much stronger — attack than the name on
#: the results row claims.
CLEAN_LABEL_ATTACKS = frozenset({SIG.name, LabelConsistent.name})

__all__ = [
    "ATTACKS",
    "CLEAN_LABEL_ATTACKS",
    "SIG",
    "AdaptiveBlend",
    "BadNets",
    "Blended",
    "LabelConsistent",
    "LabelMode",
    "Trigger",
    "WaNet",
]
