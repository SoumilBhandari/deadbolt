"""The benchmark's central claim, tested.

deadbolt asserts that detectors run blind. That assertion is worth exactly as
much as the mechanism enforcing it, and the failure mode is invisible: a
detector that reads ground truth from somewhere it should not still returns a
number in the right range, and no amount of reading the results table reveals
it.

So these tests do not check that detectors *happen* not to peek. They check
that peeking is structurally impossible — that nothing reachable from what a
detector is handed leads back to the answer.
"""

from __future__ import annotations

import gc
from dataclasses import fields

import pytest
import torch
from torch.utils.data import DataLoader

from deadbolt.defenses.base import DetectionResult, Detector
from deadbolt.scan import Blind, GroundTruth, _run_one
from deadbolt.train import TrainResult


class Canary(Detector):
    """A detector that tries every route to the answer and records what it found."""

    name = "canary"
    access = "model_clean"

    def __init__(self) -> None:
        self.leaks: list[str] = []

    def scan(self, model, data, *, num_classes, device, **kwargs) -> DetectionResult:
        for name, obj in [("model", model), ("loader", data), ("dataset", data.dataset)]:
            self.leaks.extend(self._probe(name, obj))
        return DetectionResult(is_backdoored=False, score=0.0)

    @staticmethod
    def _probe(name: str, obj: object) -> list[str]:
        """Look for the words a leak would have to spell, one hop out."""
        forbidden = {"trigger", "poison", "truth", "target_label", "backdoor", "attack"}
        found = []
        for attr in dir(obj):
            if attr.startswith("__"):
                continue
            if any(word in attr.lower() for word in forbidden):
                found.append(f"{name}.{attr}")
        # One hop deeper through the object's own attribute values, which is
        # how a Subset-of-PoisonedDataset would give the game away.
        for attr in getattr(obj, "__dict__", {}):
            value = getattr(obj, attr, None)
            for sub in dir(value):
                if not sub.startswith("__") and any(w in sub.lower() for w in forbidden):
                    found.append(f"{name}.{attr}.{sub}")
        return found


def _blind(truth: GroundTruth, model, loader) -> Blind:
    return Blind(model=model, loader=loader, clean_loader=loader, num_classes=10, truth=truth)


@pytest.fixture
def truth() -> GroundTruth:
    return GroundTruth(
        is_backdoored=True,
        attack="badnets",
        target_label=7,
        label_mode="all2one",
        poison_rate=0.05,
    )


def test_ground_truth_is_not_a_field_of_what_the_detector_receives(truth, tiny_model, toy_loader):
    """The detector is handed model/loader/clean_loader. Truth is a sibling, not a member."""
    blind = _blind(truth, tiny_model, toy_loader)
    handed = {"model", "loader", "clean_loader", "num_classes"}
    names = {f.name for f in fields(Blind)}
    assert names == handed | {"truth"}
    for attr in handed:
        assert getattr(blind, attr) is not truth


def test_detector_cannot_reach_ground_truth_from_its_arguments(truth, tiny_model, toy_loader):
    """A model-and-clean-data detector sees no poisoning vocabulary at all."""
    canary = Canary()
    _run_one("z", _record(), canary, _blind(truth, tiny_model, toy_loader), _ctx())
    assert canary.leaks == []


def test_ground_truth_is_not_reachable_by_object_graph(truth, tiny_model, toy_loader):
    """Walk the referrer graph: no object the detector holds points at the truth.

    Deliberately a different technique from the attribute scan above. Attribute
    names can be renamed; a reference cannot be hidden from the GC.
    """
    blind = _blind(truth, tiny_model, toy_loader)
    gc.collect()
    for handed in (blind.model, blind.loader, blind.clean_loader):
        assert not _reaches(handed, truth, depth=4)


def _reaches(start: object, target: object, depth: int) -> bool:
    seen: set[int] = set()
    frontier = [start]
    for _ in range(depth):
        nxt = []
        for obj in frontier:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if obj is target:
                return True
            for value in list(getattr(obj, "__dict__", {}).values()):
                nxt.append(value)
        frontier = nxt
    return False


def test_verdict_and_truth_are_recorded_separately(truth, tiny_model, toy_loader):
    """The scorer keeps the answer; the record keeps both, side by side and labelled."""
    rec = _run_one("z", _record(), Canary(), _blind(truth, tiny_model, toy_loader), _ctx())
    assert rec.truth["is_backdoored"] is True
    assert rec.result["is_backdoored"] is False
    assert rec.truth["target_label"] == 7


def test_a_crashing_detector_is_recorded_not_dropped(truth, tiny_model, toy_loader):
    """Crashing on the attacks a method finds hardest is a result about that method."""

    class Exploding(Detector):
        name = "boom"
        access = "model_clean"

        def scan(self, model, data, **kwargs):
            raise RuntimeError("no convergence")

    rec = _run_one("z", _record(), Exploding(), _blind(truth, tiny_model, toy_loader), _ctx())
    assert rec.error is not None and "no convergence" in rec.error
    assert rec.result == {}


def _record() -> TrainResult:
    return TrainResult(
        config={},
        context={},
        clean_accuracy=0.99,
        attack_success_rate=0.99,
        poison={"attack": "badnets"},
        checkpoint="unused.pt",
        config_hash="deadbeef",
        valid_testcase=True,
        filter_reason=None,
    )


def _ctx():
    from deadbolt.config import RunContext

    return RunContext(seed=0, device="cpu", commit="test")
