from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripture import tracker_environment
from scripture.tracker_environment import complaints

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER = REPO_ROOT / "scripture" / "cotracker_tracking.py"


class FakeTorch:

    def __init__(self, cuda_version="12.8", sees_a_device=True):
        self.__version__ = "2.6.0"
        self.version = type("version", (), {"cuda": cuda_version})
        self.cuda = type("cuda", (), {"is_available": staticmethod(lambda: sees_a_device)})


@pytest.fixture
def machine(monkeypatch):
    def described(*, driver=True, torch=None):
        monkeypatch.setattr(tracker_environment, "_has_an_nvidia_driver", lambda: driver)
        monkeypatch.setattr(tracker_environment, "_torch",
                            lambda: FakeTorch() if torch is None else torch)
        return complaints()
    return described


class TestWhatTheCheckerSays:

    def test_a_working_cuda_build_draws_no_complaint(self, machine):
        assert machine() == []

    def test_the_cpu_wheel_is_named_along_with_the_way_out_of_it(self, machine):
        (said,) = machine(torch=FakeTorch(cuda_version=None))

        assert "CPU build" in said
        assert "download.pytorch.org/whl/cu" in said

    def test_a_cuda_build_the_driver_is_too_old_for_is_told_apart(self, machine):
        (said,) = machine(torch=FakeTorch(sees_a_device=False))

        assert "sees no device" in said
        assert "CPU build" not in said

    def test_a_machine_with_no_driver_is_not_a_machine_that_tracks(self, machine):
        assert machine(driver=False) == []

    def test_no_torch_at_all_is_said_rather_than_raised(self, machine):
        assert "torch is not installed" in machine(torch=False)[0]


class TestThePremiseTheCheckerRestsOn:

    def test_the_tracker_still_pins_every_tensor_to_cuda(self):
        devices = {node.value
                   for node in ast.walk(ast.parse(TRACKER.read_text(encoding="utf-8")))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)}

        assert "cuda" in devices
        assert "cpu" not in devices, (
            "cotracker_tracking now names a CPU device; if it can fall back, the "
            "tracker-environment check's premise no longer holds"
        )
