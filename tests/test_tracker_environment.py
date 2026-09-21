"""The tracker needs a CUDA build of torch, and only prose said so.

``cotracker_tracking`` pins every tensor to ``"cuda"`` -- the model, the frame
batch, the query points -- with no CPU path anywhere. So a CPU-only torch does
not make tracking slow, it makes it raise the first time you track anything,
long after the app has opened and looked fine.

``pyproject.toml`` cannot express that. It can only ask for ``torch>=2.0``, and
on Windows PyPI answers with the CPU wheel; the CUDA build comes from PyTorch's
own index and has to be installed on purpose. That makes "rebuild the venv" a
way to silently lose the tracker.

The check used to be two tests gated on an NVIDIA driver, which meant they could
only ever run on a developer's GPU machine: the merge gate's own green run was
"162 passed, 2 skipped" and these were the two skips, so the gate enforced
nothing while the suite stayed permanently short of zero skips (audit
scripture/all/tests/011). The check now lives in
:mod:`scripture.tracker_environment`, which the app runs at startup on the
machine that actually tracks -- and what is tested here is the checker itself,
against a stand-in torch, so it runs everywhere and skips nowhere.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripture import tracker_environment
from scripture.tracker_environment import complaints

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER = REPO_ROOT / "scripture" / "cotracker_tracking.py"


class FakeTorch:
    """A torch build: which CUDA it was compiled against, and what it can see."""

    def __init__(self, cuda_version="12.8", sees_a_device=True):
        self.__version__ = "2.6.0"
        self.version = type("version", (), {"cuda": cuda_version})
        self.cuda = type("cuda", (), {"is_available": staticmethod(lambda: sees_a_device)})


@pytest.fixture
def machine(monkeypatch):
    """A machine with a driver and a working CUDA torch, until a test says else."""
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
        """Why the checker exists. If a CPU path is ever added here, the CUDA
        build stops being a hard requirement and the checker should go -- so the
        premise is checked rather than assumed."""
        devices = {node.value
                   for node in ast.walk(ast.parse(TRACKER.read_text(encoding="utf-8")))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)}

        assert "cuda" in devices
        assert "cpu" not in devices, (
            "cotracker_tracking now names a CPU device; if it can fall back, the "
            "tracker-environment check's premise no longer holds"
        )
