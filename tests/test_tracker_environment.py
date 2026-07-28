"""The tracker needs a CUDA build of torch, and until now only prose said so.

``cotracker_tracking`` pins every tensor to ``"cuda"`` -- the model, the frame
batch, the query points -- with no CPU path anywhere. So a CPU-only torch does
not make tracking slow, it makes it raise the first time you track anything,
long after the app has opened and looked fine.

``pyproject.toml`` cannot express that. It can only ask for ``torch>=2.0``, and
on Windows PyPI answers with the CPU wheel; the CUDA build comes from PyTorch's
own index and has to be installed on purpose. That makes "rebuild the venv" a
way to silently lose the tracker, with ``CLAUDE.md`` asking you to remember as
the only thing in the way -- which is exactly the kind of rule worth checking
instead of writing down.

This is that check. It runs wherever a CUDA launch is possible at all, which is
the presence of an NVIDIA driver, and skips everywhere else: CI has no GPU,
installs the CPU wheel deliberately, and never loads a model -- every test
there drives the numpy/OpenCV logic with synthetic data.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKER = REPO_ROOT / "scripture" / "cotracker_tracking.py"

_FIX = (
    "install the CUDA build into this interpreter:\n"
    "  python -m pip install --upgrade torch torchvision "
    "--index-url https://download.pytorch.org/whl/cuXXX\n"
    "choosing the cuXXX index your driver supports (nvidia-smi prints the "
    "highest CUDA version it can run). See CLAUDE.md."
)


def _has_an_nvidia_driver() -> bool:
    """Whether a CUDA launch is even possible here.

    Asking the driver rather than asking torch: torch is the thing under test,
    so gating on ``torch.cuda.is_available()`` would make this pass by being
    broken.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0


needs_a_gpu = pytest.mark.skipif(
    not _has_an_nvidia_driver(),
    reason="no NVIDIA driver -- CI runs the CPU wheel on purpose and loads no model",
)


def test_the_tracker_still_pins_every_tensor_to_cuda():
    """Why the two tests below exist. If a CPU path is ever added here, the
    CUDA build stops being a hard requirement and this whole file should go --
    so the premise is checked rather than assumed."""
    source = TRACKER.read_text(encoding="utf-8")

    assert 'device="cuda"' in source or '.to("cuda")' in source
    assert '"cpu"' not in source, (
        "cotracker_tracking now names a CPU device; if it can fall back, this "
        "file's premise no longer holds and it should be revisited"
    )


@needs_a_gpu
def test_torch_is_a_cuda_build():
    """The CPU wheel is what a plain ``pip install torch`` leaves behind, and
    it is indistinguishable from the right one until the tracker runs."""
    import torch

    assert torch.version.cuda is not None, (
        f"torch {torch.__version__} is the CPU build, so tracking will raise "
        f"on first use; {_FIX}"
    )


@needs_a_gpu
def test_torch_can_actually_reach_the_gpu():
    """A CUDA build alone is not enough -- one compiled against a newer CUDA
    than the installed driver supports imports fine and then finds no device,
    which fails in the same place, at the same time, for a different reason."""
    import torch

    assert torch.cuda.is_available(), (
        f"torch {torch.__version__} is a CUDA build but sees no device -- most "
        f"likely built against a newer CUDA than this driver supports; {_FIX}"
    )
