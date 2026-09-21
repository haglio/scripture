"""Whether this machine's torch can actually run the tracker.

`cotracker_tracking` pins every tensor to "cuda" with no CPU path, so a CPU-only
torch does not slow tracking down -- it raises the first time you track, long
after the app has opened and looked fine.  `pyproject.toml` can only ask for
`torch>=2.0`, and on Windows PyPI answers with the CPU wheel, so rebuilding the
venv is a way to lose the tracker silently.

This ran as two pytest tests gated on an NVIDIA driver, which meant the merge
gate skipped both and enforced nothing.  It belongs where failing is useful: at
startup, on the machine that tracks.
"""
from __future__ import annotations

import shutil
import subprocess
from types import ModuleType

_FIX = (
    "install the CUDA build into this interpreter:\n"
    "  python -m pip install --upgrade torch torchvision "
    "--index-url https://download.pytorch.org/whl/cuXXX\n"
    "choosing the cuXXX index your driver supports (nvidia-smi prints the "
    "highest CUDA version it can run). See CLAUDE.md."
)


def complaints() -> list[str]:
    """What is wrong with this machine's torch for the tracker, if anything.

    Nothing to say where there is no NVIDIA driver: tracking is impossible there
    whatever torch is installed, which is also why the merge gate -- which
    deliberately installs the CPU wheel and loads no model -- hears nothing.
    """
    if not _has_an_nvidia_driver():
        return []
    torch = _torch()
    if not torch:
        return [f"torch is not installed, so the tracker cannot run; {_FIX}"]
    if torch.version.cuda is None:
        return [(f"torch {torch.__version__} is the CPU build, so tracking will "
                 f"raise on first use; {_FIX}")]
    if not torch.cuda.is_available():
        return [(f"torch {torch.__version__} is a CUDA build but sees no device -- "
                 f"most likely built against a newer CUDA than this driver "
                 f"supports; {_FIX}")]
    return []


def _has_an_nvidia_driver() -> bool:
    """Whether a CUDA launch is even possible here.

    Asking the driver rather than asking torch: torch is the thing being
    checked, so gating on `torch.cuda.is_available()` would pass by being broken.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi"], capture_output=True, check=False).returncode == 0


def _torch() -> ModuleType | None:
    try:
        # Local: the whole point is that this import may not be satisfiable.
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch
