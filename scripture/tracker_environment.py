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
    if shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi"], capture_output=True, check=False).returncode == 0


def _torch() -> ModuleType | None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch
