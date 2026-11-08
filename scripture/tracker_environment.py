from __future__ import annotations

import shutil
import subprocess
from types import ModuleType

_FIX_ON_WINDOWS_PYPI_ANSWERS_TORCH_WITH_THE_CPU_WHEEL = (
    "install the CUDA build into this interpreter:\n"
    "  python -m pip install --upgrade torch torchvision "
    "--index-url https://download.pytorch.org/whl/cuXXX\n"
    "choosing the cuXXX index your driver supports (nvidia-smi prints the "
    "highest CUDA version it can run). See CLAUDE.md."
)


def complaints() -> list[str]:
    if not _has_an_nvidia_driver_asked_of_nvidia_smi_not_of_the_torch_being_checked():
        return []
    torch = _torch()
    if not torch:
        return [f"torch is not installed, so the tracker cannot run; {_FIX_ON_WINDOWS_PYPI_ANSWERS_TORCH_WITH_THE_CPU_WHEEL}"]
    if torch.version.cuda is None:
        return [(f"torch {torch.__version__} is the CPU build, so tracking will "
                 f"raise on first use; {_FIX_ON_WINDOWS_PYPI_ANSWERS_TORCH_WITH_THE_CPU_WHEEL}")]
    if not torch.cuda.is_available():
        return [(f"torch {torch.__version__} is a CUDA build but sees no device -- "
                 f"most likely built against a newer CUDA than this driver "
                 f"supports; {_FIX_ON_WINDOWS_PYPI_ANSWERS_TORCH_WITH_THE_CPU_WHEEL}")]
    return []


def _has_an_nvidia_driver_asked_of_nvidia_smi_not_of_the_torch_being_checked(
        *, which=shutil.which, run=subprocess.run) -> bool:
    if which("nvidia-smi") is None:
        return False
    return run(["nvidia-smi"], capture_output=True, check=False).returncode == 0


def _torch() -> ModuleType | None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch
