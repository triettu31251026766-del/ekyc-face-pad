"""src/reproducibility.py — đảm bảo khả năng tái lập thí nghiệm.

Tệp này dùng để (theo mục 36 của tài liệu kỹ thuật):
- set_seed(seed): cố định seed cho Python random, NumPy và PyTorch (cả CUDA)
  để mọi thí nghiệm có thể tái lập được.
- get_environment_info(): ghi lại thông tin môi trường để lưu kèm kết quả:
      python_version, torch_version, torchvision_version, numpy_version,
      platform, device (tên + có CUDA hay không), git_commit.
- get_git_commit(path): lấy mã commit Git hiện tại (dùng để biết phiên bản
  code đã chạy thí nghiệm).

Theo mục 36 tài liệu, kết quả thí nghiệm cần lưu: phiên bản Python/PyTorch/
torchvision, thông tin dataset, tệp config, seed, Git commit, device. Phần
dataset/config/seed do experiments/ lưu, phần còn lại dùng module này.

Cách dùng:
    from src.reproducibility import set_seed, get_environment_info
    set_seed(42)
    env_info = get_environment_info()
"""

from __future__ import annotations

import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Cố định seed cho toàn bộ nguồn ngẫu nhiên (mục 36 tài liệu).

    Args:
        seed: Seed nguyên không âm.

    Raises:
        ValueError: nếu seed không phải số nguyên không âm.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Để kết quả CUDA tái lập được: tắt benchmark, bật chế độ deterministic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_environment_info() -> dict:
    """Thu thập thông tin môi trường chạy thí nghiệm (mục 36 tài liệu).

    Returns:
        dict gồm: python_version, torch_version, torchvision_version,
        numpy_version, platform, device, cuda_available, git_commit.
    """
    info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": _torchvision_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "device": _resolve_device_name(),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": get_git_commit(),
    }
    return info


def get_git_commit(path: str | Path = ".") -> str | None:
    """Lấy mã commit Git hiện tại (40 ký tự hex); trả None nếu không phải repo Git.

    Dùng subprocess thay vì thư viện gitpython để tránh thêm dependency.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        # Không tìm thấy lệnh git trong PATH.
        return None

    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit if commit else None


def _torchvision_version() -> str:
    """Lấy phiên bản torchvision, trả "n/a" nếu chưa cài."""
    try:
        import torchvision

        return torchvision.__version__
    except ImportError:
        return "n/a"


def _resolve_device_name() -> str:
    """Tên thiết bị thực tế sẽ được dùng (CPU hoặc tên GPU đầu tiên)."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"cpu ({platform.processor() or 'unknown'})"
