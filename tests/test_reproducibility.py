"""tests/test_reproducibility.py — kiểm thử (unit test) cho module src/reproducibility.py.

Tệp này dùng để (theo mục 36 của tài liệu kỹ thuật):
- Kiểm tra set_seed: cùng seed -> cùng dãy số ngẫu nhiên (random, numpy, torch);
  seed khác -> dãy khác.
- Kiểm tra seed không hợp lệ báo lỗi rõ ràng.
- Kiểm tra get_environment_info trả về đủ các khóa thông tin môi trường.
- Kiểm tra get_git_commit trả về mã commit hợp lệ khi chạy trong repo Git.

Chạy kiểm thử:
    python -m pytest tests/test_reproducibility.py
"""

import random

import numpy as np
import pytest
import torch

from src.reproducibility import get_environment_info, get_git_commit, set_seed


def test_set_seed_reproduces_python_random():
    set_seed(123)
    first = [random.random() for _ in range(5)]
    set_seed(123)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_seed_reproduces_numpy():
    set_seed(7)
    first = np.random.rand(4).tolist()
    set_seed(7)
    second = np.random.rand(4).tolist()
    assert first == second


def test_set_seed_reproduces_torch():
    set_seed(123)
    first = torch.rand(6).tolist()
    set_seed(123)
    second = torch.rand(6).tolist()
    assert first == second


def test_different_seeds_give_different_sequences():
    set_seed(1)
    first = [random.random() for _ in range(8)]
    set_seed(2)
    second = [random.random() for _ in range(8)]
    assert first != second


@pytest.mark.parametrize("seed", [-1, 1.5, "123", True])
def test_invalid_seed_raises(seed):
    with pytest.raises(ValueError, match="seed"):
        set_seed(seed)


def test_environment_info_has_required_keys():
    info = get_environment_info()
    required = [
        "python_version", "torch_version", "torchvision_version",
        "numpy_version", "platform", "device", "cuda_available", "git_commit",
    ]
    for key in required:
        assert key in info
    assert info["python_version"].count(".") >= 1
    assert info["torch_version"].strip()
    assert info["cuda_available"] in (True, False)


def test_get_git_commit_returns_commit_in_repo():
    """Chạy test từ thư mục gốc repo -> phải lấy được commit hiện tại."""
    commit = get_git_commit(".")
    assert commit is not None
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_get_git_commit_returns_none_outside_repo(tmp_path):
    """Thư mục không phải repo Git -> trả về None."""
    assert get_git_commit(tmp_path) is None
