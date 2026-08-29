"""src/utils.py — các hàm tiện ích dùng chung cho thí nghiệm.

Tệp này dùng để (theo mục 5 của tài liệu kỹ thuật):
- resolve_device(name): chọn thiết bị chạy từ cấu hình ("auto"/"cpu"/"cuda").
- count_parameters(model) / model_size_mb(model): thống kê model
  (dùng cho các trường tùy chọn parameter_count, model_size_mb ở mục 29).
- save_json(data, path) / save_csv(rows, path): ghi kết quả thí nghiệm
  (mục 30 tài liệu yêu cầu lưu cả CSV và JSON).
- get_experiment_logger(experiment_id, log_dir): ghi log ra tệp theo
  experiment_id (mục 37: không chỉ dựa vào output terminal).
- measure_latency_ms(model, input_tensor, ...): đo latency theo đúng
  protocol mục 42 (warm-up rồi mới đo).

Chú ý: module này KHÔNG tính metric, KHÔNG huấn luyện, KHÔNG suy giảm chất
lượng — chỉ chứa các hàm tiện ích thuần túy, có thể dùng ở mọi nơi.

Cách dùng:
    from src.utils import resolve_device, save_json, get_experiment_logger
    device = resolve_device(config["device"]["name"])
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

VALID_DEVICE_NAMES = ("auto", "cpu", "cuda")


def resolve_device(name: str) -> torch.device:
    """Chọn torch.device từ tên trong cấu hình (mục 25 tài liệu).

    Args:
        name: "auto" (CUDA nếu có, ngược lại CPU), "cpu" hoặc "cuda".

    Returns:
        torch.device tương ứng.

    Raises:
        ValueError: nếu tên không hợp lệ.
        RuntimeError: nếu yêu cầu "cuda" nhưng CUDA không khả dụng.
    """
    if name not in VALID_DEVICE_NAMES:
        raise ValueError(
            f"Unknown device name: {name!r}. Expected one of {list(VALID_DEVICE_NAMES)}"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests 'cuda' but CUDA is not available")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số tham số của model (dùng cho trường parameter_count, mục 29)."""
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Ước lượng dung lượng model (MB) từ state_dict (trường model_size_mb, mục 29)."""
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024.0 * 1024.0)


def save_json(data: dict, path: str | Path) -> None:
    """Ghi dict ra tệp JSON (mục 30 tài liệu), tự tạo thư mục cha nếu chưa có."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def save_csv(rows: list[dict], path: str | Path) -> None:
    """Ghi danh sách dict ra tệp CSV (mục 30 tài liệu) bằng pandas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def get_experiment_logger(
    experiment_id: str,
    log_dir: str | Path = "results/raw",
) -> logging.Logger:
    """Tạo logger ghi ra tệp results/raw/<experiment_id>.log + ra console (mục 37).

    Mỗi experiment_id có logger riêng với tệp log riêng. Nếu logger đã tồn
    tại (cùng id) thì trả về logger cũ, không gắn thêm handler trùng lặp.

    Args:
        experiment_id: Mã thí nghiệm (ví dụ "E01_baseline_seed123").
        log_dir: Thư mục chứa tệp log.

    Returns:
        logging.Logger đã cấu hình (level INFO).
    """
    logger = logging.getLogger(f"experiment.{experiment_id}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{experiment_id}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def measure_latency_ms(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str | torch.device = "cpu",
    runs: int = 100,
    warmup: int = 20,
) -> dict:
    """Đo latency suy luận trung bình theo đúng protocol mục 42 tài liệu.

    Protocol: chạy warmup lần trước (không tính), sau đó đo runs lần bằng
    time.perf_counter, trả về thời gian trung bình mỗi lần chạy.

    Args:
        model: Model cần đo (chuyển sang mode eval()).
        input_tensor: Tensor đầu vào đã ở đúng thiết bị và đúng shape
            (ví dụ shape (1, 3, 224, 224)).
        device: Thiết bị đo.
        runs: Số lần chạy được đo.
        warmup: Số lần chạy khởi động (không tính vào kết quả).

    Returns:
        dict gồm: mean_latency_ms, runs, warmup, device, batch_size, image_size
        (đủ các trường phải báo cáo theo mục 42).

    Raises:
        ValueError: nếu runs < 1 hoặc warmup < 0.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    device = torch.device(device)
    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        # Warm-up: khởi động kernel / bộ nhớ, không đo.
        for _ in range(warmup):
            model(input_tensor)

        # Đo thời gian thực tế.
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total_seconds = end - start
    mean_ms = (total_seconds / runs) * 1000.0

    return {
        "mean_latency_ms": mean_ms,
        "runs": runs,
        "warmup": warmup,
        "device": str(device),
        "batch_size": int(input_tensor.shape[0]),
        "image_size": _image_size_from_tensor(input_tensor),
    }


# --- Các hàm nội bộ ---


def _image_size_from_tensor(tensor: torch.Tensor) -> int | None:
    """Lấy kích thước ảnh (H) từ tensor dạng (B, C, H, W); None nếu khác dạng."""
    if tensor.ndim == 4:
        return int(tensor.shape[2])
    return None
