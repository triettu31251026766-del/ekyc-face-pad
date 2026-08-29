"""tests/test_utils.py — kiểm thử (unit test) cho module src/utils.py.

Tệp này dùng để (theo mục 5, 29, 30, 37, 42 của tài liệu kỹ thuật):
- Kiểm tra resolve_device với "auto"/"cpu"/tên sai và "cuda" không khả dụng.
- Kiểm tra count_parameters / model_size_mb cho giá trị dương hợp lý.
- Kiểm tra save_json / save_csv ghi và đọc lại đúng dữ liệu, tự tạo thư mục.
- Kiểm tra get_experiment_logger ghi được nội dung log vào tệp đúng id.
- Kiểm tra measure_latency_ms tuân thủ protocol mục 42 (warmup + runs).

Chạy kiểm thử:
    python -m pytest tests/test_utils.py
"""

import json

import pandas as pd
import pytest
import torch

from src.model import build_model
from src.utils import (
    count_parameters,
    get_experiment_logger,
    measure_latency_ms,
    model_size_mb,
    resolve_device,
    save_csv,
    save_json,
)


def test_resolve_device_cpu():
    assert str(resolve_device("cpu")) == "cpu"


def test_resolve_device_auto():
    device = resolve_device("auto")
    if torch.cuda.is_available():
        assert str(device) == "cuda"
    else:
        assert str(device) == "cpu"


def test_resolve_device_unknown_raises():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="chỉ test khi không có CUDA")
def test_resolve_device_cuda_unavailable_raises():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_count_parameters_positive():
    model = build_model("custom_cnn", num_classes=1)
    assert count_parameters(model) > 0
    assert isinstance(count_parameters(model), int)


def test_model_size_mb_positive():
    model = build_model("custom_cnn", num_classes=1)
    size = model_size_mb(model)
    assert size > 0.0
    assert isinstance(size, float)


def test_save_json_roundtrip(tmp_path):
    out = tmp_path / "nested" / "result.json"
    data = {"experiment_id": "E01_baseline_seed123", "metrics": {"f1": 0.9}}
    save_json(data, out)
    assert out.is_file()
    with out.open("r", encoding="utf-8") as handle:
        assert json.load(handle) == data


def test_save_csv_roundtrip(tmp_path):
    out = tmp_path / "sub" / "result.csv"
    rows = [
        {"path": "a.jpg", "label": 0, "probability_spoof": 0.1},
        {"path": "b.jpg", "label": 1, "probability_spoof": 0.9},
    ]
    save_csv(rows, out)
    frame = pd.read_csv(out)
    assert list(frame.columns) == ["path", "label", "probability_spoof"]
    assert len(frame) == 2


def test_experiment_logger_writes_file(tmp_path):
    logger = get_experiment_logger("E99_test", log_dir=tmp_path)
    logger.info("seed=123")
    logger.info("final f1=0.95")

    log_file = tmp_path / "E99_test.log"
    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "seed=123" in content
    assert "final f1=0.95" in content


def test_experiment_logger_no_duplicate_handlers(tmp_path):
    first = get_experiment_logger("E99_dup", log_dir=tmp_path)
    second = get_experiment_logger("E99_dup", log_dir=tmp_path)
    assert first is second
    assert len(first.handlers) == 2  # 1 file + 1 console, không nhân bản


def test_measure_latency_protocol(tmp_path=None):
    model = build_model("custom_cnn", num_classes=1)
    x = torch.randn(1, 3, 32, 32)
    result = measure_latency_ms(model, x, device="cpu", runs=5, warmup=2)
    assert set(result.keys()) == {"mean_latency_ms", "runs", "warmup",
                                  "device", "batch_size", "image_size"}
    assert result["runs"] == 5
    assert result["warmup"] == 2
    assert result["batch_size"] == 1
    assert result["image_size"] == 32
    assert result["mean_latency_ms"] >= 0.0


def test_measure_latency_invalid_args_raise():
    model = build_model("custom_cnn", num_classes=1)
    x = torch.randn(1, 3, 32, 32)
    with pytest.raises(ValueError, match="runs"):
        measure_latency_ms(model, x, runs=0)
    with pytest.raises(ValueError, match="warmup"):
        measure_latency_ms(model, x, warmup=-1)
