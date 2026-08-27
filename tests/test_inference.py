"""tests/test_inference.py — kiểm thử (unit test) cho module src/inference.py.

Tệp này dùng để (theo mục 19 của tài liệu kỹ thuật):
- Kiểm tra predict_image / predict_pil trả về đúng cấu trúc:
  {"probability_spoof": float trong [0, 1], "prediction": "spoof"|"bona_fide"}.
- Dùng model giả có logit cố định để kiểm tra đúng quy tắc ngưỡng:
  probability >= threshold -> "spoof".
- Kiểm tra báo lỗi rõ ràng khi tệp ảnh không tồn tại.

Chạy kiểm thử:
    python -m pytest tests/test_inference.py
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.inference import predict_image, predict_pil


class FixedLogitModel(nn.Module):
    """Model giả: luôn trả về logit cố định, không cần huấn luyện."""

    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.logit, dtype=torch.float32)


def _transform(image):
    """Transform đơn giản: resize 32x32 + ToTensor (không normalize cũng được)."""
    from torchvision.transforms import ToTensor

    return ToTensor()(image.resize((32, 32)))


def _structured_image(tmp_path, name="test.png"):
    rng = np.random.default_rng(3)
    image = Image.fromarray(rng.integers(0, 256, size=(64, 64, 3)).astype(np.uint8))
    path = tmp_path / name
    image.save(path)
    return image, str(path)


def test_predict_image_structure(tmp_path):
    _, path = _structured_image(tmp_path)
    model = FixedLogitModel(logit=1.0)
    result = predict_image(model, path, _transform, device="cpu")
    assert set(result.keys()) == {"probability_spoof", "prediction"}
    assert isinstance(result["probability_spoof"], float)
    assert 0.0 <= result["probability_spoof"] <= 1.0
    assert result["prediction"] in ("spoof", "bona_fide")


def test_positive_logit_gives_spoof(tmp_path):
    _, path = _structured_image(tmp_path)
    model = FixedLogitModel(logit=2.0)
    result = predict_image(model, path, _transform, device="cpu")
    # sigmoid(2.0) ~= 0.88 >= 0.5 -> spoof
    assert result["prediction"] == "spoof"
    assert result["probability_spoof"] == pytest.approx(0.8808, abs=1e-3)


def test_negative_logit_gives_bona_fide(tmp_path):
    _, path = _structured_image(tmp_path)
    model = FixedLogitModel(logit=-2.0)
    result = predict_image(model, path, _transform, device="cpu")
    assert result["prediction"] == "bona_fide"
    assert result["probability_spoof"] == pytest.approx(0.1192, abs=1e-3)


def test_threshold_boundary_uses_greater_or_equal(tmp_path):
    image, _ = _structured_image(tmp_path)
    model = FixedLogitModel(logit=0.0)  # sigmoid(0) = 0.5
    # Ngưỡng 0.5: 0.5 >= 0.5 -> "spoof"; ngưỡng 0.5001 -> "bona_fide".
    result = predict_pil(model, image, _transform, device="cpu", threshold=0.5)
    assert result["prediction"] == "spoof"
    result = predict_pil(model, image, _transform, device="cpu", threshold=0.5001)
    assert result["prediction"] == "bona_fide"


def test_predict_pil_matches_predict_image(tmp_path):
    image, path = _structured_image(tmp_path)
    model = FixedLogitModel(logit=1.0)
    from_file = predict_image(model, path, _transform, device="cpu")
    from_pil = predict_pil(model, image, _transform, device="cpu")
    assert from_file == from_pil


def test_missing_image_raises(tmp_path):
    model = FixedLogitModel(logit=0.0)
    with pytest.raises(FileNotFoundError, match="not found"):
        predict_image(model, str(tmp_path / "khong_ton_tai.png"), _transform)
