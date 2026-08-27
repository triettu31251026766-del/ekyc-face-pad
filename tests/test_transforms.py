"""tests/test_transforms.py — kiểm thử (unit test) cho module src/transforms.py.

Tệp này dùng để (theo mục 9, 13 của tài liệu kỹ thuật):
- Kiểm tra transform train/eval cho ra Tensor đúng kích thước (C, size, size).
- Kiểm tra giá trị normalize đúng theo công thức ImageNet.
- Kiểm tra transform EVAL là tất định (cùng ảnh -> cùng Tensor).
- Kiểm tra báo lỗi rõ ràng khi config thiếu "model.image_size".

Chạy kiểm thử:
    python -m pytest tests/test_transforms.py
"""

import pytest
import torch
from PIL import Image

from src.config import ConfigError
from src.transforms import build_eval_transform, build_train_transform


def _config(size=32):
    return {"model": {"image_size": size}}


def _make_image():
    # Ảnh RGB màu xám (128/255 = 0.502) để kiểm tra normalize bằng công thức.
    return Image.new("RGB", (64, 48), color=(128, 128, 128))


def test_train_transform_output_shape():
    transform = build_train_transform(_config(size=32))
    tensor = transform(_make_image())
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32


def test_eval_transform_output_shape():
    transform = build_eval_transform(_config(size=16))
    tensor = transform(_make_image())
    assert tensor.shape == (3, 16, 16)


def test_eval_transform_is_deterministic():
    """Transform eval phải tất định: 2 lần áp dụng cho cùng ảnh phải giống hệt nhau."""
    transform = build_eval_transform(_config(size=16))
    image = _make_image()
    assert torch.equal(transform(image), transform(image))


def test_normalize_uses_imagenet_stats():
    """Kênh 0 của ảnh xám 0.502: (0.502 - 0.485) / 0.229 ~= 0.0742."""
    transform = build_eval_transform(_config(size=8))
    tensor = transform(_make_image())
    assert tensor[0, 0, 0] == pytest.approx((0.502 - 0.485) / 0.229, abs=1e-3)
    assert tensor[1, 0, 0] == pytest.approx((0.502 - 0.456) / 0.224, abs=1e-3)
    assert tensor[2, 0, 0] == pytest.approx((0.502 - 0.406) / 0.225, abs=1e-3)


def test_missing_image_size_raises():
    with pytest.raises(ConfigError, match="image_size"):
        build_eval_transform({"model": {}})


def test_invalid_image_size_raises():
    with pytest.raises(ConfigError, match="image_size"):
        build_eval_transform({"model": {"image_size": 0}})
