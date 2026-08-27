"""tests/test_robustness.py — kiểm thử (unit test) cho module src/robustness.py.

Tệp này dùng để (theo mục 21, 27 của tài liệu kỹ thuật):
- Kiểm tra build_robustness_transform / apply_training_quality_augmentation
  hoạt động đúng với khối cấu hình "robustness".
- Kiểm tra robustness disabled -> ảnh giữ nguyên.
- Kiểm tra tăng cường thay đổi ảnh, giữ nguyên kích thước.
- Kiểm tra tái lập được: cùng seed -> cùng kết quả (nguồn ngẫu nhiên là
  module `random`, được set_seed kiểm soát).
- Kiểm tra mức độ tăng cường nằm trong khoảng cấu hình (brightness).
- Kiểm tra lỗi rõ ràng khi thiếu khối "robustness" hoặc spec sai.

Chạy kiểm thử:
    python -m pytest tests/test_robustness.py
"""

import random

import numpy as np
import pytest
from PIL import Image
from torchvision import transforms as T

from src.config import ConfigError
from src.robustness import (
    apply_training_quality_augmentation,
    build_robustness_transform,
)
from src.transforms import build_train_transform

SIZE = (64, 64)


def _structured_image():
    rng = np.random.default_rng(11)
    return Image.fromarray(rng.integers(0, 256, size=(SIZE[0], SIZE[1], 3)).astype(np.uint8))


def _config(enabled=True, probability=1.0, augmentations=("jpeg",)):
    """Cấu hình robustness nhỏ gọn cho test."""
    specs = {
        "jpeg": {"enabled": "jpeg" in augmentations, "quality_range": [30, 60], "probability": probability},
        "resize": {"enabled": "resize" in augmentations, "scale_range": [0.3, 0.8], "probability": probability},
        "blur": {"enabled": "blur" in augmentations, "sigma_range": [0.5, 2.0], "probability": probability},
        "noise": {"enabled": "noise" in augmentations, "std_range": [0.005, 0.03], "probability": probability},
        "brightness": {"enabled": "brightness" in augmentations, "factor_range": [0.7, 1.3], "probability": probability},
    }
    return {"seed": 42, "robustness": {"enabled": enabled, "augmentations": specs}}


def test_disabled_robustness_keeps_image():
    image = _structured_image()
    transform = build_robustness_transform(_config(enabled=False))
    result = transform(image)
    assert np.array_equal(np.asarray(result), np.asarray(image))


def test_missing_robustness_section_raises():
    with pytest.raises(ConfigError, match="robustness"):
        build_robustness_transform({"seed": 42})


def test_augmentation_changes_image_and_keeps_size():
    random.seed(0)
    image = _structured_image()
    result = apply_training_quality_augmentation(image, _config(probability=1.0))
    assert result.size == SIZE
    assert result.mode == "RGB"
    assert not np.array_equal(np.asarray(result), np.asarray(image))


def test_augmentation_reproducible_with_same_seed():
    image = _structured_image()
    config = _config(probability=1.0, augmentations=("jpeg", "resize", "blur", "noise"))

    random.seed(42)
    first = apply_training_quality_augmentation(image, config)
    random.seed(42)
    second = apply_training_quality_augmentation(image, config)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_zero_probability_keeps_image():
    random.seed(0)
    image = _structured_image()
    result = apply_training_quality_augmentation(
        image, _config(probability=0.0, augmentations=("jpeg", "noise"))
    )
    assert np.array_equal(np.asarray(result), np.asarray(image))


def test_brightness_stays_within_configured_range():
    random.seed(0)
    # Ảnh xám đồng đều: hệ số độ sáng = mean(sau)/mean(trước).
    image = Image.new("RGB", SIZE, color=(100, 100, 100))
    config = {
        "seed": 42,
        "robustness": {
            "enabled": True,
            "augmentations": {
                "brightness": {
                    "enabled": True,
                    "factor_range": [0.7, 1.3],
                    "probability": 1.0,
                }
            },
        },
    }
    for _ in range(10):
        result = apply_training_quality_augmentation(image, config)
        factor = np.asarray(result).mean() / np.asarray(image).mean()
        assert 0.69 <= factor <= 1.31  # dung sai nhỏ do làm tròn uint8


def test_blur_kernel_derived_from_sigma():
    """kernel suy ra từ sigma phải là số lẻ >= 3 (kiểm tra gián tiếp: không lỗi)."""
    random.seed(0)
    image = _structured_image()
    config = {
        "seed": 42,
        "robustness": {
            "enabled": True,
            "augmentations": {
                "blur": {"enabled": True, "sigma_range": [0.1, 3.0], "probability": 1.0}
            },
        },
    }
    result = apply_training_quality_augmentation(image, config)
    assert result.size == SIZE


def test_transform_composes_with_train_transform():
    """Robustness transform phải ghép được với transform chuẩn (mục 21)."""
    random.seed(0)
    image = _structured_image()
    config = _config(probability=1.0, augmentations=("jpeg",))
    composed = T.Compose(
        [
            T.Lambda(build_robustness_transform(config)),
            build_train_transform({"model": {"image_size": 32}}),
        ]
    )
    tensor = composed(image)
    assert tensor.shape == (3, 32, 32)


def test_invalid_probability_raises():
    random.seed(0)
    config = {
        "robustness": {
            "enabled": True,
            "augmentations": {
                "jpeg": {"enabled": True, "quality_range": [30, 60], "probability": 1.5}
            },
        }
    }
    with pytest.raises(ConfigError, match="probability"):
        apply_training_quality_augmentation(_structured_image(), config)


def test_missing_range_raises():
    random.seed(0)
    config = {
        "robustness": {
            "enabled": True,
            "augmentations": {"jpeg": {"enabled": True, "probability": 1.0}},
        }
    }
    with pytest.raises(ConfigError, match="quality_range"):
        apply_training_quality_augmentation(_structured_image(), config)
