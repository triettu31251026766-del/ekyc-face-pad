"""tests/test_degradation.py — kiểm thử (unit test) cho module src/degradation.py.

Tệp này dùng để (theo mục 34, 35 của tài liệu kỹ thuật):
- Dùng ẢNH TỔNG HỢP (mục 35) để cô lập lỗi xử lý ảnh khỏi lỗi dataset/model.
- Kiểm tra: JPEG output khác ảnh gốc; resize/blur/noise/brightness giữ nguyên
  kích thước; tham số không hợp lệ phải báo lỗi rõ ràng.
- Kiểm tra tính TẤT ĐỊNH của từng phép suy giảm (mục 13 tài liệu).

Chạy kiểm thử:
    python -m pytest tests/test_degradation.py
"""

import numpy as np
import pytest
from PIL import Image

from src.degradation import (
    apply_degradation,
    apply_degradation_config,
    brightness_adjustment,
    gaussian_blur,
    gaussian_noise,
    jpeg_compression,
    resize_degradation,
)

SIZE = (224, 224)


def _structured_image():
    """Ảnh tổng hợp có cấu trúc (nhiều mức xám) — KHÔNG dùng ảnh phẳng một màu
    vì ảnh phẳng có thể nén JPEG gần như không thay đổi pixel."""
    rng = np.random.default_rng(7)
    array = (rng.integers(0, 256, size=(SIZE[0], SIZE[1], 3))).astype(np.uint8)
    return Image.fromarray(array)


def _flat_image():
    return Image.new("RGB", SIZE, color="gray")


# --- JPEG ---


def test_jpeg_output_differs_from_original():
    image = _structured_image()
    degraded = jpeg_compression(image, quality=30)
    assert degraded.size == image.size
    assert degraded.mode == "RGB"
    assert not np.array_equal(np.asarray(degraded), np.asarray(image))


def test_jpeg_quality_100_keeps_size():
    image = _structured_image()
    degraded = jpeg_compression(image, quality=100)
    assert degraded.size == image.size


@pytest.mark.parametrize("quality", [0, 101, -5])
def test_jpeg_invalid_quality_raises(quality):
    with pytest.raises(ValueError, match="quality"):
        jpeg_compression(_flat_image(), quality)


def test_jpeg_non_integer_quality_raises():
    with pytest.raises(ValueError, match="quality"):
        jpeg_compression(_flat_image(), quality=50.5)


# --- Resize ---


def test_resize_output_has_expected_dimensions():
    image = _structured_image()
    degraded = resize_degradation(image, scale=0.5)
    assert degraded.size == SIZE  # thu nhỏ rồi phóng lại kích thước gốc
    assert degraded.mode == "RGB"


def test_resize_output_differs_from_original():
    image = _structured_image()
    degraded = resize_degradation(image, scale=0.25)
    assert not np.array_equal(np.asarray(degraded), np.asarray(image))


def test_resize_scale_1_keeps_pixels():
    image = _structured_image()
    degraded = resize_degradation(image, scale=1.0)
    assert np.array_equal(np.asarray(degraded), np.asarray(image))


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.5])
def test_resize_invalid_scale_raises(scale):
    with pytest.raises(ValueError, match="scale"):
        resize_degradation(_flat_image(), scale)


# --- Blur ---


def test_blur_output_has_expected_dimensions():
    image = _structured_image()
    degraded = gaussian_blur(image, kernel_size=7, sigma=2.0)
    assert degraded.size == SIZE
    assert degraded.mode == "RGB"


def test_blur_is_deterministic():
    image = _structured_image()
    first = gaussian_blur(image, kernel_size=5, sigma=1.5)
    second = gaussian_blur(image, kernel_size=5, sigma=1.5)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_blur_output_differs_from_original():
    image = _structured_image()
    degraded = gaussian_blur(image, kernel_size=7, sigma=2.0)
    assert not np.array_equal(np.asarray(degraded), np.asarray(image))


def test_blur_even_kernel_raises():
    with pytest.raises(ValueError, match="odd"):
        gaussian_blur(_flat_image(), kernel_size=8, sigma=2.0)


def test_blur_non_positive_sigma_raises():
    with pytest.raises(ValueError, match="sigma"):
        gaussian_blur(_flat_image(), kernel_size=5, sigma=0.0)


# --- Noise ---


def test_noise_output_has_expected_dimensions():
    image = _structured_image()
    degraded = gaussian_noise(image, std=0.03, seed=42)
    assert degraded.size == SIZE
    assert degraded.mode == "RGB"


def test_noise_output_differs_from_original():
    image = _structured_image()
    degraded = gaussian_noise(image, std=0.03, seed=42)
    assert not np.array_equal(np.asarray(degraded), np.asarray(image))


def test_noise_deterministic_with_same_seed():
    image = _structured_image()
    first = gaussian_noise(image, std=0.03, seed=42)
    second = gaussian_noise(image, std=0.03, seed=42)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_noise_different_seed_gives_different_output():
    image = _structured_image()
    first = gaussian_noise(image, std=0.03, seed=1)
    second = gaussian_noise(image, std=0.03, seed=2)
    assert not np.array_equal(np.asarray(first), np.asarray(second))


def test_noise_zero_std_keeps_pixels():
    image = _structured_image()
    degraded = gaussian_noise(image, std=0.0, seed=42)
    assert np.array_equal(np.asarray(degraded), np.asarray(image))


def test_noise_negative_std_raises():
    with pytest.raises(ValueError, match="std"):
        gaussian_noise(_flat_image(), std=-0.1)


# --- Brightness ---


def test_brightness_output_has_expected_dimensions():
    image = _structured_image()
    degraded = brightness_adjustment(image, factor=0.7)
    assert degraded.size == SIZE
    assert degraded.mode == "RGB"


def test_brightness_darkens_with_factor_below_1():
    image = _flat_image()  # xám đồng đều để so sánh giá trị trung bình
    degraded = brightness_adjustment(image, factor=0.5)
    assert np.asarray(degraded).mean() < np.asarray(image).mean()


def test_brightness_factor_1_keeps_pixels():
    image = _structured_image()
    degraded = brightness_adjustment(image, factor=1.0)
    assert np.array_equal(np.asarray(degraded), np.asarray(image))


def test_brightness_non_positive_factor_raises():
    with pytest.raises(ValueError, match="factor"):
        brightness_adjustment(_flat_image(), factor=0.0)


# --- Unified API ---


def test_apply_degradation_jpeg_matches_direct_call():
    image = _structured_image()
    unified = apply_degradation(image, "jpeg", severity=50)
    direct = jpeg_compression(image, quality=50)
    assert np.array_equal(np.asarray(unified), np.asarray(direct))


def test_apply_degradation_blur_with_tuple():
    image = _structured_image()
    unified = apply_degradation(image, "blur", severity=(7, 2.0))
    direct = gaussian_blur(image, kernel_size=7, sigma=2.0)
    assert np.array_equal(np.asarray(unified), np.asarray(direct))


def test_apply_degradation_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown degradation"):
        apply_degradation(_flat_image(), "solarize", severity=1)


def test_apply_degradation_bad_blur_severity_raises():
    with pytest.raises(ValueError, match="blur"):
        apply_degradation(_flat_image(), "blur", severity=7)


def test_apply_degradation_config_from_dict():
    image = _structured_image()
    config = {"seed": 42, "degradation": {"name": "jpeg", "quality": 50}}
    unified = apply_degradation_config(image, config)
    direct = jpeg_compression(image, quality=50)
    assert np.array_equal(np.asarray(unified), np.asarray(direct))


def test_apply_degradation_config_noise_uses_seed():
    image = _structured_image()
    config = {"seed": 42, "degradation": {"name": "noise", "std": 0.03}}
    first = apply_degradation_config(image, config)
    second = apply_degradation_config(image, config)
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_apply_degradation_config_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown degradation"):
        apply_degradation_config(_flat_image(), {"degradation": {"name": "flip"}})
