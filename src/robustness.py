"""src/robustness.py — tăng cường chất lượng cho huấn luyện robustness.

Tệp này dùng để (theo mục 21, 27 của tài liệu kỹ thuật):
- Xây dựng phép tăng cường suy giảm chất lượng NGẪU NHIÊN cho TẬP TRAIN:
      ảnh gốc -> (có thể) JPEG / resize / blur / noise / brightness -> model
- build_robustness_transform(config): trả về hàm nhận ảnh PIL, trả về ảnh PIL
  đã tăng cường — dùng kèm với transform chuẩn của src/transforms.py.
- apply_training_quality_augmentation(image, config): áp dụng tăng cường cho
  một ảnh dựa trên khối "robustness" trong cấu hình (configs/robustness.yaml).

Cách hoạt động (đơn giản — mục 21 tài liệu):
- Với mỗi phép tăng cường có "enabled: true", mẫu ảnh được áp dụng với xác suất
  "probability" (mặc định 0.5) — nhiều phép có thể chồng nhau.
- Mức độ (severity) được lấy NGẪU NHIÊN đều trong khoảng cho trước:
      jpeg      : quality  ~ U[quality_range]
      resize    : scale    ~ U[scale_range]
      blur      : sigma    ~ U[sigma_range] (kernel_size suy ra từ sigma)
      noise     : std      ~ U[std_range]
      brightness: factor   ~ U[factor_range]

QUAN TRỌNG:
- Module này KHÔNG viết lại các phép suy giảm — chỉ GỌI src/degradation.py
  (quy tắc mục 21: không trùng lặp cài đặt).
- Ngẫu nhiên CHỈ dùng cho huấn luyện (mục 13 tài liệu); đánh giá luôn dùng
  degradation.py với tham số cố định.
- Nguồn ngẫu nhiên là module `random` của Python, được kiểm soát bởi
  src/reproducibility.set_seed -> kết quả tăng cường có thể tái lập.

Cách dùng:
    from torchvision import transforms as T
    from src.robustness import build_robustness_transform
    from src.transforms import build_train_transform

    train_tf = T.Compose([
        T.Lambda(build_robustness_transform(config)),
        build_train_transform(config),
    ])
"""

from __future__ import annotations

import random
from typing import Callable

from PIL import Image

from src.config import ConfigError
from src.degradation import (
    brightness_adjustment,
    gaussian_blur,
    gaussian_noise,
    jpeg_compression,
    resize_degradation,
)

# Xác suất mặc định mỗi phép tăng cường được áp dụng cho một mẫu.
DEFAULT_PROBABILITY = 0.5

# Kernel blur tối thiểu (số lẻ) khi sigma quá nhỏ.
MIN_BLUR_KERNEL = 3


def build_robustness_transform(config: dict) -> Callable[[Image.Image], Image.Image]:
    """Trả về hàm tăng cường chất lượng ngẫu nhiên cho ảnh huấn luyện.

    Args:
        config: Cấu hình chứa khối "robustness" (xem configs/robustness.yaml).

    Returns:
        Hàm nhận ảnh PIL và trả về ảnh PIL đã tăng cường. Nếu
        "robustness.enabled" là False thì trả về hàm đồng nhất (giữ nguyên ảnh).

    Raises:
        ConfigError: nếu thiếu khối "robustness" trong config.
    """
    robustness = _require_robustness(config)
    if not robustness.get("enabled", False):
        return lambda image: image

    return lambda image: apply_training_quality_augmentation(image, config)


def apply_training_quality_augmentation(image: Image.Image, config: dict) -> Image.Image:
    """Áp dụng tăng cường suy giảm chất lượng ngẫu nhiên cho MỘT ảnh (mục 21).

    Mỗi phép tăng cường bật trong config được áp dụng với xác suất riêng
    (mặc định 0.5), mức độ lấy ngẫu nhiên trong khoảng cấu hình. Các phép
    được xét theo thứ tự cố định để kết quả tái lập được khi seed giống nhau.

    Args:
        image: Ảnh PIL đầu vào.
        config: Cấu hình chứa khối "robustness".

    Returns:
        Ảnh PIL đã tăng cường (cùng kích thước).

    Raises:
        ConfigError: nếu thiếu khối "robustness" hoặc spec phép tăng cường sai.
    """
    robustness = _require_robustness(config)
    if not robustness.get("enabled", False):
        return image

    augmentations = robustness.get("augmentations", {})
    # Thứ tự cố định giúp kết quả tất định khi cùng seed.
    for name in ("jpeg", "resize", "blur", "noise", "brightness"):
        spec = augmentations.get(name)
        if not spec or not spec.get("enabled", False):
            continue

        probability = spec.get("probability", DEFAULT_PROBABILITY)
        if not isinstance(probability, (int, float)) or not (0.0 <= probability <= 1.0):
            raise ConfigError(
                f"robustness.augmentations.{name}.probability must be in [0, 1], "
                f"got {probability!r}"
            )

        if random.random() >= probability:
            continue  # mẫu này không áp dụng phép tăng cường đang xét.

        image = _apply_one(name, image, spec)

    return image


# --- Các hàm nội bộ ---


def _require_robustness(config: dict) -> dict:
    """Lấy khối "robustness" từ config, báo lỗi rõ ràng nếu thiếu."""
    if not isinstance(config, dict) or "robustness" not in config:
        raise ConfigError("Config must contain a 'robustness' section")
    robustness = config["robustness"]
    if not isinstance(robustness, dict):
        raise ConfigError(f"'robustness' must be a mapping, got {type(robustness).__name__}")
    return robustness


def _apply_one(name: str, image: Image.Image, spec: dict) -> Image.Image:
    """Áp dụng MỘT phép tăng cường với mức độ ngẫu nhiên trong khoảng cấu hình."""
    if name == "jpeg":
        low, high = _require_range(name, spec, "quality_range")
        quality = random.randint(int(low), int(high))
        return jpeg_compression(image, quality)

    if name == "resize":
        low, high = _require_range(name, spec, "scale_range")
        scale = random.uniform(float(low), float(high))
        return resize_degradation(image, scale)

    if name == "blur":
        low, high = _require_range(name, spec, "sigma_range")
        sigma = random.uniform(float(low), float(high))
        kernel_size = _kernel_from_sigma(sigma)
        return gaussian_blur(image, kernel_size, sigma)

    if name == "noise":
        low, high = _require_range(name, spec, "std_range")
        std = random.uniform(float(low), float(high))
        # Sinh seed nhiễu từ luồng `random` chung để toàn bộ chuỗi tăng cường
        # tái lập được khi gọi set_seed (mục 13, 36 tài liệu).
        noise_seed = random.randint(0, 2**32 - 1)
        return gaussian_noise(image, std, seed=noise_seed)

    # brightness
    low, high = _require_range(name, spec, "factor_range")
    factor = random.uniform(float(low), float(high))
    return brightness_adjustment(image, factor)


def _require_range(name: str, spec: dict, key: str) -> tuple[float, float]:
    """Lấy khoảng [low, high] của phép tăng cường, báo lỗi nếu sai dạng."""
    value = spec.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(
            f"robustness.augmentations.{name}.{key} must be a list of two numbers, "
            f"got {value!r}"
        )
    low, high = value
    if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ConfigError(
            f"robustness.augmentations.{name}.{key} must be a list of two numbers, "
            f"got {value!r}"
        )
    if low < 0 or high < 0 or low > high:
        raise ConfigError(
            f"robustness.augmentations.{name}.{key} must be [low, high] with "
            f"0 <= low <= high, got {value!r}"
        )
    return float(low), float(high)


def _kernel_from_sigma(sigma: float) -> int:
    """Suy ra kernel_size (số lẻ) từ sigma: xấp xỉ 6*sigma rồi làm tròn lên số lẻ.

    Quy tắc kinh điển: kernel ~ 6*sigma + 1 để bao phủ gần trọn phân phối
    Gaussian. Không nằm trong cấu hình (mục 27 chỉ có sigma_range) nên được
    suy ra tường minh và ghi chú ở đây.
    """
    kernel = int(round(sigma * 6.0))
    if kernel % 2 == 0:
        kernel += 1
    return max(kernel, MIN_BLUR_KERNEL)
