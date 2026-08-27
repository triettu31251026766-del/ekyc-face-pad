"""src/degradation.py — suy giảm chất lượng ảnh có kiểm soát (tất định).

Tệp này là module TRUNG TÂM của dự án (theo mục 10-13 của tài liệu kỹ thuật):
- Cung cấp 5 phép suy giảm chất lượng, độc lập hoàn toàn với model:
      jpeg_compression(image, quality)    : nén JPEG với quality 1-100
      resize_degradation(image, scale)    : hạ độ phân giải theo tỉ lệ scale
      gaussian_blur(image, kernel, sigma) : làm mờ Gaussian
      gaussian_noise(image, std, seed)    : thêm nhiễu Gaussian (std theo thang 0-1)
      brightness_adjustment(image, factor): chỉnh độ sáng theo hệ số nhân
- Unified API (mục 10): apply_degradation(image, name, severity, seed=...).
- Tham số tường minh, có kiểm tra hợp lệ (mục 11, 12): KHÔNG dùng tên mơ hồ
  như "strong"/"weak", mọi giá trị phải bằng số cụ thể.

QUY TẮC TẤT ĐỊNH (mục 13 tài liệu) — rất quan trọng:
- Khi ĐÁNH GIÁ: cùng ảnh + cùng tham số (+ cùng seed cho noise) PHẢI cho
  cùng một ảnh kết quả. Không dùng ngẫu nhiên trừ khi có seed tường minh.
- Khi HUẤN LUYỆN (robustness): phép tăng cường ngẫu nhiên nằm ở
  src/robustness.py và GỌI các hàm của module này.

Quy ước dữ liệu:
- Hàm nhận ảnh PIL (bất kỳ mode, sẽ chuẩn hóa về RGB) và trả về ảnh PIL RGB
  cùng kích thước với đầu vào.
- resize_degradation: thu nhỏ về (W*scale, H*scale) rồi PHÓNG LẠI kích thước
  gốc, mô phỏng ảnh chụp ở độ phân giải thấp nhưng vẫn giữ nguyên kích thước
  đầu vào cho model (mục 23 tài liệu: cùng model, chỉ thay đổi chất lượng input).

Chú ý: module này KHÔNG tính metric, KHÔNG huấn luyện model, KHÔNG chứa
transform tiền xử lý chuẩn (xem src/transforms.py).

Cách dùng:
    from PIL import Image
    from src.degradation import apply_degradation
    image = Image.open("anh.jpg")
    degraded = apply_degradation(image, "jpeg", severity=50)
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

VALID_DEGRADATIONS = ("jpeg", "resize", "blur", "noise", "brightness")

# Hằng số chất lượng JPEG tối đa / tối thiểu theo chuẩn PIL.
JPEG_QUALITY_MIN = 1
JPEG_QUALITY_MAX = 100


def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Nén JPEG với chất lượng cho trước (mục 12 tài liệu).

    Args:
        image: Ảnh PIL đầu vào.
        quality: Chất lượng JPEG, số nguyên từ 1 (tệ nhất) đến 100 (tốt nhất).

    Returns:
        Ảnh PIL RGB cùng kích thước, đã qua nén-giải nén JPEG.

    Raises:
        ValueError: nếu quality ngoài khoảng [1, 100] hoặc không phải số nguyên.
    """
    if not isinstance(quality, int) or isinstance(quality, bool):
        raise ValueError(f"jpeg quality must be an integer, got {quality!r}")
    if not (JPEG_QUALITY_MIN <= quality <= JPEG_QUALITY_MAX):
        raise ValueError(
            f"jpeg quality must be in [{JPEG_QUALITY_MIN}, {JPEG_QUALITY_MAX}], "
            f"got {quality}"
        )

    image = _to_rgb(image)
    # Ghi vào bộ đệm rồi đọc lại để mô phỏng đúng pipeline nén -> giải nén.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def resize_degradation(image: Image.Image, scale: float) -> Image.Image:
    """Hạ độ phân giải: thu nhỏ theo tỉ lệ scale rồi phóng lại kích thước gốc.

    Mô phỏng ảnh được chụp/ghi ở độ phân giải thấp (mục 12, 23 tài liệu).
    Ảnh kết quả có CÙNG kích thước với ảnh đầu vào.

    Args:
        image: Ảnh PIL đầu vào.
        scale: Tỉ lệ thu nhỏ, số thực trong khoảng (0, 1].

    Returns:
        Ảnh PIL RGB cùng kích thước với đầu vào.

    Raises:
        ValueError: nếu scale ngoài khoảng (0, 1].
    """
    scale = _require_float(scale, "scale")
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"resize scale must be in (0, 1], got {scale}")

    image = _to_rgb(image)
    width, height = image.size
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # BILINEAR cho cả 2 bước: xuống rồi lên, kết quả tất định với cùng tham số.
    small = image.resize(small_size, Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.BILINEAR)


def gaussian_blur(
    image: Image.Image,
    kernel_size: int,
    sigma: float,
) -> Image.Image:
    """Làm mờ Gaussian với kernel_size và sigma tường minh (mục 12 tài liệu).

    Args:
        image: Ảnh PIL đầu vào.
        kernel_size: Kích thước kernel, số nguyên LẺ >= 1 (1 = không đổi).
        sigma: Độ lệch chuẩn Gaussian, số thực > 0.

    Returns:
        Ảnh PIL RGB cùng kích thước.

    Raises:
        ValueError: nếu kernel_size chẵn/nhỏ hơn 1 hoặc sigma <= 0.
    """
    if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
        raise ValueError(f"kernel_size must be an integer, got {kernel_size!r}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be an odd integer >= 1, got {kernel_size}")
    sigma = _require_float(sigma, "sigma")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    image = _to_rgb(image)
    if kernel_size == 1:
        # Kernel 1x1 là phép đồng nhất: trả bản sao để giữ ngữ nghĩa "không đổi".
        return image.copy()

    # Dùng OpenCV để tôn trọng đúng cả kernel_size lẫn sigma.
    array = np.asarray(image)
    blurred = cv2.GaussianBlur(array, (kernel_size, kernel_size), sigmaX=sigma)
    return Image.fromarray(blurred)


def gaussian_noise(
    image: Image.Image,
    std: float,
    seed: int | None = None,
) -> Image.Image:
    """Thêm nhiễu Gaussian với độ lệch chuẩn std (mục 12 tài liệu).

    std tính theo thang cường độ 0-1 (ví dụ 0.03 ~= 7.65/255), khớp với
    cấu hình robustness.yaml: std_range: [0.005, 0.03].

    TÍNH TẤT ĐỊNH (mục 13): nếu truyền seed thì cùng ảnh + cùng std + cùng
    seed luôn cho cùng kết quả. Khi đánh giá PHẢI truyền seed.

    Args:
        image: Ảnh PIL đầu vào.
        std: Độ lệch chuẩn nhiễu (thang 0-1), số thực >= 0.
        seed: Seed cho bộ sinh nhiễu (None = ngẫu nhiên, chỉ dùng khi train).

    Returns:
        Ảnh PIL RGB cùng kích thước.

    Raises:
        ValueError: nếu std < 0.
    """
    std = _require_float(std, "std")
    if std < 0.0:
        raise ValueError(f"std must be >= 0, got {std}")

    image = _to_rgb(image)
    if std == 0.0:
        return image.copy()

    array = np.asarray(image).astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=array.shape)
    noisy = np.clip(array + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).round().astype(np.uint8))


def brightness_adjustment(image: Image.Image, factor: float) -> Image.Image:
    """Chỉnh độ sáng theo hệ số nhân factor (mục 12 tài liệu).

    Args:
        image: Ảnh PIL đầu vào.
        factor: Hệ số nhân độ sáng, số thực > 0.
            factor < 1 -> tối hơn, factor > 1 -> sáng hơn, factor = 1 -> không đổi.

    Returns:
        Ảnh PIL RGB cùng kích thước.

    Raises:
        ValueError: nếu factor <= 0.
    """
    factor = _require_float(factor, "factor")
    if factor <= 0.0:
        raise ValueError(f"factor must be > 0, got {factor}")

    image = _to_rgb(image)
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_degradation(
    image: Image.Image,
    degradation_name: str,
    severity: Any,
    seed: int | None = None,
) -> Image.Image:
    """API thống nhất để áp dụng suy giảm chất lượng (mục 10 tài liệu).

    Args:
        image: Ảnh PIL đầu vào.
        degradation_name: "jpeg", "resize", "blur", "noise" hoặc "brightness".
        severity: Tham số của phép suy giảm:
            - jpeg       : int (quality)
            - resize     : float (scale)
            - blur       : tuple (kernel_size, sigma)
            - noise      : float (std)
            - brightness : float (factor)
        seed: Seed cho nhiễu (chỉ dùng với "noise", để đánh giá tất định).

    Returns:
        Ảnh PIL RGB cùng kích thước.

    Raises:
        ValueError: nếu tên phép suy giảm không hợp lệ hoặc severity sai dạng.

    Ví dụ (mục 10 tài liệu):
        apply_degradation(image, degradation_name="jpeg", severity=50)
    """
    if degradation_name not in VALID_DEGRADATIONS:
        raise ValueError(
            f"Unknown degradation: {degradation_name!r}. "
            f"Expected one of {list(VALID_DEGRADATIONS)}"
        )

    if degradation_name == "jpeg":
        return jpeg_compression(image, int(severity))
    if degradation_name == "resize":
        return resize_degradation(image, float(severity))
    if degradation_name == "blur":
        # severity là tuple (kernel_size, sigma).
        if not isinstance(severity, (tuple, list)) or len(severity) != 2:
            raise ValueError(
                f"blur severity must be (kernel_size, sigma), got {severity!r}"
            )
        return gaussian_blur(image, int(severity[0]), float(severity[1]))
    if degradation_name == "noise":
        return gaussian_noise(image, float(severity), seed=seed)
    return brightness_adjustment(image, float(severity))


def apply_degradation_config(image: Image.Image, config: dict) -> Image.Image:
    """Áp dụng suy giảm chất lượng từ cấu hình {"name": ..., tham số...}.

    Đọc trực tiếp khối "degradation" của các tệp configs/degradation_*.yaml
    (mục 26 tài liệu), giúp thí nghiệm đánh giá tất định theo cấu hình.

    Args:
        image: Ảnh PIL đầu vào.
        config: Dict cấu hình, ví dụ {"name": "jpeg", "quality": 50}.

    Returns:
        Ảnh PIL RGB cùng kích thước.
    """
    degradation = config.get("degradation", config)
    name = degradation.get("name")
    if name not in VALID_DEGRADATIONS:
        raise ValueError(
            f"Unknown degradation in config: {name!r}. "
            f"Expected one of {list(VALID_DEGRADATIONS)}"
        )

    if name == "jpeg":
        return jpeg_compression(image, int(degradation["quality"]))
    if name == "resize":
        return resize_degradation(image, float(degradation["scale"]))
    if name == "blur":
        return gaussian_blur(
            image, int(degradation["kernel_size"]), float(degradation["sigma"])
        )
    if name == "noise":
        # Seed lấy từ config để đánh giá TẤT ĐỊNH (mục 13 tài liệu).
        return gaussian_noise(image, float(degradation["std"]), seed=config.get("seed"))
    return brightness_adjustment(image, float(degradation["factor"]))


# --- Các hàm nội bộ ---


def _to_rgb(image: Image.Image) -> Image.Image:
    """Chuẩn hóa ảnh về mode RGB (ảnh xám/P chỉ có < 3 kênh)."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _require_float(value: Any, name: str) -> float:
    """Ép giá trị về float, báo lỗi rõ ràng nếu không phải số."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return float(value)
