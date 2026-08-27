"""src/transforms.py — tiền xử lý chuẩn cho ảnh (không phải suy giảm chất lượng).

Tệp này dùng để (theo mục 9 của tài liệu kỹ thuật):
- Xây dựng pipeline tiền xử lý chuẩn cho ảnh:
      ảnh (PIL)
        -> RGB
        -> resize (image_size x image_size)
        -> Tensor
        -> normalize (mean/std của ImageNet)
- build_train_transform(config): dùng khi HUẤN LUYỆN, được phép có tăng cường
  ngẫu nhiên (RandomHorizontalFlip).
- build_eval_transform(config): dùng khi ĐÁNH GIÁ, PHẢI tất định —
  KHÔNG chứa bất kỳ tăng cường ngẫu nhiên nào (mục 13 tài liệu).

Quan trọng:
- Module này KHÔNG chứa suy giảm chất lượng (JPEG, blur, ...); các hàm đó
  nằm ở src/degradation.py.
- Mọi transform suy giảm chất lượng cho đánh giá phải tất định và được
  áp dụng TRƯỚC bước resize/tensor/normalize của module này.

Cách dùng:
    from src.transforms import build_train_transform, build_eval_transform
    train_tf = build_train_transform(config)
    eval_tf = build_eval_transform(config)
"""

from __future__ import annotations

from torchvision import transforms as T

from src.config import ConfigError

# Giá trị chuẩn hóa theo ImageNet (MobileNetV2 dùng trọng số pretrained ImageNet).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(config: dict) -> T.Compose:
    """Transform cho tập TRAIN: resize + flip ngang ngẫu nhiên + tensor + normalize.

    Args:
        config: Cấu hình huấn luyện (cần có "model.image_size").

    Returns:
        torchvision Compose nhận ảnh PIL, trả về Tensor chuẩn hóa.

    Raises:
        ConfigError: nếu thiếu "model.image_size" trong config.
    """
    size = _image_size(config)
    return T.Compose(
        [
            T.Resize((size, size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(config: dict) -> T.Compose:
    """Transform cho tập EVAL/TEST: resize + tensor + normalize (tất định).

    Quy tắc (mục 9, 13 tài liệu): KHÔNG được chứa tăng cường ngẫu nhiên.
    Cùng một ảnh đầu vào phải luôn cho cùng một Tensor đầu ra.

    Args:
        config: Cấu hình đánh giá (cần có "model.image_size").

    Returns:
        torchvision Compose nhận ảnh PIL, trả về Tensor chuẩn hóa.

    Raises:
        ConfigError: nếu thiếu "model.image_size" trong config.
    """
    size = _image_size(config)
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _image_size(config: dict) -> int:
    """Lấy kích thước ảnh đầu vào từ cấu hình, báo lỗi rõ ràng nếu thiếu."""
    try:
        size = int(config["model"]["image_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(
            "Config must contain 'model.image_size' (an integer) to build transforms"
        ) from exc
    if size < 1:
        raise ConfigError(f"'model.image_size' must be >= 1, got {size}")
    return size
