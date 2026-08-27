"""src/model.py — định nghĩa các kiến trúc model PAD.

Tệp này dùng để (theo mục 14 của tài liệu kỹ thuật):
- Cung cấp build_model(model_name, num_classes=1) để tạo model phân loại nhị phân
  bona_fide/spoof với MỘT logit đầu ra (số lớp mặc định = 1).
- Hỗ trợ các kiến trúc nhẹ:
      - "mobilenet_v2"  : MobileNetV2 (khuyến nghị làm baseline đầu tiên)
      - "mobilenet_v3"  : MobileNetV3-Small (thay thế)
      - "custom_cnn"    : CNN nhỏ tự viết (dùng để gỡ lỗi pipeline nhanh)
- Quy ước đầu ra: với nhị phân, model trả về 1 logit cho mỗi ảnh.
      logit = model(image)          # shape (B, 1)
      prob  = torch.sigmoid(logit)  # xác suất spoof
  Dùng torch.nn.BCEWithLogitsLoss trực tiếp với logit (KHÔNG sigmoid trước khi
  đưa vào loss, vì hàm loss đã bao gồm sigmoid).

Chú ý: module này KHÔNG chứa suy giảm chất lượng, KHÔNG chứa vòng lặp huấn luyện.
Chỉ định nghĩa kiến trúc và hàm tạo model.

Cách dùng:
    from src.model import build_model
    model = build_model("mobilenet_v2", num_classes=1)
    logits = model(batch_images)
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tv_models

VALID_MODEL_NAMES = ("mobilenet_v2", "mobilenet_v3", "custom_cnn")


class CustomCNN(nn.Module):
    """CNN nhỏ tự viết — dùng làm baseline gỡ lỗi (mục 14 tài liệu).

    Cấu trúc: 3 khối Conv2d -> ReLU -> MaxPool, sau đó AdaptiveAvgPool về
    4x4 nên KHÔNG phụ thuộc kích thước ảnh đầu vào, cuối cùng là 2 lớp Linear.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Truyền xuôi: trả về logits shape (B, num_classes)."""
        features = self.features(x)
        return self.classifier(features)


def build_model(
    model_name: str,
    num_classes: int = 1,
    pretrained: bool = False,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """Tạo model PAD theo tên kiến trúc (mục 14 tài liệu).

    Args:
        model_name: "mobilenet_v2", "mobilenet_v3" hoặc "custom_cnn".
        num_classes: Số logit đầu ra. Với phân loại nhị phân dùng 1 (mặc định).
        pretrained: Nếu True thì tải trọng số ImageNet (cần kết nối mạng).
        dropout_rate: Tỉ lệ dropout ở lớp phân loại (chỉ áp dụng cho MobileNet).

    Returns:
        Model nn.Module với forward trả về logits shape (B, num_classes).

    Raises:
        ValueError: nếu model_name không được hỗ trợ hoặc num_classes < 1.
    """
    if model_name not in VALID_MODEL_NAMES:
        raise ValueError(
            f"Unknown model name: {model_name!r}. Expected one of {list(VALID_MODEL_NAMES)}"
        )
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes must be an integer >= 1, got {num_classes!r}")

    if model_name == "mobilenet_v2":
        return _build_mobilenet_v2(num_classes, pretrained, dropout_rate)
    if model_name == "mobilenet_v3":
        return _build_mobilenet_v3(num_classes, pretrained, dropout_rate)
    return CustomCNN(num_classes=num_classes)


def _build_mobilenet_v2(
    num_classes: int,
    pretrained: bool,
    dropout_rate: float,
) -> nn.Module:
    """MobileNetV2: thay lớp classifier cuối bằng Linear -> num_classes logits."""
    weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v2(weights=weights)

    # Lớp classifier gốc: Sequential(Dropout, Linear(1280, 1000)).
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_mobilenet_v3(
    num_classes: int,
    pretrained: bool,
    dropout_rate: float,
) -> nn.Module:
    """MobileNetV3-Small: thay lớp classifier cuối bằng Linear -> num_classes logits."""
    weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = tv_models.mobilenet_v3_small(weights=weights)

    # Lớp classifier gốc: Sequential(Linear(576, 1024), HSwish, Dropout, Linear(1024, 1000)).
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, num_classes),
        nn.Dropout(p=dropout_rate),
    )
    return model
