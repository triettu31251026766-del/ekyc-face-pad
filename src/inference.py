"""src/inference.py — API dự đoán PAD cho một ảnh đơn lẻ.

Tệp này dùng để (theo mục 19 của tài liệu kỹ thuật):
- predict_image(model, image_path, transform, device): dự đoán một tệp ảnh,
  trả về {"probability_spoof": float, "prediction": "spoof" | "bona_fide"}.
- predict_pil(model, image, transform, device): tương tự nhưng nhận ảnh PIL
  trong bộ nhớ (tiện cho video.py gọi theo từng frame).

Quy ước (mục 6, 14 tài liệu):
    logit = model(image)            # 1 logit
    probability_spoof = sigmoid(logit)
    prediction = "spoof" nếu probability >= threshold, ngược lại "bona_fide"

Chú ý: KHÔNG tạo web server cho phiên bản đầu (mục 19). Module chỉ nhận ảnh
và trả kết quả; mọi logic khác (video, server) nằm ở module riêng.

Cách dùng:
    from src.inference import predict_image
    result = predict_image(model, "anh.jpg", eval_transform, device="cpu")
"""

from __future__ import annotations

from typing import Callable

import torch
from PIL import Image
from torch import nn

DEFAULT_THRESHOLD = 0.5


def predict_image(
    model: nn.Module,
    image_path: str,
    transform: Callable,
    device: str | torch.device = "cpu",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Dự đoán PAD cho một tệp ảnh (mục 19 tài liệu).

    Args:
        model: Model PAD đã huấn luyện (1 logit đầu ra).
        image_path: Đường dẫn tệp ảnh.
        transform: Transform eval tất định (xem src/transforms.py) nhận ảnh
            PIL và trả về Tensor.
        device: Thiết bị chạy ("cpu", "cuda") hoặc torch.device.
        threshold: Ngưỡng quyết định (mặc định 0.5).

    Returns:
        {"probability_spoof": float trong [0, 1],
         "prediction": "spoof" hoặc "bona_fide"}

    Raises:
        FileNotFoundError: nếu tệp ảnh không tồn tại.
    """
    import os

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    return predict_pil(model, image, transform, device=device, threshold=threshold)


def predict_pil(
    model: nn.Module,
    image: Image.Image,
    transform: Callable,
    device: str | torch.device = "cpu",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Dự đoán PAD cho một ảnh PIL trong bộ nhớ (từng frame của video chẳng hạn).

    Args:
        model: Model PAD đã huấn luyện (1 logit đầu ra).
        image: Ảnh PIL (sẽ được chuẩn hóa về RGB).
        transform: Transform eval tất định nhận ảnh PIL, trả về Tensor.
        device: Thiết bị chạy.
        threshold: Ngưỡng quyết định (mặc định 0.5).

    Returns:
        {"probability_spoof": float trong [0, 1],
         "prediction": "spoof" hoặc "bona_fide"}
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)  # thêm chiều batch

    with torch.no_grad():
        logits = model(tensor)
        probability = float(torch.sigmoid(logits).reshape(-1)[0].item())

    prediction = "spoof" if probability >= threshold else "bona_fide"
    return {"probability_spoof": probability, "prediction": prediction}
