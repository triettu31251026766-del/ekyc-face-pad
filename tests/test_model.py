"""tests/test_model.py — kiểm thử (unit test) cho module src/model.py.

Tệp này dùng để (theo mục 34 của tài liệu kỹ thuật):
- Kiểm tra tạo model với từng tên kiến trúc hỗ trợ.
- Kiểm tra forward pass với input chuẩn (B, 3, 224, 224) cho ra output đúng shape.
- Kiểm tra quy ước 1 logit cho phân loại nhị phân (mục 14).
- Kiểm tra tính toán loss bằng BCEWithLogitsLoss (dùng logit trực tiếp, không sigmoid trước).
- Kiểm tra báo lỗi khi tên model / num_classes không hợp lệ.

Chạy kiểm thử:
    python -m pytest tests/test_model.py
"""

import pytest
import torch
from torch import nn

from src.model import build_model


def test_build_model_all_names():
    for name in ("mobilenet_v2", "mobilenet_v3", "custom_cnn"):
        model = build_model(name, num_classes=1)
        assert isinstance(model, nn.Module)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("resnet5000")


def test_invalid_num_classes_raises():
    with pytest.raises(ValueError, match="num_classes"):
        build_model("custom_cnn", num_classes=0)


def test_forward_output_shape_binary():
    """Với num_classes=1, output phải là (B, 1) — một logit mỗi ảnh."""
    x = torch.randn(2, 3, 224, 224)
    for name in ("mobilenet_v2", "mobilenet_v3", "custom_cnn"):
        model = build_model(name, num_classes=1)
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 1), f"{name} cho output {y.shape}, mong đợi (2, 1)"


def test_forward_output_shape_multiclass():
    model = build_model("custom_cnn", num_classes=2)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(4, 3, 64, 64))
    assert y.shape == (4, 2)


def test_custom_cnn_any_input_size():
    """custom_cnn dùng AdaptiveAvgPool nên chạy được với kích thước ảnh khác nhau."""
    model = build_model("custom_cnn", num_classes=1)
    model.eval()
    for size in (32, 96, 224):
        with torch.no_grad():
            y = model(torch.randn(2, 3, size, size))
        assert y.shape == (2, 1)


def test_sigmoid_gives_probability():
    """Quy ước mục 14: probability = sigmoid(logit) phải nằm trong [0, 1]."""
    model = build_model("custom_cnn", num_classes=1)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(8, 3, 64, 64))
    probs = torch.sigmoid(logits)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))


def test_loss_with_bce_with_logits():
    """BCEWithLogitsLoss nhận logit trực tiếp và tính được loss hữu hạn (mục 14)."""
    model = build_model("custom_cnn", num_classes=1)
    loss_fn = nn.BCEWithLogitsLoss()
    x = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    logits = model(x)
    loss = loss_fn(logits, labels)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_different_models_have_different_parameters():
    model_a = build_model("custom_cnn", num_classes=1)
    model_b = build_model("custom_cnn", num_classes=1)
    # Hai model khởi tạo riêng biệt phải độc lập về tham số.
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert p_a is not p_b
