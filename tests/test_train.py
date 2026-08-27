"""tests/test_train.py — kiểm thử (unit test) cho module src/train.py.

Tệp này dùng để (theo mục 16, 34 của tài liệu kỹ thuật):
- Kiểm tra train_one_epoch trả về loss hữu hạn (float).
- Kiểm tra train_model trả về lịch sử đúng cấu trúc:
  [{"epoch", "train_loss", "val_loss"}, ...] đúng số epoch.
- Kiểm tra val_loss = None khi không có val_loader, hữu hạn khi có.
- Smoke test: model nhỏ học được trên tập dữ liệu tổng hợp (loss giảm).
- Kiểm tra báo lỗi khi epochs < 1.

Chạy kiểm thử:
    python -m pytest tests/test_train.py
"""

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.data import Sample
from src.dataset import PADDataset
from src.model import build_model
from src.train import train_model, train_one_epoch


def _make_synthetic_loader(tmp_path, n_samples=16, seed=0):
    """Tạo DataLoader giả với ảnh nhiễu và nhãn ngẫu nhiên 0/1 (tất định theo seed)."""
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(n_samples):
        img = tmp_path / f"{i}.png"
        # Ảnh 32x32: class 0 nền tối, class 1 nền sáng (để model học được tín hiệu).
        label = i % 2
        base = 40 if label == 0 else 215
        Image.new("RGB", (32, 32), color=(base, base, base)).save(img)
        samples.append(
            Sample(path=str(img), label=label, subject_id=str(i),
                   attack_type="bona_fide" if label == 0 else "photo")
        )
    dataset = PADDataset(samples)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_train_one_epoch_returns_finite_loss(tmp_path):
    torch.manual_seed(0)
    loader = _make_synthetic_loader(tmp_path)
    model = build_model("custom_cnn", num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    loss = train_one_epoch(model, loader, optimizer, loss_fn, device="cpu")
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_train_model_history_structure(tmp_path):
    torch.manual_seed(0)
    loader = _make_synthetic_loader(tmp_path)
    model = build_model("custom_cnn", num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    history = train_model(model, loader, None, optimizer, loss_fn,
                          epochs=2, device="cpu")
    assert len(history) == 2
    for entry in history:
        assert set(entry.keys()) == {"epoch", "train_loss", "val_loss"}
        assert entry["val_loss"] is None
    assert history[0]["epoch"] == 1 and history[1]["epoch"] == 2


def test_train_model_with_val_loader(tmp_path):
    torch.manual_seed(0)
    train_loader = _make_synthetic_loader(tmp_path, n_samples=8, seed=0)
    val_loader = _make_synthetic_loader(tmp_path, n_samples=8, seed=1)
    model = build_model("custom_cnn", num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    history = train_model(model, train_loader, val_loader, optimizer, loss_fn,
                          epochs=2, device="cpu")
    assert history[0]["val_loss"] is not None
    assert torch.isfinite(torch.tensor(history[0]["val_loss"]))


def test_smoke_model_learns_small_dataset(tmp_path):
    """Smoke test: model nhỏ phải giảm loss rõ rệt trên 8 mẫu dễ học."""
    torch.manual_seed(0)
    loader = _make_synthetic_loader(tmp_path, n_samples=8)
    model = build_model("custom_cnn", num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    history = train_model(model, loader, None, optimizer, loss_fn,
                          epochs=8, device="cpu")
    assert history[0]["train_loss"] > history[-1]["train_loss"]
    assert history[-1]["train_loss"] < 0.35


def test_invalid_epochs_raises(tmp_path):
    torch.manual_seed(0)
    loader = _make_synthetic_loader(tmp_path, n_samples=4)
    model = build_model("custom_cnn", num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="epochs"):
        train_model(model, loader, None, optimizer, nn.BCEWithLogitsLoss(),
                    epochs=0, device="cpu")


def test_epoch_callback_is_called(tmp_path):
    torch.manual_seed(0)
    loader = _make_synthetic_loader(tmp_path, n_samples=4)
    model = build_model("custom_cnn", num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    calls = []

    train_model(model, loader, None, optimizer, nn.BCEWithLogitsLoss(),
                epochs=3, device="cpu",
                on_epoch_end=lambda m, e, metrics: calls.append((e, metrics)))
    assert [epoch for epoch, _ in calls] == [1, 2, 3]
    assert all("train_loss" in metrics for _, metrics in calls)
