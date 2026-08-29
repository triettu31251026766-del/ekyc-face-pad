"""src/train.py — vòng lặp huấn luyện PyTorch chung cho model PAD.

Tệp này dùng để (theo mục 16 của tài liệu kỹ thuật):
- Cung cấp train_one_epoch(model, dataloader, optimizer, loss_fn, device):
  huấn luyện đúng 1 epoch, trả về loss trung bình (float).
- Cung cấp train_model(...): chạy nhiều epoch, trả về lịch sử huấn luyện
  có cấu trúc:
      [{"epoch": 1, "train_loss": ..., "val_loss": ...}, ...]
- KHÔNG gắn với thí nghiệm cụ thể: không ghi tên tệp checkpoint, không chọn
  model, không quyết định chạy thí nghiệm nào (việc đó thuộc experiments/).
- Nhận optimizer và loss_fn từ caller (model-agnostic).
- Hỗ trợ callback on_epoch_end (tùy chọn) để experiments/ có thể lưu
  checkpoint sau mỗi epoch mà không cần sửa module này.

Quy ước batch: DataLoader trả về dict {"image": Tensor (B, C, H, W),
"label": Tensor (B,)} theo src/dataset.py. Nhãn được chuyển về (B, 1) để
khớp với BCEWithLogitsLoss (1 logit đầu ra — xem src/model.py).

Cách dùng:
    from src.train import train_model
    history = train_model(model, train_loader, val_loader,
                          optimizer, loss_fn, epochs=20, device="cpu")
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Callback được gọi sau mỗi epoch: nhận (model, epoch, metrics_dict).
# experiments/ dùng để lưu checkpoint mà không đưa logic thí nghiệm vào đây.
EpochCallback = Callable[[nn.Module, int, dict], None]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str | torch.device,
    epoch: Optional[int] = None,
) -> float:
    """Huấn luyện model đúng 1 epoch, trả về loss trung bình (có trọng số theo batch).

    Args:
        model: Model cần huấn luyện (sẽ được chuyển sang mode train()).
        dataloader: DataLoader trả về dict {"image", "label"}.
        optimizer: Optimizer đã khởi tạo với tham số của model.
        loss_fn: Hàm loss, ví dụ nn.BCEWithLogitsLoss.
        device: Thiết bị chạy ("cpu", "cuda") hoặc torch.device.
        epoch: Số thứ tự epoch (tùy chọn) — chỉ dùng để hiển thị trên thanh tiến độ.

    Returns:
        float: loss trung bình toàn epoch (tính theo tổng mẫu, không theo số batch).
    """
    device = torch.device(device)
    model.train()

    total_loss = 0.0
    total_samples = 0

    desc = f"train epoch {epoch}" if epoch is not None else "train"
    progress = tqdm(dataloader, desc=desc, unit="batch", leave=False, dynamic_ncols=True)

    for batch in progress:
        images = batch["image"].to(device)
        labels = _labels_to_column(batch["label"]).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Cộng dồn có trọng số theo số mẫu trong batch (batch cuối có thể thiếu mẫu).
        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        raise ValueError("dataloader is empty: cannot train_one_epoch with 0 samples")

    return total_loss / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: str | torch.device,
    on_epoch_end: Optional[EpochCallback] = None,
) -> list[dict]:
    """Huấn luyện model nhiều epoch, trả về lịch sử huấn luyện có cấu trúc.

    Args:
        model: Model cần huấn luyện.
        train_loader: DataLoader tập train.
        val_loader: DataLoader tập validation (nếu None thì val_loss = None).
        optimizer: Optimizer cho tham số model.
        loss_fn: Hàm loss (BCEWithLogitsLoss cho nhị phân).
        epochs: Số epoch huấn luyện.
        device: Thiết bị chạy ("cpu", "cuda") hoặc torch.device.
        on_epoch_end: Callback tùy chọn, gọi sau mỗi epoch với
            (model, epoch, {"train_loss": ..., "val_loss": ...}).

    Returns:
        list[dict] theo mục 16 tài liệu:
            [{"epoch": 1, "train_loss": 0.6, "val_loss": 0.5}, ...]

    Raises:
        ValueError: nếu epochs < 1 hoặc train_loader rỗng.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    device = torch.device(device)
    model.to(device)

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device,
                                     epoch=epoch)

        val_loss = None
        if val_loader is not None:
            val_loss = _compute_loader_loss(model, val_loader, loss_fn, device,
                                            epoch=epoch)

        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(metrics)

        # In tóm tắt mỗi epoch để người chạy theo dõi được tiến độ (mục 37 tài liệu).
        val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        tqdm.write(f"epoch {epoch}/{epochs}: train_loss={train_loss:.4f} | "
                   f"val_loss={val_str}")

        if on_epoch_end is not None:
            on_epoch_end(model, epoch, dict(metrics))

    return history


def _compute_loader_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: Optional[int] = None,
) -> float:
    """Tính loss trung bình trên một DataLoader ở mode eval() (không cập nhật tham số).

    Dùng cho validation loss trong quá trình huấn luyện.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    desc = f"val epoch {epoch}" if epoch is not None else "val"
    progress = tqdm(dataloader, desc=desc, unit="batch", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for batch in progress:
            images = batch["image"].to(device)
            labels = _labels_to_column(batch["label"]).to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * images.shape[0]
            total_samples += images.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        raise ValueError("dataloader is empty: cannot compute loss with 0 samples")

    return total_loss / total_samples


def _labels_to_column(labels: torch.Tensor) -> torch.Tensor:
    """Chuẩn hóa nhãn về shape (B, 1) float để khớp với output (B, 1) của model.

    - Tensor (B,) -> (B, 1)
    - Tensor (B, 1) -> giữ nguyên
    """
    if labels.ndim == 1:
        return labels.unsqueeze(1).float()
    return labels.float()
