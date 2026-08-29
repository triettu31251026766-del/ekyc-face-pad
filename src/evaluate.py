"""src/evaluate.py — pipeline đánh giá model PAD (tất định).

Tệp này dùng để (theo mục 18 của tài liệu kỹ thuật):
- evaluate_model(model, dataloader, device, threshold): chạy model ở mode
  eval() trên toàn bộ DataLoader, trả về:
      {
          "metrics": {...},          # accuracy, f1, roc_auc, apcer, bpcer, acer, ...
          "predictions": [...],      # nhãn dự đoán nhị phân 0/1
          "probabilities": [...],    # xác suất spoof
          "labels": [...],           # nhãn thật
          "paths": [...],            # đường dẫn ảnh (phục vụ phân tích lỗi)
          "subject_ids": [...],
          "attack_types": [...],
      }
- save_predictions(result, path): lưu CSV dự đoán thô theo mục 39 tài liệu
  với các cột: path, subject_id, attack_type, label, probability_spoof,
  prediction, correct — để phân tích false positive / false negative về sau.

Yêu cầu quan trọng:
- Đánh giá PHẢI tất định: không có bất kỳ tăng cường ngẫu nhiên nào ở đây.
  Caller phải dùng DataLoader với shuffle=False và transform eval tất định
  (xem src/transforms.py). Cùng model + cùng dữ liệu => cùng kết quả.
- Module này KHÔNG huấn luyện, KHÔNG áp dụng suy giảm chất lượng.

Cách dùng:
    from src.evaluate import evaluate_model, save_predictions
    result = evaluate_model(model, test_loader, device="cpu", threshold=0.5)
    save_predictions(result, "results/raw/E01_baseline_seed123.csv")
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import acer, apcer, bpcer, classification_metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    threshold: float = 0.5,
) -> dict:
    """Đánh giá model trên toàn bộ dataloader một cách tất định (mục 18 tài liệu).

    Args:
        model: Model PAD đã huấn luyện (sẽ chuyển sang mode eval()).
        dataloader: DataLoader tập test với shuffle=False (bắt buộc để tất định).
        device: Thiết bị chạy ("cpu", "cuda") hoặc torch.device.
        threshold: Ngưỡng quyết định; probability >= threshold -> spoof (1).

    Returns:
        dict gồm "metrics", "predictions", "probabilities", "labels",
        "paths", "subject_ids", "attack_types" (các danh sách cùng độ dài).

    Raises:
        ValueError: nếu dataloader rỗng.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    labels: list[int] = []
    probabilities: list[float] = []
    paths: list[str] = []
    subject_ids: list[str] = []
    attack_types: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            batch_labels = _labels_to_list(batch["label"])

            # 1 logit mỗi ảnh -> xác suất spoof qua sigmoid (mục 14 tài liệu).
            logits = model(images)
            probs = torch.sigmoid(logits).reshape(-1).cpu().tolist()

            labels.extend(batch_labels)
            probabilities.extend(float(p) for p in probs)
            paths.extend(str(p) for p in batch["path"])
            subject_ids.extend(str(s) for s in batch.get("subject_id", [""] * len(batch_labels)))
            attack_types.extend(str(a) for a in batch.get("attack_type", [""] * len(batch_labels)))

    if not labels:
        raise ValueError("dataloader is empty: cannot evaluate with 0 samples")

    predictions = [1 if p >= threshold else 0 for p in probabilities]

    # Metric phân loại chuẩn + metric PAD ở ngưỡng cố định.
    metrics = classification_metrics(labels, probabilities, threshold=threshold)
    metrics["apcer"] = apcer(labels, predictions)
    metrics["bpcer"] = bpcer(labels, predictions)
    metrics["acer"] = acer(labels, predictions)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
        "labels": labels,
        "paths": paths,
        "subject_ids": subject_ids,
        "attack_types": attack_types,
    }


def save_predictions(result: dict, path: str) -> None:
    """Lưu dự đoán thô ra CSV phục vụ phân tích lỗi (mục 18, 39 tài liệu).

    Cột CSV:
        path, subject_id, attack_type, label, probability_spoof,
        prediction, correct

    Args:
        result: Kết quả trả về từ evaluate_model.
        path: Đường dẫn tệp CSV đầu ra (thư mục cha sẽ được tạo tự động).
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    rows = []
    for label, prob, pred, image_path, subject_id, attack_type in zip(
        result["labels"],
        result["probabilities"],
        result["predictions"],
        result["paths"],
        result["subject_ids"],
        result["attack_types"],
    ):
        rows.append(
            {
                "path": image_path,
                "subject_id": subject_id,
                "attack_type": attack_type,
                "label": label,
                "probability_spoof": prob,
                "prediction": pred,
                "correct": int(pred == label),
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def _labels_to_list(labels) -> list[int]:
    """Chuyển nhãn từ batch (Tensor/List) thành danh sách int."""
    if isinstance(labels, torch.Tensor):
        labels = labels.reshape(-1).cpu().tolist()
    return [int(label) for label in labels]
