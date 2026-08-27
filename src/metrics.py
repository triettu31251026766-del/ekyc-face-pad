"""src/metrics.py — tập trung toàn bộ metric đánh giá model PAD.

Tệp này dùng để (theo mục 17 của tài liệu kỹ thuật):
- classification_metrics(y_true, y_prob, threshold): trả về
      {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
- apcer(y_true, y_pred), bpcer(y_true, y_pred), acer(y_true, y_pred):
  các metric chuẩn của bài toán PAD (ISO/IEC 30107-3).

QUY ƯỚC NHÃN (phải dùng nhất quán toàn dự án — mục 6, 17 tài liệu):
    positive = spoof      (nhãn 1)
    negative = bona_fide  (nhãn 0)

Từ đó:
    APCER (Attack Presentation Classification Error Rate):
        tỉ lệ mẫu spoof bị dự đoán nhầm là bona_fide (bỏ sót tấn công).
    BPCER (Bona fide Presentation Classification Error Rate):
        tỉ lệ mẫu bona_fide bị dự đoán nhầm là spoof (cảnh báo giả).
    ACER = (APCER + BPCER) / 2

Chú ý: module này KHÔNG huấn luyện, KHÔNG tải model. Chỉ nhận nhãn và
xác suất/dự đoán để tính metric một cách tất định.

Cách dùng:
    from src.metrics import classification_metrics, apcer, bpcer, acer
    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# --- Hằng số quy ước nhãn (giống src/data.py) ---
LABEL_BONA_FIDE = 0
LABEL_SPOOF = 1


def classification_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict:
    """Tính các metric phân loại chuẩn ở ngưỡng cố định (mục 17 tài liệu).

    Args:
        y_true: Nhãn thật, các giá trị 0 (bona_fide) hoặc 1 (spoof).
        y_prob: Xác suất spoof dự đoán, trong khoảng [0, 1].
        threshold: Ngưỡng quyết định; y_prob >= threshold -> dự đoán spoof (1).

    Returns:
        dict gồm: accuracy, precision, recall, f1, roc_auc, pr_auc.
        roc_auc / pr_auc bằng None nếu dữ liệu chỉ có một lớp (không thể
        tính AUC), các metric còn lại luôn có giá trị.

    Raises:
        ValueError: nếu nhãn không thuộc {0, 1}, xác suất ngoài [0, 1] hoặc
            độ dài y_true và y_prob khác nhau.
    """
    y_true = _as_array(y_true, "y_true")
    y_prob = _as_array(y_prob, "y_prob")

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"y_true and y_prob must have the same length, "
            f"got {len(y_true)} and {len(y_prob)}"
        )
    _validate_labels(y_true)
    _validate_probabilities(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=LABEL_SPOOF, zero_division=0
    )

    # AUC chỉ tính được khi có đủ cả hai lớp.
    roc_auc = _safe_auc(y_true, y_prob, roc_auc_score)
    pr_auc = _safe_auc(y_true, y_prob, average_precision_score)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def apcer(y_true, y_pred) -> float | None:
    """APCER: tỉ lệ mẫu spoof bị dự đoán nhầm là bona_fide (bỏ sót tấn công).

    Args:
        y_true: Nhãn thật (0 = bona_fide, 1 = spoof).
        y_pred: Nhãn dự đoán nhị phân (0/1).

    Returns:
        float trong [0, 1]; None nếu không có mẫu spoof nào trong y_true
        (khi đó tỉ lệ này không xác định được).
    """
    y_true, y_pred = _validate_pair(y_true, y_pred)
    mask_spoof = y_true == LABEL_SPOOF
    if mask_spoof.sum() == 0:
        return None
    return float(np.mean(y_pred[mask_spoof] == LABEL_BONA_FIDE))


def bpcer(y_true, y_pred) -> float | None:
    """BPCER: tỉ lệ mẫu bona_fide bị dự đoán nhầm là spoof (cảnh báo giả).

    Returns:
        float trong [0, 1]; None nếu không có mẫu bona_fide nào trong y_true.
    """
    y_true, y_pred = _validate_pair(y_true, y_pred)
    mask_bona = y_true == LABEL_BONA_FIDE
    if mask_bona.sum() == 0:
        return None
    return float(np.mean(y_pred[mask_bona] == LABEL_SPOOF))


def acer(y_true, y_pred) -> float | None:
    """ACER = (APCER + BPCER) / 2 — metric cân bằng tổng của bài toán PAD.

    Returns:
        float trong [0, 1]; None nếu APCER hoặc BPCER không xác định được
        (thiếu một trong hai lớp trong y_true).
    """
    apcer_value = apcer(y_true, y_pred)
    bpcer_value = bpcer(y_true, y_pred)
    if apcer_value is None or bpcer_value is None:
        return None
    return (apcer_value + bpcer_value) / 2.0


# --- Các hàm nội bộ ---


def _as_array(values, name: str) -> np.ndarray:
    """Chuyển đầu vào thành numpy array 1 chiều kiểu float."""
    array = np.asarray(values)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array.astype(float)


def _validate_labels(y_true: np.ndarray) -> None:
    """Nhãn chỉ được phép là 0 hoặc 1 (quy ước mục 6, 17 tài liệu)."""
    unique = set(np.unique(y_true).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(
            f"Labels must be 0 (bona_fide) or 1 (spoof), found {sorted(unique)}"
        )


def _validate_probabilities(y_prob: np.ndarray) -> None:
    """Xác suất phải hữu hạn và nằm trong [0, 1]."""
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("y_prob contains NaN or infinite values")
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError(
            f"y_prob must be in [0, 1], got range "
            f"[{y_prob.min():.4f}, {y_prob.max():.4f}]"
        )


def _validate_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Kiểm tra chung cho các hàm APCER/BPCER/ACER."""
    y_true = _as_array(y_true, "y_true")
    y_pred = _as_array(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )
    _validate_labels(y_true)
    _validate_labels(y_pred)
    return y_true, y_pred


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray, scorer) -> float | None:
    """Tính AUC; trả về None khi dữ liệu chỉ có một lớp duy nhất."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(scorer(y_true, y_prob))
