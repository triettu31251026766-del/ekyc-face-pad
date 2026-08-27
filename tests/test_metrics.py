"""tests/test_metrics.py — kiểm thử (unit test) cho module src/metrics.py.

Tệp này dùng để (theo mục 17, 34 của tài liệu kỹ thuật):
- Kiểm tra metric với các trường hợp nhỏ tính tay được:
  dự đoán đúng hết, sai hết, chỉ có bona_fide, chỉ có spoof, mất cân bằng.
- Kiểm tra quy ước nhãn: positive = spoof (1), negative = bona_fide (0),
  từ đó APCER = spoof đoán nhầm thành bona_fide, BPCER = bona_fide đoán
  nhầm thành spoof.
- Kiểm tra báo lỗi với nhãn/xác suất không hợp lệ.

Chạy kiểm thử:
    python -m pytest tests/test_metrics.py
"""

import pytest

from src.metrics import acer, apcer, bpcer, classification_metrics


def test_all_predictions_correct():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.9, 0.8]
    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert apcer(y_true, [0, 0, 1, 1]) == 0.0
    assert bpcer(y_true, [0, 0, 1, 1]) == 0.0
    assert acer(y_true, [0, 0, 1, 1]) == 0.0


def test_all_predictions_wrong():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    metrics = classification_metrics(y_true, [0.9, 0.8, 0.1, 0.2], threshold=0.5)
    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["roc_auc"] == 0.0
    # Cả 2 mẫu spoof đều bị đoán là bona_fide -> APCER = 1.
    assert apcer(y_true, y_pred) == 1.0
    # Cả 2 mẫu bona_fide đều bị đoán là spoof -> BPCER = 1.
    assert bpcer(y_true, y_pred) == 1.0
    assert acer(y_true, y_pred) == 1.0


def test_only_bona_fide():
    y_true = [0, 0]
    y_pred = [0, 0]
    metrics = classification_metrics(y_true, [0.1, 0.2], threshold=0.5)
    assert metrics["accuracy"] == 1.0
    # Chỉ có một lớp -> không tính được AUC.
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    # Không có mẫu spoof -> APCER và ACER không xác định.
    assert apcer(y_true, y_pred) is None
    assert bpcer(y_true, y_pred) == 0.0
    assert acer(y_true, y_pred) is None


def test_only_spoof():
    y_true = [1, 1]
    y_pred = [1, 1]
    metrics = classification_metrics(y_true, [0.9, 0.8], threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] is None
    assert apcer(y_true, y_pred) == 0.0
    assert bpcer(y_true, y_pred) is None
    assert acer(y_true, y_pred) is None


def test_imbalanced_labels():
    # 9 bona_fide + 1 spoof: 1 bona_fide đoán nhầm thành spoof, spoof đoán đúng.
    y_true = [0] * 9 + [1]
    y_pred = [0] * 8 + [1] + [1]  # vị trí 9 (bona_fide) bị đoán nhầm là spoof
    assert apcer(y_true, y_pred) == 0.0
    assert bpcer(y_true, y_pred) == pytest.approx(1 / 9)
    assert acer(y_true, y_pred) == pytest.approx((0.0 + 1 / 9) / 2)

    y_prob = [0.1] * 8 + [0.9, 0.8]
    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] == 0.9
    assert metrics["precision"] == 0.5  # 2 dự đoán spoof, 1 đúng
    assert metrics["recall"] == 1.0  # bắt đúng mẫu spoof duy nhất
    assert metrics["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    # AUC: 8 cặp đúng / 9 cặp (bona_fide prob 0.9 xếp trên spoof prob 0.8).
    assert metrics["roc_auc"] == pytest.approx(8 / 9)


def test_threshold_boundary():
    # y_prob >= threshold mới là spoof (mục 17 tài liệu).
    y_true = [0, 1]
    assert classification_metrics(y_true, [0.5, 0.5], threshold=0.5)["accuracy"] == 0.5
    assert classification_metrics(y_true, [0.49, 0.49], threshold=0.5)["accuracy"] == 0.5


def test_partial_errors():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 1, 1]
    # Spoof thứ 2 đoán nhầm -> APCER = 1/2; bona_fide thứ 2 đoán nhầm -> BPCER = 1/2.
    assert apcer(y_true, y_pred) == 0.5
    assert bpcer(y_true, y_pred) == 0.5
    assert acer(y_true, y_pred) == 0.5


def test_invalid_labels_raise():
    with pytest.raises(ValueError, match="Labels"):
        classification_metrics([0, 2, 1], [0.1, 0.2, 0.9])
    with pytest.raises(ValueError, match="Labels"):
        apcer([0, -1], [0, 1])


def test_invalid_probabilities_raise():
    with pytest.raises(ValueError, match="y_prob"):
        classification_metrics([0, 1], [0.1, 1.5])
    with pytest.raises(ValueError, match="NaN"):
        classification_metrics([0, 1], [0.1, float("nan")])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        classification_metrics([0, 1], [0.5])
    with pytest.raises(ValueError, match="length"):
        apcer([0, 1], [0])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        classification_metrics([], [])
