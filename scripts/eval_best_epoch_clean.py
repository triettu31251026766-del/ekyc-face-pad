"""scripts/eval_best_epoch_clean.py — đánh giá checkpoint BEST-EPOCH trên test sạch.

Tệp này dùng để:
- Sinh bộ metric ĐẦY ĐỦ (accuracy, precision, recall, f1, roc_auc, pr_auc,
  apcer, bpcer, acer) + predictions CSV cho checkpoint BEST-EPOCH (epoch 18)
  của E01 baseline và E07 robust trên tập test SẠCH (40.643 ảnh).
- KHÔNG ghi đè file kết quả gốc (epoch 20): đầu ra ghi với experiment_id mới
  `E01_baseline_seed123_bestepoch` / `E07_robust_seed123_bestepoch`.
- In thêm confusion matrix tính lại từ predictions CSV để dùng viết báo cáo.

Lý do: lần chạy lưới suy giảm chỉ lưu metric (hàng clean của
results/tables/degradation_*.csv) mà KHÔNG lưu predictions, nên không có
precision/recall/confusion matrix/ROC của checkpoint best-epoch. File này
lấp chỗ trống đó (dùng đúng logic của experiments/eval_clean.py).

KHÔNG huấn luyện lại — chỉ load checkpoint và đánh giá (mục 23 tài liệu).

Cách dùng (chạy từ thư mục gốc dự án):
    python -m scripts.eval_best_epoch_clean
"""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.metrics import confusion_matrix

from experiments.eval_clean import run as run_clean_eval

# (checkpoint, experiment_id mới) — giữ nguyên checkpoint best-epoch đã huấn luyện.
JOBS = [
    ("results/checkpoints/E01_baseline_seed123.pt", "E01_baseline_seed123_bestepoch"),
    ("results/checkpoints/E07_robust_seed123.pt", "E07_robust_seed123_bestepoch"),
]

# Giá trị MONG ĐỢI (hàng clean của bảng lưới suy giảm) để đối chiếu nhanh.
EXPECTED = {
    "E01_baseline_seed123_bestepoch": {"f1": 0.986776, "roc_auc": 0.997767, "acer": 0.021111},
    "E07_robust_seed123_bestepoch": {"f1": 0.983057, "roc_auc": 0.996944, "acer": 0.025553},
}

RESULTS_DIR = Path("results/raw")


def _print_confusion_matrix(predictions_path: Path) -> None:
    """Đọc predictions CSV và in confusion matrix (positive = spoof)."""
    with predictions_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [int(r["label"]) for r in rows]
    preds = [int(r["prediction"]) for r in rows]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    print(f"    confusion matrix (positive=spoof): TN={tn} FP={fp} FN={fn} TP={tp} "
          f"(total {tn + fp + fn + tp})")


def main() -> None:
    """Chạy đánh giá clean cho cả 2 checkpoint best-epoch và in kết quả."""
    for checkpoint_path, experiment_id in JOBS:
        config = {
            "seed": 123,
            "experiment_id": experiment_id,
            "device": {"name": "auto"},
        }
        print(f"\n=== {experiment_id} | checkpoint: {checkpoint_path} ===")
        record = run_clean_eval(config, checkpoint_path)

        metrics = {key: record[key] for key in (
            "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "apcer", "bpcer", "acer",
        )}
        for key, value in metrics.items():
            print(f"    {key:10s} = {value:.6f}")

        expected = EXPECTED[experiment_id]
        ok = (abs(metrics["f1"] - expected["f1"]) < 1e-4
              and abs(metrics["roc_auc"] - expected["roc_auc"]) < 1e-4
              and abs(metrics["acer"] - expected["acer"]) < 1e-4)
        print(f"    ĐỐI CHIẾU grid clean row: {'KHỚP' if ok else 'LỆCH'} "
              f"(f1={expected['f1']:.6f}, roc_auc={expected['roc_auc']:.6f}, "
              f"acer={expected['acer']:.6f})")

        _print_confusion_matrix(RESULTS_DIR / f"{experiment_id}_predictions.csv")
        print(f"    đã lưu: {RESULTS_DIR / (experiment_id + '.json')}, "
              f"{RESULTS_DIR / (experiment_id + '_predictions.csv')}")


if __name__ == "__main__":
    main()
