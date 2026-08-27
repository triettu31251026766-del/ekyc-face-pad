"""tests/test_train_baseline.py — kiểm thử smoke end-to-end cho thí nghiệm E01.

Tệp này dùng để (theo mục 22, 31, 33 của tài liệu kỹ thuật):
- Chạy TOÀN BỘ pipeline E01 (train_baseline) trên dataset tổng hợp nhỏ,
  từ config đến lưu kết quả — đúng cột mốc "Minimum First Milestone" mục 33:
      Dataset -> split -> model -> huấn luyện -> clean test -> F1 + ROC-AUC
      -> checkpoint + metrics JSON + metrics CSV.
- Kiểm tra đầy đủ tệp đầu ra: splits, checkpoint, JSON, CSV, predictions CSV.
- Kiểm tra record kết quả có đủ các trường theo mục 29.
- Kiểm tra thí nghiệm tái lập được: chạy 2 lần cùng config cho metric bằng nhau.

Chạy kiểm thử:
    python -m pytest tests/test_train_baseline.py
"""

import json

import pandas as pd
import pytest
from PIL import Image

from experiments.train_baseline import run

EXPECTED_RECORD_FIELDS = {
    "experiment_id", "seed", "dataset", "split_strategy", "model",
    "training_mode", "degradation_name", "degradation_parameters", "threshold",
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "apcer", "bpcer", "acer", "runtime_seconds", "parameter_count",
    "model_size_mb", "environment", "train_history",
}


def _make_synthetic_dataset(root):
    """Dựng dataset giả lập CelebA-Spoof: 9 subject x 4 ảnh (36 mẫu, 2 lớp)."""
    image_root = root / "SpoofingData"
    image_root.mkdir(parents=True, exist_ok=True)
    lines = []
    for subject in range(1001, 1010):
        for frame, raw_label in enumerate([0, 1, 0, 2]):
            name = f"{subject}_{frame}.jpg"
            Image.new("RGB", (32, 32), color=(20 if raw_label == 0 else 230, 90, 90)).save(
                image_root / name
            )
            lines.append(f"{name} {raw_label} 0 0")
    (root / "train_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config(dataset_root, experiment_id):
    """Cấu hình E01 hợp lệ, model nhỏ + 1 epoch để chạy test nhanh."""
    return {
        "seed": 42,
        "experiment_id": experiment_id,
        "dataset": {"name": "celeba_spoof", "root": str(dataset_root)},
        "split": {"strategy": "subject_disjoint"},
        "model": {"name": "custom_cnn", "image_size": 32},
        "training": {"epochs": 1, "batch_size": 8,
                     "learning_rate": 0.001, "weight_decay": 0.00001},
        "loss": {"name": "bce_with_logits", "use_pos_weight": False},
        "evaluation": {"threshold": 0.5},
        "device": {"name": "cpu"},
    }


def _run_once(tmp_path, experiment_id):
    root = tmp_path / "dataset"
    _make_synthetic_dataset(root)
    record = run(
        _config(root, experiment_id),
        splits_dir=tmp_path / "splits",
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    return record


def test_e01_end_to_end_outputs(tmp_path):
    record = _run_once(tmp_path, "E_test")

    # 1) Tệp splits được tạo (mục 7).
    splits_file = tmp_path / "splits" / "celeba_spoof_seed42_subject_disjoint.json"
    assert splits_file.is_file()

    # 2) Checkpoint được lưu (mục 38) với đủ nội dung.
    checkpoint = tmp_path / "checkpoints" / "E_test.pt"
    assert checkpoint.is_file()
    import torch

    ckpt = torch.load(checkpoint, weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "epoch", "config", "seed"):
        assert key in ckpt

    # 3) JSON + CSV + predictions CSV (mục 22, 30, 39).
    assert (tmp_path / "results" / "E_test.json").is_file()
    assert (tmp_path / "results" / "E_test.csv").is_file()
    assert (tmp_path / "results" / "E_test_predictions.csv").is_file()

    # 4) Record đủ trường theo mục 29.
    assert EXPECTED_RECORD_FIELDS <= set(record.keys())
    assert record["experiment_id"] == "E_test"
    assert record["training_mode"] == "clean"
    assert record["degradation_name"] == "none"
    assert len(record["train_history"]) == 1
    assert 0.0 <= record["f1"] <= 1.0
    assert record["roc_auc"] is not None

    # 5) JSON lưu lại phải khớp record.
    with (tmp_path / "results" / "E_test.json").open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["f1"] == record["f1"]
    assert saved["environment"]["git_commit"]

    # 6) CSV 1 dòng có cột f1.
    frame = pd.read_csv(tmp_path / "results" / "E_test.csv")
    assert len(frame) == 1
    assert "f1" in frame.columns

    # 7) Predictions CSV có đúng số mẫu test theo splits đã lưu.
    from src.data import load_splits

    splits = load_splits(splits_file)
    preds = pd.read_csv(tmp_path / "results" / "E_test_predictions.csv")
    assert len(preds) == len(splits["test"])
    assert len(preds) > 0


def test_e01_reproducible_metric(tmp_path):
    """Chạy 2 lần cùng config (seed 42) -> metric gần như trùng khớp (mục 36)."""
    first = _run_once(tmp_path, "E_rep_a")
    second = _run_once(tmp_path, "E_rep_b")
    assert second["f1"] == pytest.approx(first["f1"], abs=1e-6)
    assert second["roc_auc"] == pytest.approx(first["roc_auc"], abs=1e-6)
    assert second["acer"] == pytest.approx(first["acer"], abs=1e-6)
