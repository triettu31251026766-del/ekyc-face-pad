"""tests/test_train_robust.py — kiểm thử smoke cho thí nghiệm E07 (train_robust).

Tệp này dùng để (theo mục 24, 31 của tài liệu kỹ thuật):
- Chạy train_robust trên dataset tổng hợp với khối robustness, kiểm tra
  record training_mode="robust" và đủ tệp đầu ra (checkpoint/JSON/CSV).
- Kiểm tra so sánh công bằng (mục 24): cùng seed/split/model/epochs — robust
  dùng lại đúng splits của baseline (không tạo split mới).
- Kiểm tra tái lập: 2 lần chạy cùng config cho metric bằng nhau.
- Kiểm tra lỗi rõ khi thiếu khối robustness.

Chạy kiểm thử:
    python -m pytest tests/test_train_robust.py
"""

import json

import pytest
from PIL import Image

from experiments.train_baseline import run as run_baseline
from experiments.train_robust import run as run_robust


def _make_dataset(root):
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


def _base_config(dataset_root):
    return {
        "seed": 42,
        "dataset": {"name": "celeba_spoof", "root": str(dataset_root)},
        "split": {"strategy": "subject_disjoint"},
        "model": {"name": "custom_cnn", "image_size": 32},
        "training": {"epochs": 1, "batch_size": 8,
                     "learning_rate": 0.001, "weight_decay": 0.00001},
        "loss": {"name": "bce_with_logits", "use_pos_weight": False},
        "evaluation": {"threshold": 0.5},
        "device": {"name": "cpu"},
    }


def _robustness_config(probability=1.0):
    return {
        "seed": 42,
        "robustness": {
            "enabled": True,
            "augmentations": {
                "jpeg": {"enabled": True, "quality_range": [30, 60], "probability": probability},
                "brightness": {"enabled": True, "factor_range": [0.7, 1.3], "probability": probability},
            },
        },
    }


def test_train_robust_end_to_end(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)

    record = run_robust(
        _base_config(dataset_root),
        _robustness_config(),
        splits_dir=tmp_path / "splits",
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    assert record["experiment_id"] == "E07_robust_seed42"
    assert record["training_mode"] == "robust"
    assert record["degradation_name"] == "none"  # đánh giá trên test sạch (mục 24)
    assert 0.0 <= record["f1"] <= 1.0
    assert len(record["train_history"]) == 1

    # Đủ tệp đầu ra.
    assert (tmp_path / "checkpoints" / "E07_robust_seed42.pt").is_file()
    assert (tmp_path / "results" / "E07_robust_seed42.json").is_file()
    assert (tmp_path / "results" / "E07_robust_seed42.csv").is_file()

    # JSON khớp record.
    with (tmp_path / "results" / "E07_robust_seed42.json").open("r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["training_mode"] == "robust"
    assert saved["f1"] == record["f1"]


def test_train_robust_reuses_baseline_splits(tmp_path):
    """Mục 24: robust phải dùng ĐÚNG splits đã lưu của baseline, không tạo mới."""
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)

    baseline_record = run_baseline(
        _base_config(dataset_root),
        splits_dir=tmp_path / "splits",
        results_dir=tmp_path / "results_base",
        checkpoints_dir=tmp_path / "checkpoints_base",
    )
    splits_file = tmp_path / "splits" / "celeba_spoof_seed42_subject_disjoint.json"
    splits_content = splits_file.read_text(encoding="utf-8")

    robust_record = run_robust(
        _base_config(dataset_root),
        _robustness_config(),
        splits_dir=tmp_path / "splits",
        results_dir=tmp_path / "results_robust",
        checkpoints_dir=tmp_path / "checkpoints_robust",
    )

    # Tệp splits KHÔNG bị ghi đè -> robust đánh giá cùng test set với baseline.
    assert splits_file.read_text(encoding="utf-8") == splits_content
    assert robust_record["seed"] == baseline_record["seed"]
    assert robust_record["split_strategy"] == baseline_record["split_strategy"]
    assert robust_record["model"] == baseline_record["model"]


def test_train_robust_reproducible(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)

    first = run_robust(
        _base_config(dataset_root), _robustness_config(),
        splits_dir=tmp_path / "splits",
        results_dir=tmp_path / "results_a",
        checkpoints_dir=tmp_path / "checkpoints_a",
    )
    second = run_robust(
        _base_config(dataset_root), _robustness_config(),
        splits_dir=tmp_path / "splits",
        results_dir=tmp_path / "results_b",
        checkpoints_dir=tmp_path / "checkpoints_b",
    )
    assert second["f1"] == pytest.approx(first["f1"], abs=1e-6)
    assert second["roc_auc"] == pytest.approx(first["roc_auc"], abs=1e-6)


def test_train_robust_missing_robustness_raises(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    with pytest.raises(ValueError, match="robustness"):
        run_robust(_base_config(dataset_root), robustness_config=None,
                   splits_dir=tmp_path / "splits",
                   results_dir=tmp_path / "results",
                   checkpoints_dir=tmp_path / "checkpoints")
