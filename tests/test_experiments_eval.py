"""tests/test_experiments_eval.py — kiểm thử smoke cho eval_clean và eval_degradation.

Tệp này dùng để (theo mục 23, 31 của tài liệu kỹ thuật):
- Chạy E01 (train_baseline) trên dataset tổng hợp để có checkpoint thật.
- eval_clean: đánh giá lại checkpoint trên test SẠCH -> metric khớp với
  kết quả E01 (cùng model, cùng test set, cùng ngưỡng — mục 24).
- eval_degradation: đánh giá với suy giảm JPEG/noise TẤT ĐỊNH -> ghi đúng
  degradation_name/parameters, chạy 2 lần cho kết quả giống hệt nhau (mục 13).
- KHÔNG huấn luyện lại trong các script eval (mục 23).

Chạy kiểm thử:
    python -m pytest tests/test_experiments_eval.py
"""

import json

import pandas as pd
import pytest
from PIL import Image

from experiments.eval_clean import run as run_eval_clean
from experiments.eval_degradation import run as run_eval_degradation
from experiments.train_baseline import run as run_baseline


def _make_dataset(root):
    """Dataset tổng hợp 9 subject x 4 ảnh (giống test_train_baseline)."""
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


def _base_config(dataset_root, experiment_id="E01_test"):
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


@pytest.fixture()
def trained(tmp_path):
    """Chạy E01 để sinh checkpoint + splits, trả về các đường dẫn liên quan."""
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    record = run_baseline(
        _base_config(dataset_root),
        splits_dir=tmp_path / "splits",
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    return {
        "tmp_path": tmp_path,
        "checkpoint": tmp_path / "checkpoints" / "E01_test.pt",
        "baseline_f1": record["f1"],
        "config": _base_config(dataset_root),
    }


def test_eval_clean_matches_baseline_metrics(trained):
    checkpoint = trained["checkpoint"]
    config = trained["config"]

    record = run_eval_clean(
        config,
        checkpoint,
        splits_dir=trained["tmp_path"] / "splits",
        results_dir=trained["tmp_path"] / "results_eval",
    )

    # Cùng model + cùng test set + cùng ngưỡng -> metric phải khớp E01 (mục 24).
    assert record["f1"] == pytest.approx(trained["baseline_f1"], abs=1e-6)
    assert record["degradation_name"] == "none"
    assert record["training_mode"] == "clean"
    # Tệp kết quả được lưu.
    assert (trained["tmp_path"] / "results_eval" / f"{record['experiment_id']}.json").is_file()
    assert (trained["tmp_path"] / "results_eval" / f"{record['experiment_id']}.csv").is_file()


def test_eval_degradation_jpeg_deterministic(trained):
    checkpoint = trained["checkpoint"]
    tmp_path = trained["tmp_path"]
    config = {
        "seed": 42,
        "experiment_id": "E02_jpeg30_seed42",
        "degradation": {"name": "jpeg", "quality": 30},
    }

    first = run_eval_degradation(config, checkpoint,
                                 splits_dir=tmp_path / "splits",
                                 results_dir=tmp_path / "results_deg")
    second = run_eval_degradation(config, checkpoint,
                                  splits_dir=tmp_path / "splits",
                                  results_dir=tmp_path / "results_deg")

    assert first["degradation_name"] == "jpeg"
    assert first["degradation_parameters"] == {"quality": 30}
    # Suy giảm tất định -> 2 lần chạy phải cho metric giống hệt nhau (mục 13).
    assert second["f1"] == first["f1"]
    assert second["roc_auc"] == first["roc_auc"]
    assert second["acer"] == first["acer"]
    # Tệp kết quả đúng tên experiment_id.
    assert (tmp_path / "results_deg" / "E02_jpeg30_seed42.json").is_file()
    assert (tmp_path / "results_deg" / "E02_jpeg30_seed42_predictions.csv").is_file()


def test_eval_degradation_noise_uses_seed(trained):
    checkpoint = trained["checkpoint"]
    tmp_path = trained["tmp_path"]
    config = {"seed": 42, "degradation": {"name": "noise", "std": 0.03}}

    first = run_eval_degradation(config, checkpoint,
                                 splits_dir=tmp_path / "splits",
                                 results_dir=tmp_path / "results_noise")
    second = run_eval_degradation(config, checkpoint,
                                  splits_dir=tmp_path / "splits",
                                  results_dir=tmp_path / "results_noise")

    # Noise có seed từ config -> vẫn tất định khi đánh giá (mục 13).
    assert first["degradation_name"] == "noise"
    assert first["degradation_parameters"] == {"std": 0.03}
    assert second["f1"] == first["f1"]

    # CSV có đủ số dòng test (không rỗng).
    preds = pd.read_csv(
        tmp_path / "results_noise" / f"{first['experiment_id']}_predictions.csv"
    )
    assert len(preds) > 0


def test_eval_degradation_missing_splits_raises(trained):
    checkpoint = trained["checkpoint"]
    tmp_path = trained["tmp_path"]
    config = {"seed": 42, "degradation": {"name": "jpeg", "quality": 50}}
    with pytest.raises(FileNotFoundError, match="Splits file not found"):
        run_eval_degradation(config, checkpoint,
                             splits_dir=tmp_path / "khong_ton_tai")


def test_load_checkpoint_missing_file_raises(trained):
    from experiments._common import load_checkpoint

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(trained["tmp_path"] / "khong_co.pt", device="cpu")
