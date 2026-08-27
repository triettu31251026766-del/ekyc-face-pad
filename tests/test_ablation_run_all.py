"""tests/test_ablation_run_all.py — kiểm thử smoke cho ablation.py và run_all.py.

Tệp này dùng để (theo mục 31, 32 của tài liệu kỹ thuật):
- ablation: bảng ablation đúng cột/dòng (baseline + từng biến thể + robust
  đầy đủ), figure fig_ablation.png được tạo từ dữ liệu thật.
- run_all: chạy đúng thứ tự E01 -> suy giảm -> E07 -> suy giảm robust ->
  compare, dừng đúng thứ tự và đủ tệp đầu ra.

Dữ liệu test là dataset tổng hợp + 1 epoch custom_cnn (như các smoke test khác).

Chạy kiểm thử:
    python -m pytest tests/test_ablation_run_all.py
"""

import pandas as pd
import pytest
from PIL import Image

from experiments.ablation import ablation_table, plot_ablation, run as run_ablation
from experiments.run_all import run as run_all
from experiments.train_baseline import run as run_baseline


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


def _robustness_config():
    return {
        "seed": 42,
        "robustness": {
            "enabled": True,
            "augmentations": {
                "jpeg": {"enabled": True, "quality_range": [30, 60]},
                "brightness": {"enabled": True, "factor_range": [0.7, 1.3]},
            },
        },
    }


# --- Kiểm thử ablation ---


def _sample_records():
    """Record mẫu cho test bảng/figure ablation (không cần huấn luyện)."""
    def record(eid, f1, acer):
        return {"experiment_id": eid, "training_mode": "robust" if "robust" in eid or "ablation" in eid else "clean",
                "degradation_name": "none", "f1": f1, "roc_auc": 0.9,
                "apcer": 0.1, "bpcer": 0.1, "acer": acer}

    return [
        record("E01_baseline_seed42", 0.80, 0.20),
        record("E07_robust_seed42", 0.87, 0.12),
        record("E09_ablation_jpeg_seed42", 0.84, 0.14),
        record("E09_ablation_brightness_seed42", 0.82, 0.17),
    ]


def test_ablation_table_rows_and_columns():
    table = ablation_table(_sample_records())
    assert list(table.columns) == ["experiment_id", "variant", "f1",
                                   "roc_auc", "apcer", "bpcer", "acer"]
    variants = set(table["variant"])
    assert variants == {"baseline", "all", "jpeg", "brightness"}
    assert len(table) == 4


def test_ablation_table_skips_degradation_records():
    records = _sample_records() + [
        {"experiment_id": "E02_jpeg70_seed42", "training_mode": "clean",
         "degradation_name": "jpeg", "degradation_parameters": {"quality": 70},
         "f1": 0.6, "roc_auc": 0.8, "apcer": 0.3, "bpcer": 0.3, "acer": 0.3},
    ]
    table = ablation_table(records)
    assert len(table) == 4  # record suy giảm bị loại khỏi bảng ablation


def test_plot_ablation_creates_png(tmp_path):
    out = tmp_path / "fig_ablation.png"
    assert plot_ablation(_sample_records(), out) is True
    assert out.is_file() and out.stat().st_size > 0


def test_ablation_end_to_end_single_variant(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)

    # Cần baseline trước để bảng ablation có dòng "baseline".
    run_baseline(_base_config(dataset_root),
                 splits_dir=tmp_path / "splits",
                 results_dir=tmp_path / "results",
                 checkpoints_dir=tmp_path / "checkpoints")

    result = run_ablation(
        _base_config(dataset_root), _robustness_config(),
        variants=("jpeg",),
        splits_dir=tmp_path / "splits",
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
    )
    assert result["variants"] == 1
    table = pd.read_csv(tmp_path / "tables" / "ablation_table.csv")
    assert set(table["variant"]) == {"baseline", "jpeg"}
    assert (tmp_path / "figures" / "fig_ablation.png").is_file()
    # Checkpoint của biến thể ablation được lưu.
    assert (tmp_path / "checkpoints" / "E09_ablation_jpeg_seed42.pt").is_file()


# --- Kiểm thử run_all ---


def test_run_all_end_to_end_order(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)

    degradation_configs = [
        {"seed": 42, "degradation": {"name": "jpeg", "quality": 70}},
        {"seed": 42, "degradation": {"name": "blur", "kernel_size": 3, "sigma": 1.0}},
    ]

    summary = run_all(
        _base_config(dataset_root),
        _robustness_config(),
        degradation_configs,
        splits_dir=tmp_path / "splits",
        results_dir=tmp_path / "results",
        checkpoints_dir=tmp_path / "checkpoints",
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        include_ablation=False,
    )

    # Thứ tự đúng mục 32: E01 -> E02/E03 (baseline + suy giảm) -> E07 -> E08/E09.
    assert summary["experiment_ids"] == [
        "E01_baseline_seed42",
        "E02_jpeg70_seed42",
        "E03_blur3_seed42",
        "E07_robust_seed42",
        "E08_robust_jpeg70_seed42",
        "E09_robust_blur3_seed42",
    ]

    # Đủ checkpoint và bảng so sánh cuối cùng.
    assert (tmp_path / "checkpoints" / "E01_baseline_seed42.pt").is_file()
    assert (tmp_path / "checkpoints" / "E07_robust_seed42.pt").is_file()
    assert (tmp_path / "tables" / "comparison_table.csv").is_file()
    table = pd.read_csv(tmp_path / "tables" / "comparison_table.csv")
    assert len(table) == 6

    # Các thí nghiệm đánh giá suy giảm ghi đúng degradation_name.
    names = set(table["degradation_name"])
    assert names == {"none", "jpeg", "blur"}
