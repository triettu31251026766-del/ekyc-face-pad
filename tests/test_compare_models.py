"""tests/test_compare_models.py — kiểm thử (unit test) cho experiments/compare_models.py.

Tệp này dùng để (theo mục 31, 40 của tài liệu kỹ thuật):
- Kiểm tra load_results đọc đúng record, bỏ qua JSON không phải kết quả.
- Kiểm tra comparison_table có đúng cột và số dòng.
- Kiểm tra các hàm vẽ tạo ra tệp PNG thật từ dữ liệu (KHÔNG bịa số liệu).
- Kiểm tra run() tạo bảng CSV + figure, và xử lý thư mục rỗng không lỗi.

Ghi chú: dữ liệu test là các record tổng hợp trong tmp_path (chỉ để kiểm
thử logic vẽ/bảng) — KHÔNG ghi số liệu giả vào results/ của repo (Rule 5).

Chạy kiểm thử:
    python -m pytest tests/test_compare_models.py
"""

import json

import pandas as pd
import pytest

from experiments.compare_models import (
    comparison_table,
    load_results,
    plot_apcer_bpcer,
    plot_baseline_vs_robust,
    plot_degradation_curve,
    run,
)


def _write_record(results_dir, experiment_id, training_mode, degradation_name,
                  degradation_parameters=None, f1=0.8, roc_auc=0.9,
                  acer=0.1, apcer=0.1, bpcer=0.1):
    record = {
        "experiment_id": experiment_id,
        "training_mode": training_mode,
        "degradation_name": degradation_name,
        "degradation_parameters": degradation_parameters or {},
        "threshold": 0.5,
        "accuracy": 0.8, "precision": 0.8, "recall": 0.8,
        "f1": f1, "roc_auc": roc_auc, "pr_auc": 0.9,
        "apcer": apcer, "bpcer": bpcer, "acer": acer,
        "runtime_seconds": 1.0,
    }
    with (results_dir / f"{experiment_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle)
    return record


@pytest.fixture()
def records(tmp_path):
    """Tập record tổng hợp: baseline + jpeg 90/70 + robust + robust jpeg70."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_record(results_dir, "E01_baseline_seed42", "clean", "none", f1=0.85, acer=0.12)
    _write_record(results_dir, "E02_jpeg90_seed42", "clean", "jpeg",
                  {"quality": 90}, f1=0.80, acer=0.18)
    _write_record(results_dir, "E03_jpeg70_seed42", "clean", "jpeg",
                  {"quality": 70}, f1=0.70, acer=0.30)
    _write_record(results_dir, "E04_resize75_seed42", "clean", "resize",
                  {"scale": 0.75}, f1=0.78, acer=0.22)
    _write_record(results_dir, "E05_resize50_seed42", "clean", "resize",
                  {"scale": 0.50}, f1=0.68, acer=0.35)
    _write_record(results_dir, "E07_robust_seed42", "robust", "none", f1=0.87, acer=0.10)
    _write_record(results_dir, "E08_robust_jpeg70_seed42", "robust", "jpeg",
                  {"quality": 70}, f1=0.78, acer=0.20)
    # Tệp JSON không phải record -> phải bị bỏ qua.
    (results_dir / "note.json").write_text('{"ghi_chu": "khong phai ket qua"}',
                                           encoding="utf-8")
    return results_dir


def test_load_results_reads_and_skips(records):
    loaded = load_results(records)
    assert len(loaded) == 7
    ids = [record["experiment_id"] for record in loaded]
    assert ids == sorted(ids)


def test_comparison_table_columns_and_rows(records):
    table = comparison_table(load_results(records))
    assert len(table) == 7
    for column in ("experiment_id", "training_mode", "degradation_name",
                   "f1", "roc_auc", "apcer", "bpcer", "acer"):
        assert column in table.columns


def test_plot_baseline_vs_robust_creates_png(records, tmp_path):
    out = tmp_path / "fig_f1_baseline_vs_robust.png"
    assert plot_baseline_vs_robust(load_results(records), out) is True
    assert out.is_file() and out.stat().st_size > 0


def test_plot_baseline_vs_robust_skips_without_both_modes(records, tmp_path):
    # Chỉ còn record clean -> không vẽ được hình so sánh 2 chế độ.
    only_clean = [r for r in load_results(records) if r["training_mode"] == "clean"]
    out = tmp_path / "skip.png"
    assert plot_baseline_vs_robust(only_clean, out) is False
    assert not out.exists()


def test_plot_degradation_curve_creates_png(records, tmp_path):
    out = tmp_path / "fig_acer_vs_jpeg_quality.png"
    drawn = plot_degradation_curve(load_results(records), degradation_name="jpeg",
                                   parameter_name="quality", metric="acer",
                                   out_path=out)
    assert drawn is True
    assert out.is_file() and out.stat().st_size > 0


def test_plot_degradation_curve_skips_with_insufficient_data(records, tmp_path):
    # Chỉ 1 điểm jpeg -> không đủ vẽ đường.
    one = [r for r in load_results(records) if r["degradation_name"] == "jpeg"][:1]
    out = tmp_path / "skip.png"
    assert plot_degradation_curve(one, degradation_name="jpeg",
                                  parameter_name="quality", metric="acer",
                                  out_path=out) is False


def test_plot_apcer_bpcer_creates_png(records, tmp_path):
    out = tmp_path / "fig_apcer_bpcer_comparison.png"
    assert plot_apcer_bpcer(load_results(records), out) is True
    assert out.is_file() and out.stat().st_size > 0


def test_run_creates_table_and_figures(records, tmp_path):
    result = run(results_dir=records,
                 tables_dir=tmp_path / "tables",
                 figures_dir=tmp_path / "figures")
    assert result["records"] == 7
    assert (tmp_path / "tables" / "comparison_table.csv").is_file()
    table = pd.read_csv(tmp_path / "tables" / "comparison_table.csv")
    assert len(table) == 7
    # Phải tạo được đủ 4 hình (dữ liệu test đủ điều kiện).
    assert len(result["figures"]) == 4


def test_run_empty_dir_does_not_crash(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run(results_dir=empty,
                 tables_dir=tmp_path / "tables",
                 figures_dir=tmp_path / "figures")
    assert result["records"] == 0
    assert result["table"] is None
