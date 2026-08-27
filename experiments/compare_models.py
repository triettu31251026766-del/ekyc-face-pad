"""experiments/compare_models.py — so sánh baseline vs robust từ file kết quả.

Tệp này dùng để (theo mục 31, 40, 41 của tài liệu kỹ thuật):
Pipeline:
    load saved result files -> align experiments -> create comparison table
    -> save CSV -> generate figures

QUY TẮC (Prompt 10, mục 40, 41):
- KHÔNG huấn luyện lại model; KHÔNG nhập tay số liệu — mọi bảng và hình
  phải được sinh từ các tệp kết quả đã lưu (mục 40: "Do not manually type
  numbers into plots").
- So sánh baseline vs robustness phải giữ cố định dataset/split/test set/
  model/threshold (đã được đảm bảo ở bước chạy thí nghiệm — mục 24, 41).

Các tệp đầu ra:
    results/tables/comparison_table.csv
    results/figures/fig_f1_baseline_vs_robust.png
    results/figures/fig_acer_vs_jpeg_quality.png
    results/figures/fig_f1_vs_resolution.png
    results/figures/fig_apcer_bpcer_comparison.png

Chạy:
    python experiments/compare_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Agg backend: chạy được trên máy không có màn hình (server/CI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import get_experiment_logger

# Các cột so sánh trong bảng (mục 29 tài liệu).
TABLE_COLUMNS = [
    "experiment_id", "training_mode", "degradation_name",
    "degradation_parameters", "threshold", "accuracy", "precision",
    "recall", "f1", "roc_auc", "pr_auc", "apcer", "bpcer", "acer",
    "runtime_seconds",
]

logger = get_experiment_logger("compare_models")


def load_results(results_dir: str | Path = "results/raw") -> list[dict]:
    """Đọc toàn bộ record kết quả (*.json) trong thư mục, sắp xếp theo id.

    Bỏ qua các tệp JSON không phải record kết quả (thiếu "experiment_id").
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        logger.warning(f"thư mục kết quả không tồn tại: {results_dir}")
        return []

    records = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"bỏ qua tệp JSON lỗi {path}: {exc}")
            continue
        if isinstance(data, dict) and "experiment_id" in data and "f1" in data:
            records.append(data)

    records.sort(key=lambda record: record["experiment_id"])
    return records


def comparison_table(records: list[dict]) -> pd.DataFrame:
    """Dựng bảng so sánh từ danh sách record (mỗi record là 1 dòng)."""
    rows = []
    for record in records:
        row = {}
        for column in TABLE_COLUMNS:
            value = record.get(column)
            if isinstance(value, dict):
                # degradation_parameters là dict -> ghi dạng chuỗi ngắn gọn.
                value = json.dumps(value, ensure_ascii=False)
            row[column] = value
        rows.append(row)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def plot_baseline_vs_robust(records: list[dict], out_path: str | Path) -> bool:
    """Vẽ fig_f1_baseline_vs_robust.png: F1 trên test SẠCH của 2 chế độ huấn luyện.

    Returns:
        True nếu đã vẽ (có đủ dữ liệu), False nếu không đủ dữ liệu để vẽ.
    """
    frame = comparison_table(records)
    # Chỉ so sánh các thí nghiệm đánh giá trên test sạch (degradation "none").
    clean = frame[frame["degradation_name"] == "none"]
    if clean.empty or set(clean["training_mode"]) != {"clean", "robust"}:
        return False

    clean = clean.set_index("training_mode").sort_index()
    metrics = ["f1", "roc_auc", "acer"]
    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    for offset, mode in enumerate(("clean", "robust")):
        values = [clean.loc[mode, metric] for metric in metrics]
        plt.bar(
            [pos + offset * width for pos in x], values, width,
            label=f"{mode} (train)",
        )
    plt.xticks([pos + width / 2 for pos in x], metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("giá trị metric (0-1)")
    plt.title("Baseline vs Robust — đánh giá trên test SẠCH (mục 24)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def plot_degradation_curve(
    records: list[dict],
    out_path: str | Path,
    *,
    degradation_name: str,
    parameter_name: str,
    metric: str,
) -> bool:
    """Vẽ đường metric theo mức độ suy giảm (ví dụ ACER theo quality JPEG).

    Args:
        degradation_name: "jpeg", "resize", "blur", "noise", "brightness".
        parameter_name: tên tham số suy giảm ("quality", "scale", ...).
        metric: tên metric vẽ ("acer", "f1", ...).
        out_path: đường dẫn tệp PNG.

    Returns:
        True nếu đã vẽ (có >= 2 điểm), False nếu không đủ dữ liệu.
    """
    frame = comparison_table(records)
    mask = frame["degradation_name"] == degradation_name
    selected = frame[mask].copy()
    if selected.empty:
        return False

    def extract(value):
        try:
            return float(json.loads(value)[parameter_name])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    selected["severity"] = selected["degradation_parameters"].apply(extract)
    selected = selected.dropna(subset=["severity", metric]).sort_values("severity")
    if len(selected) < 2:
        return False

    plt.figure(figsize=(8, 5))
    for mode, group in selected.groupby("training_mode"):
        plt.plot(group["severity"], group[metric], marker="o", label=f"{mode} (train)")
    plt.xlabel(parameter_name)
    plt.ylabel(metric)
    plt.title(f"{metric.upper()} theo {degradation_name} ({parameter_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def plot_apcer_bpcer(records: list[dict], out_path: str | Path) -> bool:
    """Vẽ fig_apcer_bpcer_comparison.png: APCER vs BPCER từng thí nghiệm."""
    frame = comparison_table(records)
    if frame.empty:
        return False

    labels = frame["experiment_id"].tolist()
    x = range(len(frame))
    width = 0.35

    plt.figure(figsize=(max(8, len(frame) * 0.8), 5))
    plt.bar([pos - width / 2 for pos in x], frame["apcer"], width, label="APCER (spoof bị bỏ sót)")
    plt.bar([pos + width / 2 for pos in x], frame["bpcer"], width, label="BPCER (cảnh báo giả)")
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("tỉ lệ lỗi (0-1)")
    plt.title("APCER vs BPCER theo thí nghiệm (mục 17)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def run(
    results_dir: str | Path = "results/raw",
    tables_dir: str | Path = "results/tables",
    figures_dir: str | Path = "results/figures",
) -> dict:
    """Chạy so sánh đầy đủ: bảng CSV + các figure (mục 40). Không huấn luyện lại."""
    records = load_results(results_dir)
    logger.info(f"đã đọc {len(records)} record từ {results_dir}")

    if not records:
        logger.warning("chưa có record nào để so sánh — hãy chạy thí nghiệm trước")
        return {"records": 0, "table": None, "figures": []}

    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Bảng so sánh CSV.
    table = comparison_table(records)
    table_path = tables_dir / "comparison_table.csv"
    table.to_csv(table_path, index=False)
    logger.info(f"bảng so sánh: {table_path} ({len(table)} dòng)")

    # 2) Các figure (mục 40) — mỗi hình tự kiểm tra dữ liệu, không bịa số liệu.
    figure_specs = [
        (plot_baseline_vs_robust, records, figures_dir / "fig_f1_baseline_vs_robust.png"),
        (plot_degradation_curve, records, figures_dir / "fig_acer_vs_jpeg_quality.png",
         {"degradation_name": "jpeg", "parameter_name": "quality", "metric": "acer"}),
        (plot_degradation_curve, records, figures_dir / "fig_f1_vs_resolution.png",
         {"degradation_name": "resize", "parameter_name": "scale", "metric": "f1"}),
        (plot_apcer_bpcer, records, figures_dir / "fig_apcer_bpcer_comparison.png"),
    ]

    created = []
    for spec in figure_specs:
        plotter = spec[0]
        args = spec[1]
        out_path = spec[2]
        kwargs = spec[3] if len(spec) > 3 else {}
        try:
            drawn = plotter(args, out_path, **kwargs)
        except Exception as exc:  # lỗi vẽ không được chặn cả pipeline
            logger.error(f"lỗi khi vẽ {out_path}: {exc}")
            continue
        if drawn:
            created.append(str(out_path))
            logger.info(f"đã tạo figure: {out_path}")
        else:
            logger.info(f"bỏ qua figure {out_path.name} (chưa đủ dữ liệu)")

    return {"records": len(records), "table": str(table_path), "figures": created}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
