"""experiments/eval_degradation_grid.py — đánh giá checkpoint trên LƯỚI suy giảm đầy đủ.

Tệp này dùng để (theo kế hoạch thí nghiệm chính, PHASE 2 — baseline degradation
stress test và PHASE 4 — robust degradation test):
- Chạy CÙNG model (checkpoint) trên CÙNG test set cố định với MỌI mức suy giảm:
      jpeg 90/70/50/30
      resize 75/50/25%
      blur light/medium/strong
      noise low/medium/high
      brightness dark/normal/bright
- Mỗi điều kiện: suy giảm TẤT ĐỊNH (mục 13), lưu JSON/CSV/predictions riêng
  với experiment_id theo chuẩn E02-E06 (baseline) / E08-E12 (robust).
- Tổng hợp bảng: Condition|Severity|F1|ROC-AUC|PR-AUC|APCER|BPCER|ACER
  -> results/tables/degradation_<tag>.csv
- Vẽ biểu đồ F1 và ACER theo mức suy giảm từng loại -> results/figures/.

QUAN TRỌNG:
- KHÔNG huấn luyện lại model (mục 23) — chỉ đo phản ứng của model đã có.
- Các mức suy giảm được ĐỊNH NGHĨA TƯỜNG MINH ở SEVERITY_GRID dưới đây và
  được ghi vào từng kết quả (degradation_parameters) để tái lập.

Cách dùng (chạy từ thư mục gốc dự án):
    python -m experiments.eval_degradation_grid --checkpoint results/checkpoints/E01_baseline_seed123.pt --tag baseline
    python -m experiments.eval_degradation_grid --checkpoint results/checkpoints/E07_robust_seed123.pt --tag robust
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from experiments._common import load_checkpoint  # noqa: E402
from experiments.eval_degradation import run as run_degradation  # noqa: E402
from src.data import load_splits  # noqa: E402
from src.dataset import PADDataset  # noqa: E402
from src.evaluate import evaluate_model  # noqa: E402
from src.transforms import build_eval_transform  # noqa: E402
from src.utils import resolve_device  # noqa: E402

# Mã thí nghiệm theo kế hoạch: baseline E02-E06, robust E08-E12.
EXP_PREFIX = {
    "baseline": {"jpeg": "E02", "resize": "E03", "blur": "E04",
                 "noise": "E05", "brightness": "E06"},
    "robust": {"jpeg": "E08", "resize": "E09", "blur": "E10",
               "noise": "E11", "brightness": "E12"},
}

# Định nghĩa TƯỜNG MINH các mức suy giảm (mục 11-12 tài liệu):
# blur light/medium/strong phải ánh xạ về tham số cụ thể (mục 12).
SEVERITY_GRID = {
    "jpeg": [
        ("90", {"quality": 90}),
        ("70", {"quality": 70}),
        ("50", {"quality": 50}),
        ("30", {"quality": 30}),
    ],
    "resize": [
        ("75", {"scale": 0.75}),
        ("50", {"scale": 0.50}),
        ("25", {"scale": 0.25}),
    ],
    "blur": [
        ("light", {"kernel_size": 3, "sigma": 0.6}),
        ("medium", {"kernel_size": 7, "sigma": 1.8}),
        ("strong", {"kernel_size": 11, "sigma": 3.0}),
    ],
    "noise": [
        ("low", {"std": 0.005}),
        ("medium", {"std": 0.015}),
        ("high", {"std": 0.03}),
    ],
    "brightness": [
        ("dark", {"factor": 0.6}),
        ("normal", {"factor": 1.0}),
        ("bright", {"factor": 1.4}),
    ],
}

METRIC_COLS = ["f1", "roc_auc", "pr_auc", "apcer", "bpcer", "acer", "accuracy"]


def parse_args() -> argparse.Namespace:
    """Đọc tham số: checkpoint, tag (baseline/robust), các thư mục đầu ra."""
    parser = argparse.ArgumentParser(
        description="Đánh giá checkpoint trên lưới suy giảm chất lượng đầy đủ"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="đường dẫn checkpoint cần đánh giá")
    parser.add_argument("--tag", choices=["baseline", "robust"], default="baseline",
                        help="baseline (E02-E06) hoặc robust (E08-E12)")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    return parser.parse_args()


def clean_evaluation(model, checkpoint_config: dict, device: torch.device,
                     splits_dir: str) -> dict:
    """Đánh giá model trên test set SẠCH (hàng 'Clean' của bảng)."""
    seed = checkpoint_config["seed"]
    strategy = checkpoint_config["split"]["strategy"]
    splits_file = (Path(splits_dir)
                   / f"{checkpoint_config['dataset']['name']}_seed{seed}_{strategy}.json")
    if not splits_file.is_file():
        raise FileNotFoundError(
            f"Splits file not found: {splits_file}. "
            f"Hãy chạy train_baseline trước để tạo splits."
        )
    splits = load_splits(splits_file)

    test_loader = DataLoader(
        PADDataset(splits["test"], transform=build_eval_transform(checkpoint_config)),
        batch_size=checkpoint_config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    threshold = checkpoint_config["evaluation"]["threshold"]
    result = evaluate_model(model, test_loader, device=device, threshold=threshold)
    return result["metrics"]


def save_summary_table(rows: list[dict], tables_dir: Path, tag: str) -> Path:
    """Ghi bảng tổng hợp Condition|Severity|metrics ra CSV."""
    frame = pd.DataFrame(rows)
    frame = frame[["condition", "severity"] + METRIC_COLS]
    path = tables_dir / f"degradation_{tag}.csv"
    frame.to_csv(path, index=False)
    return path


def plot_category(rows: list[dict], category: str, figures_dir: Path,
                  tag: str) -> Path:
    """Vẽ F1 và ACER theo mức suy giảm của MỘT loại suy giảm."""
    cat_rows = [r for r in rows if r["condition"] == category]
    severities = ["clean"] + [str(r["severity"]) for r in cat_rows]
    f1 = [rows[0]["f1"]] + [r["f1"] for r in cat_rows]
    acer = [rows[0]["acer"]] + [r["acer"] for r in cat_rows]

    plt.figure(figsize=(6, 4))
    plt.plot(severities, f1, marker="o", label="F1")
    plt.plot(severities, acer, marker="s", label="ACER")
    plt.xlabel(f"{category} severity")
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"{tag}: F1 / ACER theo muc {category}")
    plt.tight_layout()
    path = figures_dir / f"fig_{tag}_{category}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main() -> None:
    """Chạy lưới suy giảm: clean + mọi mức -> lưu kết quả, bảng, biểu đồ."""
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint khong ton tai: {checkpoint_path}")

    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint_config, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))
    device = resolve_device(checkpoint_config["device"]["name"])
    model.to(device)
    seed = checkpoint_config["seed"]

    prefix = EXP_PREFIX[args.tag]
    print(f"=== LUOI SUY GIAM [{args.tag}] | checkpoint: {checkpoint_path} ===")
    print(f"device={device} | seed={seed} | threshold="
          f"{checkpoint_config['evaluation']['threshold']}")

    print("\n[clean] Đang đánh giá test sạch ...", flush=True)
    clean_metrics = clean_evaluation(model, checkpoint_config, device, args.splits_dir)
    rows: list[dict] = [{"condition": "clean", "severity": "-", **clean_metrics}]
    print(f"  clean: f1={clean_metrics['f1']:.4f} acer={clean_metrics['acer']:.4f}")

    for category, severities in SEVERITY_GRID.items():
        for severity_label, params in severities:
            exp_id = f"{prefix[category]}_{category}{severity_label}_seed{seed}"
            config = {
                "seed": seed,
                "degradation": {"name": category, **params},
                "experiment_id": exp_id,
            }
            print(f"\n[{exp_id}] degradation={config['degradation']}", flush=True)
            record = run_degradation(
                config, checkpoint_path,
                splits_dir=args.splits_dir,
                results_dir=args.results_dir,
            )
            rows.append({
                "condition": category,
                "severity": severity_label,
                **{key: record[key] for key in METRIC_COLS},
            })
            print(f"  f1={record['f1']:.4f} acer={record['acer']:.4f}")

    table_path = save_summary_table(rows, tables_dir, args.tag)
    print(f"\nBang tong hop: {table_path}")

    figure_paths = [plot_category(rows, category, figures_dir, args.tag)
                    for category in SEVERITY_GRID]
    print("Bieu do:")
    for path in figure_paths:
        print(f"  {path}")

    print("\n=== XONG ===")
    print(pd.DataFrame([{k: r[k] for k in ["condition", "severity", "f1", "acer"]}
                        for r in rows]).to_string(index=False))


if __name__ == "__main__":
    main()
