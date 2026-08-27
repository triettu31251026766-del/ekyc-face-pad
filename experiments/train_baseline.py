"""experiments/train_baseline.py — thí nghiệm E01: huấn luyện clean baseline.

Tệp này dùng để (theo mục 22, 31 của tài liệu kỹ thuật):
Pipeline (đúng mục 31):
    load config -> set seed -> load dataset -> load/tạo splits
    -> build transforms -> build model -> train -> save checkpoint
    -> evaluate clean test -> save results

Toàn bộ logic điều phối nằm trong experiments/_common.train_and_evaluate
(dùng chung với train_robust — mục 24: so sánh công bằng). Tệp này chỉ cung
cấp entry point riêng cho thí nghiệm clean baseline.

Đầu ra mong đợi (mục 22):
    results/raw/E01_baseline_seed42.json
    results/raw/E01_baseline_seed42.csv
    results/raw/E01_baseline_seed42_predictions.csv
    results/checkpoints/E01_baseline_seed42.pt
    data/splits/<dataset>_seed<seed>_<strategy>.json

Chạy:
    python experiments/train_baseline.py --config configs/clean.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments._common import train_and_evaluate
from src.config import load_config
from src.reproducibility import set_seed


def run(
    config: dict,
    splits_dir: str | Path = "data/splits",
    results_dir: str | Path = "results/raw",
    checkpoints_dir: str | Path = "results/checkpoints",
) -> dict:
    """Chạy thí nghiệm E01 theo cấu hình, trả về record kết quả (mục 29)."""
    seed = config["seed"]
    experiment_id = config.get("experiment_id", f"E01_baseline_seed{seed}")
    set_seed(seed)

    return train_and_evaluate(
        config,
        experiment_id=experiment_id,
        splits_dir=splits_dir,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )


def main() -> None:
    """Điểm vào CLI: đọc --config rồi chạy run()."""
    parser = argparse.ArgumentParser(description="E01: train clean baseline PAD model")
    parser.add_argument("--config", default="configs/clean.yaml",
                        help="đường dẫn tệp cấu hình YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
