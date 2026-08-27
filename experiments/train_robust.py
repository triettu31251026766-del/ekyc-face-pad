"""experiments/train_robust.py — thí nghiệm E07: huấn luyện robustness.

Tệp này dùng để (theo mục 24, 31 của tài liệu kỹ thuật):
Pipeline:
    load config (cơ sở) + config robustness
    -> bật tăng cường chất lượng khi HUẤN LUYỆN (chỉ train, không eval)
    -> train model -> save checkpoint -> evaluate clean test -> save results

QUY TẮC SO SÁNH CÔNG BẰNG (mục 24, 41):
- Giữ nguyên: dataset split, kiến trúc model, optimizer, learning rate,
  epochs, batch size, test set, threshold, random seed.
- Biến thí nghiệm DUY NHẤT: chiến lược huấn luyện robustness (tăng cường
  chất lượng ngẫu nhiên từ src/robustness.py + src/degradation.py).

Chạy:
    python experiments/train_robust.py \
        --config configs/clean.yaml \
        --robustness configs/robustness.yaml

Đầu ra (mục 28):
    results/raw/E07_robust_seed42.json
    results/raw/E07_robust_seed42.csv
    results/raw/E07_robust_seed42_predictions.csv
    results/checkpoints/E07_robust_seed42.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments._common import train_and_evaluate
from src.config import load_config
from src.reproducibility import set_seed


def run(
    config: dict,
    robustness_config: dict | None = None,
    splits_dir: str | Path = "data/splits",
    results_dir: str | Path = "results/raw",
    checkpoints_dir: str | Path = "results/checkpoints",
) -> dict:
    """Huấn luyện model robustness theo cấu hình, trả về record kết quả.

    Args:
        config: Cấu hình huấn luyện cơ sở (dataset/split/model/training/...).
        robustness_config: Cấu hình tăng cường (chứa khối "robustness").
            Nếu None, khối "robustness" phải nằm sẵn trong config.

    Returns:
        record kết quả (mục 29) với training_mode="robust".
    """
    seed = config["seed"]
    experiment_id = config.get("experiment_id", f"E07_robust_seed{seed}")

    # Ghép khối robustness vào config cơ sở (giữ nguyên mọi tham số khác).
    if robustness_config is None:
        if "robustness" not in config:
            raise ValueError(
                "Config thiếu khối 'robustness'. Truyền --robustness hoặc "
                "bổ sung 'robustness' vào config."
            )
        full_config = dict(config)
    else:
        full_config = {**config, "robustness": robustness_config["robustness"]}
        full_config["experiment_id"] = experiment_id

    set_seed(seed)
    return train_and_evaluate(
        full_config,
        experiment_id=experiment_id,
        splits_dir=splits_dir,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E07: train robust PAD model")
    parser.add_argument("--config", default="configs/clean.yaml",
                        help="đường dẫn tệp cấu hình huấn luyện cơ sở")
    parser.add_argument("--robustness", default=None,
                        help="đường dẫn tệp cấu hình robustness "
                             "(mặc định: lấy khối robustness trong --config)")
    args = parser.parse_args()

    config = load_config(args.config)
    robustness_config = load_config(args.robustness) if args.robustness else None
    run(config, robustness_config)


if __name__ == "__main__":
    main()
