"""experiments/run_all.py — chạy tuần tự toàn bộ chuỗi thí nghiệm.

Tệp này dùng để (theo mục 32 của tài liệu kỹ thuật):
Chạy ĐÚNG thứ tự quy định, dừng ngay nếu bước nào lỗi (không chạy lưới lớn
khi E01 chưa thành công):

    E01      Clean baseline (train_baseline)
        -> E02..  Baseline dưới từng suy giảm chất lượng (eval_degradation)
        -> E07    Robust training (train_robust)
        -> E08..  Robust dưới từng suy giảm chất lượng (eval_degradation)
        -> E09+   Ablation từng phép tăng cường (ablation)
        ->        Bảng so sánh + figures (compare_models)

Mã thí nghiệm (experiment_id) được đánh tự động theo thứ tự chạy (mục 28):
    E01_baseline_seed42
    E02_<tên suy giảm>_seed42 ...
    E07_robust_seed42
    E08_robust_<tên suy giảm>_seed42 ...
    E09_ablation_<phép>_seed42 ...

Chạy:
    python experiments/run_all.py \
        --config configs/clean.yaml \
        --robustness configs/robustness.yaml \
        --degradations configs/degradation_*.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import ablation, compare_models, eval_degradation, train_baseline, train_robust
from src.config import load_config
from src.utils import get_experiment_logger

logger = get_experiment_logger("run_all")


def run(
    base_config: dict,
    robustness_config: dict,
    degradation_configs: list[dict],
    splits_dir: str | Path = "data/splits",
    results_dir: str | Path = "results/raw",
    checkpoints_dir: str | Path = "results/checkpoints",
    tables_dir: str | Path = "results/tables",
    figures_dir: str | Path = "results/figures",
    include_ablation: bool = True,
) -> dict:
    """Chạy toàn bộ chuỗi thí nghiệm theo đúng thứ tự mục 32. Dừng nếu lỗi.

    Args:
        base_config: Cấu hình huấn luyện cơ sở (dataset/split/model/...).
        robustness_config: Cấu hình robustness (cho E07 và ablation).
        degradation_configs: Danh sách cấu hình suy giảm (theo thứ tự chạy).
        include_ablation: Nếu True thì chạy bước E09+ ablation (mặc định).

    Returns:
        dict tóm tắt: experiment_ids (theo thứ tự chạy) + đường dẫn bảng/figure.
    """
    seed = base_config["seed"]
    summary = {"experiment_ids": []}

    def record_id(record: dict) -> str:
        experiment_id = record["experiment_id"]
        summary["experiment_ids"].append(experiment_id)
        logger.info(f"[OK] {experiment_id}")
        return experiment_id

    # --- Bước 1 (mục 32): E01 clean baseline ---
    logger.info("===== BƯỚC 1: E01 clean baseline =====")
    baseline_record = train_baseline.run(
        base_config, splits_dir=splits_dir, results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )
    baseline_id = record_id(baseline_record)
    baseline_checkpoint = Path(checkpoints_dir) / f"{baseline_id}.pt"

    # --- Bước 2: baseline dưới từng suy giảm chất lượng (E02...) ---
    logger.info("===== BƯỚC 2: baseline dưới suy giảm chất lượng =====")
    for index, deg_config in enumerate(degradation_configs, start=2):
        experiment_id = f"E{index:02d}_{_deg_slug(deg_config)}_seed{seed}"
        tagged = {**deg_config, "experiment_id": experiment_id}
        record = eval_degradation.run(
            tagged, baseline_checkpoint,
            splits_dir=splits_dir, results_dir=results_dir,
        )
        record_id(record)

    # --- Bước 3 (mục 32): E07 robust training ---
    logger.info("===== BƯỚC 3: E07 robust training =====")
    robust_record = train_robust.run(
        base_config, robustness_config,
        splits_dir=splits_dir, results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )
    robust_id = record_id(robust_record)
    robust_checkpoint = Path(checkpoints_dir) / f"{robust_id}.pt"

    # --- Bước 4: robust dưới từng suy giảm chất lượng (E08...) ---
    logger.info("===== BƯỚC 4: robust dưới suy giảm chất lượng =====")
    start_index = 8  # sau E07_robust (mục 28, 32)
    for offset, deg_config in enumerate(degradation_configs):
        index = start_index + offset
        experiment_id = f"E{index:02d}_robust_{_deg_slug(deg_config)}_seed{seed}"
        tagged = {**deg_config, "experiment_id": experiment_id}
        record = eval_degradation.run(
            tagged, robust_checkpoint,
            splits_dir=splits_dir, results_dir=results_dir,
        )
        record_id(record)

    # --- Bước 5 (mục 32): E09+ ablation ---
    if include_ablation:
        logger.info("===== BƯỚC 5: E09+ ablation =====")
        ablation_result = ablation.run(
            base_config, robustness_config,
            splits_dir=splits_dir, results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
            tables_dir=tables_dir, figures_dir=figures_dir,
        )
        summary["ablation_table"] = ablation_result["table"]
        summary["ablation_figure"] = ablation_result["figure"]

    # --- Bước 6 (mục 32): bảng so sánh + figures cuối cùng ---
    logger.info("===== BƯỚC 6: bảng so sánh + figures =====")
    comparison = compare_models.run(
        results_dir=results_dir, tables_dir=tables_dir, figures_dir=figures_dir,
    )
    summary["comparison_table"] = comparison["table"]
    summary["figures"] = comparison["figures"]

    logger.info(f"===== HOÀN TẤT run_all: {len(summary['experiment_ids'])} thí nghiệm =====")
    return summary


def _deg_slug(deg_config: dict) -> str:
    """Tên ngắn của cấu hình suy giảm cho experiment_id (mục 28)."""
    degradation = deg_config["degradation"]
    name = degradation["name"]
    param = next(value for key, value in degradation.items() if key != "name")
    return f"{name}{param}".replace(".", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy toàn bộ chuỗi thí nghiệm (mục 32)")
    parser.add_argument("--config", default="configs/clean.yaml",
                        help="đường dẫn tệp cấu hình huấn luyện cơ sở")
    parser.add_argument("--robustness", default="configs/robustness.yaml",
                        help="đường dẫn tệp cấu hình robustness")
    parser.add_argument("--degradations", nargs="+", default=None,
                        help="danh sách tệp cấu hình suy giảm "
                             "(mặc định: configs/degradation_*.yaml)")
    parser.add_argument("--no-ablation", action="store_true",
                        help="bỏ qua bước ablation E09+")
    args = parser.parse_args()

    base_config = load_config(args.config)
    robustness_config = load_config(args.robustness)

    if args.degradations:
        degradation_configs = [load_config(path) for path in args.degradations]
    else:
        degradation_configs = [
            load_config(path)
            for path in sorted(Path("configs").glob("degradation_*.yaml"))
        ]

    run(base_config, robustness_config, degradation_configs,
        include_ablation=not args.no_ablation)


if __name__ == "__main__":
    main()
