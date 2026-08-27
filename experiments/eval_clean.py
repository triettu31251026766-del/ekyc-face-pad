"""experiments/eval_clean.py — đánh giá lại checkpoint trên tập test SẠCH.

Tệp này dùng để (theo mục 31 của tài liệu kỹ thuật):
Pipeline:
    load config -> load checkpoint -> load clean test set -> evaluate
    -> save metrics

KHÔNG huấn luyện lại (mục 23). Dùng lại đúng splits đã lưu khi huấn luyện
để so sánh công bằng với các thí nghiệm khác (mục 24).

Chạy:
    python experiments/eval_clean.py \
        --config configs/clean.yaml \
        --checkpoint results/checkpoints/E01_baseline_seed42.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments._common import finalize, load_checkpoint, load_test_loader
from src.config import load_config
from src.evaluate import evaluate_model
from src.reproducibility import set_seed
from src.utils import get_experiment_logger, resolve_device


def run(
    config: dict,
    checkpoint_path: str | Path,
    splits_dir: str | Path = "data/splits",
    results_dir: str | Path = "results/raw",
) -> dict:
    """Đánh giá checkpoint trên tập test sạch, trả về record kết quả."""
    start = time.time()
    set_seed(config["seed"])

    checkpoint_path = Path(checkpoint_path)
    model, checkpoint_config, _ = load_checkpoint(checkpoint_path, torch_device(config))
    experiment_id = config.get(
        "experiment_id", f"{checkpoint_config.get('experiment_id', 'E01')}_eval_clean_seed{config['seed']}"
    )

    logger = get_experiment_logger(experiment_id)
    logger.info(f"===== BẮT ĐẦU {experiment_id} (clean evaluation) =====")
    logger.info(f"checkpoint: {checkpoint_path}")

    # Dùng config trong checkpoint (cùng dataset/split/threshold với lúc train).
    test_loader, _ = load_test_loader(checkpoint_config, splits_dir)
    threshold = checkpoint_config["evaluation"]["threshold"]

    eval_result = evaluate_model(model, test_loader, device=torch_device(checkpoint_config),
                                 threshold=threshold)
    logger.info(f"clean test: f1={eval_result['metrics']['f1']:.4f}, "
                f"roc_auc={eval_result['metrics']['roc_auc']}")

    runtime = round(time.time() - start, 2)
    record = finalize(experiment_id, checkpoint_config, model, eval_result,
                      runtime_seconds=runtime, results_dir=results_dir)
    return record


def torch_device(config: dict):
    """Chọn thiết bị từ config (dùng chung cho load_checkpoint và evaluate)."""
    return resolve_device(config["device"]["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint trên tập test sạch")
    parser.add_argument("--config", default="configs/clean.yaml",
                        help="đường dẫn tệp cấu hình YAML")
    parser.add_argument("--checkpoint", required=True,
                        help="đường dẫn checkpoint cần đánh giá")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, args.checkpoint)


if __name__ == "__main__":
    main()
