"""experiments/eval_degradation.py — đánh giá checkpoint trên tập test SUY GIẢM.

Tệp này dùng để (theo mục 23, 31 của tài liệu kỹ thuật):
Pipeline:
    load config (degradation) -> load checkpoint -> load test set
    -> áp dụng suy giảm TẤT ĐỊNH -> evaluate -> save metrics

QUAN TRỌNG (mục 23):
- KHÔNG huấn luyện lại model. Mục đích: đo phản ứng của CÙNG model với các
  điều kiện chất lượng đầu vào khác nhau (JPEG, resize, blur, noise, brightness).
- Suy giảm PHẢI tất định (mục 13): cùng config -> cùng ảnh đã suy giảm. Với
  noise, seed lấy từ config["seed"] (xem src/degradation.py).

Chạy:
    python experiments/eval_degradation.py \
        --config configs/degradation_jpeg.yaml \
        --checkpoint results/checkpoints/E01_baseline_seed123.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from experiments._common import finalize, load_checkpoint
from src.config import load_config
from src.data import load_splits
from src.dataset import PADDataset
from src.degradation import apply_degradation_config
from src.evaluate import evaluate_model
from src.reproducibility import set_seed
from src.transforms import build_eval_transform
from src.utils import get_experiment_logger, resolve_device


def run(
    config: dict,
    checkpoint_path: str | Path,
    splits_dir: str | Path = "data/splits",
    results_dir: str | Path = "results/raw",
) -> dict:
    """Đánh giá checkpoint trên tập test đã suy giảm chất lượng (tất định)."""
    start = time.time()
    set_seed(config["seed"])

    degradation = config["degradation"]
    degradation_name = degradation["name"]

    # Tham số suy giảm ghi vào kết quả (mục 11: tham số tường minh, mục 29).
    parameters = {key: value for key, value in degradation.items() if key != "name"}

    checkpoint_path = Path(checkpoint_path)
    # Nạp checkpoint về CPU trước, rồi chọn device theo config trong checkpoint.
    model, checkpoint_config, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))
    device = resolve_device(checkpoint_config["device"]["name"])
    model.to(device)

    seed = checkpoint_config["seed"]
    experiment_id = config.get(
        "experiment_id", f"{degradation_name}{_short_params(parameters)}_seed{seed}"
    )

    logger = get_experiment_logger(experiment_id)
    logger.info(f"===== BẮT ĐẦU {experiment_id} (degradation evaluation) =====")
    logger.info(f"checkpoint: {checkpoint_path}")
    logger.info(f"degradation: {degradation}")

    # Dùng đúng splits đã lưu khi huấn luyện (cùng test set — mục 24).
    strategy = checkpoint_config["split"]["strategy"]
    splits_file = (
        Path(splits_dir)
        / f"{checkpoint_config['dataset']['name']}_seed{seed}_{strategy}.json"
    )
    if not splits_file.is_file():
        raise FileNotFoundError(
            f"Splits file not found: {splits_file}. "
            f"Hãy chạy train_baseline trước để tạo splits."
        )
    splits = load_splits(splits_file)

    # Transform eval: áp dụng suy giảm TẤT ĐỊNH TRƯỚC, rồi mới tiền xử lý chuẩn.
    eval_transform = T.Compose(
        [
            T.Lambda(lambda image: apply_degradation_config(image, config)),
            build_eval_transform(checkpoint_config),
        ]
    )
    test_loader = DataLoader(
        PADDataset(splits["test"], transform=eval_transform),
        batch_size=checkpoint_config["training"]["batch_size"],
        shuffle=False,  # bắt buộc để đánh giá tất định (mục 18)
        num_workers=0,
    )

    threshold = checkpoint_config["evaluation"]["threshold"]
    eval_result = evaluate_model(model, test_loader, device=device, threshold=threshold)
    logger.info(f"degraded test: f1={eval_result['metrics']['f1']:.4f}, "
                f"roc_auc={eval_result['metrics']['roc_auc']}")

    runtime = round(time.time() - start, 2)
    record = finalize(
        experiment_id,
        checkpoint_config,
        model,
        eval_result,
        runtime_seconds=runtime,
        results_dir=results_dir,
        record_extras={
            "degradation_name": degradation_name,
            "degradation_parameters": parameters,
        },
    )
    return record


def _short_params(parameters: dict) -> str:
    """Rút gọn tham số suy giảm thành chuỗi ngắn cho experiment_id (mục 28)."""
    return "".join(str(value) for value in parameters.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đánh giá checkpoint trên tập test đã suy giảm chất lượng"
    )
    parser.add_argument("--config", required=True,
                        help="đường dẫn tệp cấu hình suy giảm (configs/degradation_*.yaml)")
    parser.add_argument("--checkpoint", required=True,
                        help="đường dẫn checkpoint cần đánh giá")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, args.checkpoint)


if __name__ == "__main__":
    main()
