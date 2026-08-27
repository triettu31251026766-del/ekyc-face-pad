"""experiments/_common.py — hàm điều phối DÙNG CHUNG cho các script đánh giá.

Tệp này dùng để (theo mục 31 của tài liệu kỹ thuật):
- load_checkpoint(...): tải checkpoint (mục 38) và dựng lại model từ config
  đã lưu trong checkpoint — KHÔNG huấn luyện lại.
- load_test_loader(...): nạp lại đúng tập test theo splits đã lưu (mục 7, 24:
  các thí nghiệm phải dùng CÙNG test set).
- finalize(...): ghép record kết quả theo schema mục 29 và lưu JSON/CSV/
  predictions (mục 30, 39).

Chú ý: tệp này chỉ chứa code điều phối; mọi logic cốt lõi nằm trong src/.
Các script eval_clean.py / eval_degradation.py dùng chung các hàm ở đây.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import (
    build_samples,
    create_splits,
    discover_dataset,
    load_metadata,
    load_splits,
    save_splits,
)
from src.dataset import PADDataset
from src.evaluate import save_predictions
from src.model import build_model
from src.reproducibility import get_environment_info
from src.transforms import build_eval_transform
from src.utils import (
    count_parameters,
    get_experiment_logger,
    model_size_mb,
    resolve_device,
    save_csv,
    save_json,
)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    """Tải checkpoint và dựng lại model + config từ checkpoint (mục 38).

    Returns:
        (model, config, checkpoint_dict)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' thiếu 'config' hoặc 'model_state_dict'"
        )

    config = checkpoint["config"]
    model = build_model(config["model"]["name"], num_classes=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


def load_test_loader(config: dict, splits_dir: str | Path) -> tuple[DataLoader, dict]:
    """Nạp tập test theo splits đã lưu (tái sử dụng nếu có — mục 24 tài liệu).

    Returns:
        (test_loader, splits)
    """
    seed = config["seed"]
    strategy = config["split"]["strategy"]
    info = discover_dataset(config["dataset"]["root"])
    rows = load_metadata(info["root"], annotation_file=info["annotation_file"])
    samples = build_samples(rows, image_root=info["image_root"])

    splits_dir = Path(splits_dir)
    splits_path = splits_dir / f"{config['dataset']['name']}_seed{seed}_{strategy}.json"
    if splits_path.is_file():
        splits = load_splits(splits_path)
    else:
        splits = create_splits(samples, seed=seed, strategy=strategy)
        save_splits(splits, splits_path)

    transform = build_eval_transform(config)
    test_loader = DataLoader(
        PADDataset(splits["test"], transform=transform),
        batch_size=config["training"]["batch_size"],
        shuffle=False,  # bắt buộc để đánh giá tất định (mục 18)
        num_workers=0,
    )
    return test_loader, splits


def finalize(
    experiment_id: str,
    config: dict,
    model: torch.nn.Module,
    eval_result: dict,
    runtime_seconds: float,
    results_dir: str | Path,
    record_extras: dict | None = None,
) -> dict:
    """Ghép record kết quả (mục 29) và lưu JSON + CSV + predictions (mục 30, 39).

    Args:
        experiment_id: Mã thí nghiệm (mục 28).
        config: Config lưu trong checkpoint (chứa seed, model, evaluation...).
        model: Model đã đánh giá (để tính parameter_count/model_size_mb).
        eval_result: Kết quả từ src/evaluate.evaluate_model.
        runtime_seconds: Tổng thời gian chạy thí nghiệm (do script đo).
        results_dir: Thư mục lưu kết quả.
        record_extras: Các trường bổ sung (ví dụ degradation_name, ...).

    Returns:
        record đầy đủ (đã lưu ra đĩa).
    """
    record = {
        "experiment_id": experiment_id,
        "seed": config["seed"],
        "dataset": config["dataset"]["name"],
        "split_strategy": config["split"]["strategy"],
        "model": config["model"]["name"],
        "training_mode": "clean",
        "degradation_name": "none",
        "degradation_parameters": {},
        "threshold": config["evaluation"]["threshold"],
        **eval_result["metrics"],
        "parameter_count": count_parameters(model),
        "model_size_mb": round(model_size_mb(model), 4),
        "environment": get_environment_info(),
    }
    if record_extras:
        record.update(record_extras)
    record["runtime_seconds"] = runtime_seconds

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(record, results_dir / f"{experiment_id}.json")

    flat = {key: value for key, value in record.items()
            if key != "environment"}
    save_csv([flat], results_dir / f"{experiment_id}.csv")
    save_predictions(eval_result, results_dir / f"{experiment_id}_predictions.csv")

    logger = get_experiment_logger(experiment_id)
    logger.info(f"kết quả đã lưu: {results_dir / (experiment_id + '.json')}")
    return record
