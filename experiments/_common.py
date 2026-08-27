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

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.data import (
    build_samples,
    create_splits,
    discover_dataset,
    label_distribution,
    load_metadata,
    load_splits,
    save_splits,
)
from src.dataset import PADDataset
from src.evaluate import evaluate_model, save_predictions
from src.model import build_model
from src.reproducibility import get_environment_info
from src.robustness import build_robustness_transform
from src.train import train_model
from src.transforms import build_eval_transform, build_train_transform
from src.utils import (
    count_parameters,
    get_experiment_logger,
    model_size_mb,
    resolve_device,
    save_csv,
    save_json,
)


def train_and_evaluate(
    config: dict,
    experiment_id: str,
    splits_dir: str | Path = "data/splits",
    results_dir: str | Path = "results/raw",
    checkpoints_dir: str | Path = "results/checkpoints",
) -> dict:
    """Pipeline huấn luyện chung cho baseline và robustness (mục 22, 24, 31).

    Chạy: dataset -> splits -> transforms -> model -> train -> checkpoint
    -> đánh giá clean test -> lưu kết quả. Dùng chung cho train_baseline
    (training_mode=clean) và train_robust (training_mode=robust, có khối
    "robustness.enabled: true" trong config).

    So sánh công bằng (mục 24, 41): mọi tham số khác (split, model,
    optimizer, lr, epochs, batch size, test set, threshold, seed) được giữ
    nguyên từ config; biến thí nghiệm chỉ là chiến lược huấn luyện.
    """
    start_time = time.time()

    seed = config["seed"]
    logger = get_experiment_logger(experiment_id)

    robustness = config.get("robustness") or {}
    training_mode = "robust" if robustness.get("enabled") else "clean"
    logger.info(f"===== BẮT ĐẦU {experiment_id} (training_mode={training_mode}) =====")
    logger.info(f"seed={seed}")

    device = resolve_device(config["device"]["name"])
    logger.info(f"device={device}")

    # --- Bước 1: nạp dataset và splits (mục 6, 7 tài liệu) ---
    dataset_root = config["dataset"]["root"]
    info = discover_dataset(dataset_root)
    logger.info(f"dataset root: {info['root']}")

    rows = load_metadata(info["root"], annotation_file=info["annotation_file"])
    samples = build_samples(rows, image_root=info["image_root"])
    dist = label_distribution(samples)
    logger.info(
        f"samples: total={dist['total']}, bona_fide={dist['bona_fide']}, "
        f"spoof={dist['spoof']}, spoof_ratio={dist['spoof_ratio']:.4f} (mục 15)"
    )

    strategy = config["split"]["strategy"]
    splits_dir = Path(splits_dir)
    splits_path = splits_dir / f"{config['dataset']['name']}_seed{seed}_{strategy}.json"

    if splits_path.is_file():
        # Tái sử dụng splits đã lưu -> các thí nghiệm dùng cùng test set (mục 24).
        splits = load_splits(splits_path)
        logger.info(f"tái sử dụng splits: {splits_path}")
    else:
        splits = create_splits(samples, seed=seed, strategy=strategy)
        save_splits(splits, splits_path)
        logger.info(f"tạo splits mới ({strategy}) và lưu: {splits_path}")

    for name in ("train", "val", "test"):
        logger.info(f"split {name}: {splits['meta']['counts'][name]} mẫu")

    # --- Bước 2: transforms, dataset, dataloaders ---
    train_transform = build_train_transform(config)
    if training_mode == "robust":
        # Tăng cường chất lượng ngẫu nhiên (chỉ khi HUẤN LUYỆN — mục 13, 21).
        train_transform = T.Compose(
            [T.Lambda(build_robustness_transform(config)), train_transform]
        )
    eval_transform = build_eval_transform(config)

    train_loader = DataLoader(
        PADDataset(splits["train"], transform=train_transform),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_split = splits["val"]
    val_loader = None
    if val_split:
        val_loader = DataLoader(
            PADDataset(val_split, transform=eval_transform),
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )
    test_loader = DataLoader(
        PADDataset(splits["test"], transform=eval_transform),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # --- Bước 3: model, optimizer, loss ---
    model = build_model(config["model"]["name"], num_classes=1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Trọng số lớp dương (mục 15): chỉ bật khi cấu hình yêu cầu.
    loss_kwargs = {}
    if config["loss"].get("use_pos_weight"):
        train_dist = label_distribution(splits["train"])
        if train_dist["spoof"] > 0:
            loss_kwargs["pos_weight"] = torch.tensor(
                [train_dist["bona_fide"] / train_dist["spoof"]], device=device
            )
            logger.info(f"pos_weight={loss_kwargs['pos_weight'].item():.4f} (mục 15)")
    loss_fn = nn.BCEWithLogitsLoss(**loss_kwargs)

    logger.info(
        f"model={config['model']['name']}, "
        f"params={count_parameters(model)}, size={model_size_mb(model):.2f} MB"
    )

    # --- Bước 4: huấn luyện + lưu checkpoint sau mỗi epoch (mục 38) ---
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{experiment_id}.pt"

    def save_checkpoint(current_model: nn.Module, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "model_state_dict": current_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
                "seed": seed,
                "last_metrics": metrics,
            },
            checkpoint_path,
        )

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        epochs=config["training"]["epochs"],
        device=device,
        on_epoch_end=save_checkpoint,
    )
    logger.info(f"hoàn tất huấn luyện {len(history)} epoch, checkpoint: {checkpoint_path}")
    logger.info(
        f"epoch cuối: train_loss={history[-1]['train_loss']:.4f}, "
        f"val_loss={history[-1]['val_loss']}"
    )

    # --- Bước 5: đánh giá trên tập test SẠCH (mục 22, 24) ---
    threshold = config["evaluation"]["threshold"]
    eval_result = evaluate_model(model, test_loader, device=device, threshold=threshold)
    logger.info(f"clean test: f1={eval_result['metrics']['f1']:.4f}, "
                f"roc_auc={eval_result['metrics']['roc_auc']}")

    runtime_seconds = round(time.time() - start_time, 2)

    # --- Bước 6: ghép record kết quả (mục 29) và lưu (mục 30) ---
    record = {
        "experiment_id": experiment_id,
        "seed": seed,
        "dataset": config["dataset"]["name"],
        "split_strategy": strategy,
        "model": config["model"]["name"],
        "training_mode": training_mode,
        "degradation_name": "none",
        "degradation_parameters": {},
        "threshold": threshold,
        **eval_result["metrics"],
        "runtime_seconds": runtime_seconds,
        "parameter_count": count_parameters(model),
        "model_size_mb": round(model_size_mb(model), 4),
        "environment": get_environment_info(),
        "train_history": history,
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(record, results_dir / f"{experiment_id}.json")

    # CSV 1 dòng: chỉ giữ các trường số/chữ đơn giản (bỏ history/environment).
    flat_record = {key: value for key, value in record.items()
                   if key not in ("train_history", "environment")}
    save_csv([flat_record], results_dir / f"{experiment_id}.csv")

    save_predictions(eval_result, results_dir / f"{experiment_id}_predictions.csv")

    logger.info(f"kết quả đã lưu: {results_dir / (experiment_id + '.json')} "
                f"(runtime={runtime_seconds}s)")
    return record


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
