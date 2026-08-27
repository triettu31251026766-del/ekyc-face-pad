"""experiments/train_baseline.py — thí nghiệm E01: huấn luyện clean baseline.

Tệp này dùng để (theo mục 22, 31 của tài liệu kỹ thuật):
Pipeline (đúng mục 31):
    load config -> set seed -> load dataset -> load/tạo splits
    -> build transforms -> build model -> train -> save checkpoint
    -> evaluate clean test -> save results

Đầu ra mong đợi (mục 22):
    results/raw/E01_baseline_seed42.json   : kết quả đầy đủ (mục 29, 30)
    results/raw/E01_baseline_seed42.csv    : bảng 1 dòng các metric
    results/raw/E01_baseline_seed42_predictions.csv : dự đoán thô (mục 39)
    results/checkpoints/E01_baseline_seed42.pt      : checkpoint (mục 38)
    data/splits/<dataset>_seed<seed>_<strategy>.json: splits tái lập (mục 7)

Chạy:
    python experiments/train_baseline.py --config configs/clean.yaml

Chú ý: tệp này chỉ điều phối (orchestration); mọi logic cốt lõi nằm trong
src/ (data, dataset, transforms, model, train, evaluate, metrics, utils).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_config
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
from src.reproducibility import get_environment_info, set_seed
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


def run(
    config: dict,
    splits_dir: str | Path = "data/splits",
    results_dir: str | Path = "results/raw",
    checkpoints_dir: str | Path = "results/checkpoints",
) -> dict:
    """Chạy thí nghiệm E01 theo cấu hình, trả về record kết quả (mục 29)."""
    start_time = time.time()

    seed = config["seed"]
    experiment_id = config.get("experiment_id", f"E01_baseline_seed{seed}")
    set_seed(seed)

    logger = get_experiment_logger(experiment_id)
    logger.info(f"===== BẮT ĐẦU {experiment_id} (training_mode=clean) =====")
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
        # Tái sử dụng splits đã lưu -> các thí nghiệm sau dùng cùng test set (mục 24).
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

    # --- Bước 5: đánh giá trên tập test SẠCH (mục 22) ---
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
        "training_mode": "clean",
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
