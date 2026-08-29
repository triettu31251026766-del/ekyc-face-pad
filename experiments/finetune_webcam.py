"""experiments/finetune_webcam.py — fine-tune model với ảnh webcam thật.

Tệp này dùng để thu hẹp domain gap giữa dataset CelebA-Spoof và webcam
thực tế (phát hiện qua Test A/B/C/D của camera demo: model dataset gọi
mặt webcam là spoof với xác suất rất cao).

Pipeline:
    1) nạp checkpoint (mặc định E07 robust)
    2) dựng tập fine-tune: ảnh webcam live/spoof (nếu có) + trộn thêm
       mẫu live/spoof từ train split của dataset (tránh quên kiến thức cũ)
    3) đánh giá model TRƯỚC khi fine-tune trên tập ảnh webcam live
    4) fine-tune vài epoch với lr nhỏ, giữ nguyên kiến trúc model
    5) đánh giá lại SAU fine-tune + lưu checkpoint mới

Cách dùng (chạy từ thư mục gốc dự án):
    python -m scripts.collect_webcam_data --mode live --count 300
    python -m scripts.collect_webcam_data --mode spoof --count 200
    python -m experiments.finetune_webcam
    python -m scripts.camera_demo --checkpoint results/checkpoints/E20_webcam_finetune_seed123.pt --margin 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from experiments._common import load_checkpoint
from src.config import load_config
from src.data import Sample, load_splits
from src.dataset import PADDataset
from src.reproducibility import set_seed
from src.robustness import build_robustness_transform
from src.train import train_model
from src.transforms import build_eval_transform, build_train_transform
from src.utils import resolve_device

DEFAULT_OUTPUT = "results/checkpoints/E20_webcam_finetune_seed123.pt"


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh của fine-tune."""
    parser = argparse.ArgumentParser(description="Fine-tune model PAD với ảnh webcam")
    parser.add_argument("--checkpoint", default="results/checkpoints/E07_robust_seed123.pt")
    parser.add_argument("--webcam-dir", default="data/webcam_data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-live", type=int, default=1000,
                        help="số mẫu live của dataset trộn vào tập fine-tune")
    parser.add_argument("--dataset-spoof", type=int, default=2000,
                        help="số mẫu spoof của dataset trộn vào tập fine-tune")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_finetune_samples(config: dict, webcam_dir: str, dataset_live: int,
                           dataset_spoof: int) -> list[Sample]:
    """Ghép mẫu webcam + mẫu dataset thành danh sách Sample để fine-tune."""
    webcam_dir = Path(webcam_dir)
    samples: list[Sample] = []

    live_dir = webcam_dir / "live"
    spoof_dir = webcam_dir / "spoof"
    if live_dir.is_dir():
        for i, path in enumerate(sorted(live_dir.glob("*.jpg"))):
            samples.append(Sample(path=str(path), label=0,
                                  subject_id=f"webcam_live_{i}",
                                  attack_type="bona_fide"))
    if spoof_dir.is_dir():
        for i, path in enumerate(sorted(spoof_dir.glob("*.jpg"))):
            samples.append(Sample(path=str(path), label=1,
                                  subject_id=f"webcam_spoof_{i}",
                                  attack_type="photo"))

    splits_path = (Path("data/splits")
                   / f"{config['dataset']['name']}_seed{config['seed']}"
                     f"_{config['split']['strategy']}.json")
    train_split = load_splits(splits_path)["train"]
    samples += [s for s in train_split if s.label == 0][:dataset_live]
    samples += [s for s in train_split if s.label == 1][:dataset_spoof]
    return samples


def evaluate_live_probs(model: nn.Module, samples: list[Sample], transform,
                        device: torch.device, batch_size: int = 64) -> float:
    """Tính xác suất spoof trung bình trên tập ảnh webcam live (label 0)."""
    live = [s for s in samples if s.subject_id.startswith("webcam_live")]
    if not live:
        return float("nan")
    loader = DataLoader(PADDataset(live, transform=transform),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    probs: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probs.extend(torch.sigmoid(logits).flatten().tolist())
    return sum(probs) / len(probs)


def main() -> None:
    """Chạy fine-tune: nạp checkpoint -> dựng tập -> train -> lưu checkpoint mới."""
    args = parse_args()

    device = resolve_device("auto")
    model, config, _ = load_checkpoint(Path(args.checkpoint), device)
    set_seed(config["seed"])

    # Bật tăng cường robustness (bao gồm crop) để fine-tune không quên
    # khả năng chống suy giảm chất lượng.
    robust = load_config("configs/robustness.yaml")
    config["robustness"] = robust["robustness"]

    samples = build_finetune_samples(config, args.webcam_dir,
                                     args.dataset_live, args.dataset_spoof)
    webcam_count = sum(1 for s in samples if s.subject_id.startswith("webcam"))
    if webcam_count == 0:
        raise FileNotFoundError(
            f"Chua co anh webcam trong {args.webcam_dir}. "
            f"Chay truoc: python -m scripts.collect_webcam_data --mode live --count 300"
        )

    eval_transform = build_eval_transform(config)
    train_transform = T.Compose(
        [T.Lambda(build_robustness_transform(config)),
         build_train_transform(config)]
    )
    train_loader = DataLoader(
        PADDataset(samples, transform=train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
    )

    before = evaluate_live_probs(model, samples, eval_transform, device)
    print(f"Tap fine-tune: {len(samples)} mau ({webcam_count} webcam) | "
          f"epochs={args.epochs} | lr={args.lr} | batch={args.batch_size}")
    print(f"P(spoof) trung binh tren anh webcam live TRUOC fine-tune: {before:.3f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = train_model(model, train_loader, None, optimizer, loss_fn,
                          epochs=args.epochs, device=device)
    print(f"Fine-tune xong: {len(history)} epoch, "
          f"train_loss cuoi = {history[-1]['train_loss']:.4f}")

    after = evaluate_live_probs(model, samples, eval_transform, device)
    print(f"P(spoof) trung binh tren anh webcam live SAU fine-tune: {after:.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "config": config,
            "seed": config["seed"],
            "finetune": {
                "checkpoint": args.checkpoint,
                "webcam_samples": webcam_count,
                "lr": args.lr,
                "epochs": args.epochs,
                "live_prob_before": before,
                "live_prob_after": after,
            },
        },
        output,
    )
    print(f"Checkpoint da luu: {output}")
    print("Thu camera: python -m scripts.camera_demo "
          f"--checkpoint {output} --margin 0")


if __name__ == "__main__":
    main()
