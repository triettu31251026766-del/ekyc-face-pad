"""scripts/download_celeba_full.py — tải subset lớn CelebA-Spoof (~200k) từ HuggingFace.

Tệp này dùng để (giai đoạn CHÍNH của đồ án):
- Tải một phần các shard parquet của mirror "Ar4ikov/celebA_spoof"
  (~526k ảnh, chia sẵn train/valid/test theo protocol chính thức):
      train  : mặc định 45 shard  -> ~200k ảnh
      test   : mặc định 22 shard  -> ~20k ảnh  (KHÓA làm test set cố định)
      valid  : mặc định 4 shard   -> ~20k ảnh  (validation)
  Tổng ~24GB thay vì tải full 74GB từ Google Drive.
- Crop khuôn mặt theo Bbox (mở rộng 10%) giống đúng pipeline pilot 18k.
- Sinh cấu trúc data/raw/celeba_spoof_full/:
      SpoofingData/  (ảnh crop, tên 000000.jpg ...)
      train_list.txt / test_list.txt / valid_list.txt
      dataset_report.json (thống kê EDA cơ bản)
- Lưu splits cố định vào data/splits/celeba_spoof_full_seed123_subject_disjoint.json
  (train/val/test lấy THEO SPLIT CHÍNH THỨC của mirror -> subject-disjoint theo
  protocol dataset, KHÔNG trộn lại).

LƯU Ý: mirror này chỉ có nhãn nhị phân live/spoof (không có spoof type,
illumination, environment như bản full 74GB).

Cách dùng (chạy từ thư mục gốc dự án):
    python -m scripts.download_celeba_full
    python -m scripts.download_celeba_full --train-shards 22 --test-shards 11 --valid-shards 2
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = "Ar4ikov/celebA_spoof"
API_ROOT = f"https://huggingface.co/api/datasets/{REPO}/parquet"

OUT_ROOT = Path("data/raw/celeba_spoof_full")
IMAGE_DIR = OUT_ROOT / "SpoofingData"

MARGIN = 0.10  # mở rộng bbox 10% quanh mặt (khớp pilot + dataset chính thức)


def parse_args() -> argparse.Namespace:
    """Đọc tham số: số shard mỗi split + đường dẫn lưu."""
    parser = argparse.ArgumentParser(
        description="Tải subset lớn CelebA-Spoof từ HuggingFace (parquet)"
    )
    parser.add_argument("--train-shards", type=int, default=45,
                        help="số shard train (mỗi shard ~4.5k ảnh, ~300MB)")
    parser.add_argument("--test-shards", type=int, default=22,
                        help="số shard test để KHÓA làm test set (mỗi shard ~900 ảnh)")
    parser.add_argument("--valid-shards", type=int, default=4,
                        help="số shard valid (~5.2k ảnh/shard)")
    parser.add_argument("--out", default=str(OUT_ROOT))
    return parser.parse_args()


def list_shards() -> dict[str, list[str]]:
    """Lấy danh sách URL các shard parquet theo từng split từ datasets-server."""
    resp = requests.get(API_ROOT, timeout=60)
    resp.raise_for_status()
    return resp.json()["default"]


def crop_face(image: Image.Image, bbox) -> Image.Image:
    """Crop vùng mặt theo bbox [x1, y1, x2, y2], clamp vào biên ảnh + thêm lề."""
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return image

    width, height = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    dw, dh = x2 - x1, y2 - y1
    if dw <= 0 or dh <= 0:
        return image

    x1 = max(0, int(x1 - dw * MARGIN))
    y1 = max(0, int(y1 - dh * MARGIN))
    x2 = min(width, int(x2 + dw * MARGIN))
    y2 = min(height, int(y2 + dh * MARGIN))
    return image.crop((x1, y1, x2, y2))


def image_bytes(cell) -> bytes:
    """Trích bytes ảnh từ ô parquet (struct {bytes, path} hoặc bytes thô)."""
    if isinstance(cell, dict):
        for key in ("bytes", "image", "data"):
            if key in cell:
                return cell[key]
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise ValueError(f"Không nhận dạng được ô ảnh: {type(cell)!r}")


def process_shard(split: str, url: str, shard_index: int, total_shards: int,
                  start_idx: int, image_dir: Path, lines: list[str]) -> tuple[int, int]:
    """Tải + xử lý 1 shard parquet; trả về (số ảnh mới, số ảnh bị bỏ qua).

    Log đầy đủ tiến trình để người chạy biết đang tải đến đâu.
    """
    label = f"[{split} {shard_index + 1}/{total_shards}]"
    t0 = time.time()

    print(f"{label} Đang tải {url.rsplit('/', 1)[-1]} ...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "shard.parquet"
        size_mb = 0
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in r.iter_content(1 << 20):
                    handle.write(chunk)
                    size_mb += len(chunk)
        print(f"{label} Tải xong ({size_mb / 1e6:.1f} MB trong "
              f"{time.time() - t0:.0f}s). Đang crop ảnh ...", flush=True)

        idx = start_idx
        skipped = 0
        pf = pq.ParquetFile(tmp)
        try:
            for batch in pf.iter_batches(batch_size=512):
                df = batch.to_pandas()
                for i in range(len(df)):
                    cls = df["Class"].iloc[i]
                    label_value = 0 if cls == "live" else 1
                    raw = image_bytes(df["Filepath"].iloc[i])
                    try:
                        image = Image.open(io.BytesIO(raw))
                        image.load()
                        bbox = df["Bbox"].iloc[i]
                        image = crop_face(image, bbox).convert("RGB")
                    except Exception as exc:
                        skipped += 1
                        continue

                    name = f"{idx:06d}.jpg"
                    image.save(image_dir / name, "JPEG", quality=90)
                    lines.append(f"{name} {label_value}")
                    idx += 1
        finally:
            pf.close()

    elapsed = time.time() - t0
    print(f"{label} Xong: {idx - start_idx} ảnh ({skipped} bị bỏ qua) "
          f"trong {elapsed:.0f}s | tổng {idx} ảnh", flush=True)
    return idx, skipped


def main() -> None:
    """Tải shard -> crop -> ghi list files -> lưu splits cố định + báo cáo."""
    args = parse_args()

    out_root = Path(args.out)
    image_dir = out_root / "SpoofingData"
    image_dir.mkdir(parents=True, exist_ok=True)

    shards = list_shards()
    plan = {
        "train": shards.get("train", [])[:args.train_shards],
        "test": shards.get("test", [])[:args.test_shards],
        "valid": shards.get("valid", [])[:args.valid_shards],
    }
    total = sum(len(urls) for urls in plan.values())
    print(f"=== Tải CelebA-Spoof từ {REPO} ===")
    print(f"Kế hoạch: train={len(plan['train'])} shard, test={len(plan['test'])} "
          f"shard, valid={len(plan['valid'])} shard (tổng {total} shard)")
    print(f"Lưu vào: {out_root} | Có thể mất 1-3 giờ tùy mạng.\n", flush=True)

    all_lines: dict[str, list[str]] = {"train": [], "test": [], "valid": []}
    counts: dict[str, int] = {"train": 0, "test": 0, "valid": 0}
    split_order = ("train", "test", "valid")
    start_idx = 0
    t_start = time.time()

    for split_pos, split in enumerate(split_order):
        urls = plan[split]
        for i, url in enumerate(urls):
            start_idx, skipped = process_shard(
                split, url, i, len(urls), start_idx, image_dir,
                all_lines[split],
            )
            counts[split] = len(all_lines[split])
            done = sum(len(plan[k]) for k in split_order[:split_pos]) + i + 1
            elapsed = time.time() - t_start
            per_shard = elapsed / done
            eta_min = per_shard * (total - done) / 60
            print(f"    Tiến độ chung: {done}/{total} shard | đã chạy "
                  f"{elapsed / 60:.0f} phút | ETA còn ~{eta_min:.0f} phút\n",
                  flush=True)

    # Ghi các file danh sách (format <tên_ảnh> <nhãn>, nhãn 0=live 1=spoof).
    for split in ("train", "test", "valid"):
        path = out_root / f"{split}_list.txt"
        path.write_text("\n".join(all_lines[split]) + ("\n" if all_lines[split] else ""),
                        encoding="utf-8")

    # Thống kê EDA cơ bản -> dataset_report.json
    report = {
        "source": REPO,
        "created_by": "scripts/download_celeba_full.py",
        "split_protocol": "official train/valid/test split cua mirror "
                          "(subject-disjoint theo protocol chinh thuc)",
        "shards": {k: len(v) for k, v in plan.items()},
        "counts": {k: len(v) for k, v in all_lines.items()},
        "label_distribution": {
            k: {
                "total": len(v),
                "live": sum(1 for line in v if line.endswith(" 0")),
                "spoof": sum(1 for line in v if line.endswith(" 1")),
            }
            for k, v in all_lines.items()
        },
    }
    (out_root / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Lưu splits cố định cho experiments (tái dùng src/data.py).
    from src.data import Sample, save_splits

    image_root = image_dir.resolve()
    splits = {}
    # save_splits yêu cầu đúng 3 key: train / val / test.
    key_map = {"train": "train", "valid": "val", "test": "test"}
    for split in ("train", "valid", "test"):
        samples = []
        for line in all_lines[split]:
            name, label = line.split()
            samples.append(
                Sample(
                    path=str(image_root / name),
                    label=int(label),
                    subject_id=Path(name).stem,
                    attack_type="bona_fide" if label == "0" else "photo",
                    metadata={"split_source": f"{REPO} {split}"},
                )
            )
        splits[key_map[split]] = samples

    meta = {
        "seed": 123,
        "strategy": "subject_disjoint",
        "note": "Dung split train/valid/test chinh thuc cua mirror (subject-disjoint"
                " theo protocol dataset); khong chia lai bang create_splits.",
        "val_ratio": None,
        "test_ratio": None,
        "total": sum(len(v) for v in splits.values()),
        "counts": {k: len(v) for k, v in splits.items()},
        "source": REPO,
    }
    splits["meta"] = meta
    split_path = Path("data/splits") / "celeba_spoof_full_seed123_subject_disjoint.json"
    save_splits(splits, split_path)

    total_min = (time.time() - t_start) / 60
    print("\n=== HOÀN TẤT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Tổng thời gian: {total_min:.0f} phút")
    print(f"Splits đã lưu: {split_path}")
    print("Bước tiếp theo: python -m experiments.train_baseline "
          "--config configs/full_clean.yaml")


if __name__ == "__main__":
    main()
