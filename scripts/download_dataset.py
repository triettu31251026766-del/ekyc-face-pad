"""scripts/download_dataset.py — tải subset CelebA-Spoof từ HuggingFace và chuyển về cấu trúc dự án.

Tệp này dùng để:
- Tải parquet của dataset "Camilotabares1/celebA_spoof_sample_split" (18k ảnh,
  đã có nhãn live/spoof) về máy — nhẹ hơn nhiều so với full CelebA-Spoof (74GB).
- Crop mặt theo bounding box (Bbox) và lưu ảnh vào data/raw/celeba_spoof/SpoofingData/.
- Sinh train_list.txt theo định dạng <tên_ảnh> <nhãn_thô> (0 = live, 1 = spoof)
  để module src/data.py đọc được.

Chú ý: mirror này KHÔNG có subject_id, nên mỗi ảnh được đặt tên không chứa "_"
để src/data.py coi mỗi ảnh là một subject riêng (split subject_disjoint vẫn đúng,
không rò rỉ dữ liệu).

Cách dùng:
    python scripts/download_dataset.py              # tải đủ 18k ảnh
    python scripts/download_dataset.py --max 100    # tải thử 100 ảnh
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = "Camilotabares1/celebA_spoof_sample_split"
API_ROOT = f"https://huggingface.co/api/datasets/{REPO}/parquet"

OUT_ROOT = Path("data/raw/celeba_spoof")
IMAGE_DIR = OUT_ROOT / "SpoofingData"
ANNOTATION = OUT_ROOT / "train_list.txt"

MARGIN = 0.1  # mở rộng bbox 10% quanh mặt để crop không quá sát.


def _list_parquet_files() -> list[tuple[str, str]]:
    """Trả danh sách (split, url) của các tệp parquet từ datasets-server."""
    resp = requests.get(API_ROOT, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    files: list[tuple[str, str]] = []
    for split, urls in data.get("default", {}).items():
        for url in urls:
            files.append((split, url))
    return files


def _crop_face(image: Image.Image, bbox) -> Image.Image:
    """Crop vùng mặt theo bbox [x1, y1, x2, y2], clamp vào biên ảnh và thêm lề."""
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return image

    width, height = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    dw = x2 - x1
    dh = y2 - y1
    if dw <= 0 or dh <= 0:
        return image

    x1 = max(0, int(x1 - dw * MARGIN))
    y1 = max(0, int(y1 - dh * MARGIN))
    x2 = min(width, int(x2 + dw * MARGIN))
    y2 = min(height, int(y2 + dh * MARGIN))
    return image.crop((x1, y1, x2, y2))


def _image_bytes(cell):
    """Trích bytes ảnh từ ô parquet (struct {bytes, path} hoặc bytes thô)."""
    if isinstance(cell, dict):
        for key in ("bytes", "image", "data"):
            if key in cell:
                return cell[key]
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise ValueError(f"Không nhận dạng được ô ảnh: {type(cell)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải và chuyển đổi CelebA-Spoof subset.")
    parser.add_argument("--max", type=int, default=0, help="Giới hạn số ảnh (0 = tất cả).")
    args = parser.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    files = _list_parquet_files()
    print(f"Tìm thấy {len(files)} tệp parquet: "
          f"{', '.join(f'{s}({n})' for s, n in [(s, u.rsplit('/', 1)[-1]) for s, u in files])}")

    lines: list[str] = []
    idx = 0
    stop = False

    with tempfile.TemporaryDirectory() as tmpdir:
        for split, url in files:
            if stop:
                break
            print(f"Tải {split}: {url.rsplit('/', 1)[-1]} ...", flush=True)
            tmp = Path(tmpdir) / f"{split}_{url.rsplit('/', 1)[-1]}"
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in r.iter_content(1 << 20):
                        handle.write(chunk)

            pf = pq.ParquetFile(tmp)
            print(f"  Đọc {pf.metadata.num_rows} dòng ...", flush=True)
            try:
                for batch in pf.iter_batches(batch_size=512):
                    df = batch.to_pandas()
                    for i in range(len(df)):
                        if args.max and idx >= args.max:
                            stop = True
                            break
                        cls = df["Class"].iloc[i]
                        label = 0 if cls == "live" else 1
                        raw = _image_bytes(df["Filepath"].iloc[i])
                        try:
                            image = Image.open(io.BytesIO(raw))
                            image.load()
                            bbox = df["Bbox"].iloc[i]
                            image = _crop_face(image, bbox).convert("RGB")
                        except Exception as exc:
                            print(f"  Bỏ qua ảnh thứ {idx} (lỗi {exc})", flush=True)
                            continue

                        name = f"{idx:06d}.jpg"
                        image.save(IMAGE_DIR / name, "JPEG", quality=90)
                        lines.append(f"{name} {label}")
                        idx += 1
                    if stop:
                        break
            finally:
                pf.close()

    ANNOTATION.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    num_spoof = sum(1 for line in lines if line.endswith(" 1"))
    print(f"\nHoàn tất: {idx} ảnh (live={idx - num_spoof}, spoof={num_spoof}) "
          f"-> {IMAGE_DIR}")
    print(f"Đã ghi {ANNOTATION}")


if __name__ == "__main__":
    main()
