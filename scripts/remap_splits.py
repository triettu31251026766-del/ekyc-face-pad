"""scripts/remap_splits.py — đổi đường dẫn ảnh trong file split sang dataset root của máy hiện tại.

Tệp này dùng khi copy file split từ máy khác sang: file split lưu đường dẫn
TUYỆT ĐỐI của máy cũ, nên trên máy mới cần remap về dataset root của máy mới.
Nội dung split (subject_id + label) GIỮ NGUYÊN — fingerprint không đổi.

Cách dùng (tại thư mục gốc dự án, sau khi đã copy file split vào data/splits/):
    python -m scripts.remap_splits --split data/splits/celeba_spoof_full_seed123_subject_disjoint.json \
        --dataset-root data/raw/celeba_spoof_full

Sau đó xác minh:
    python -m scripts.check_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description="Remap đường dẫn trong file split")
    parser.add_argument("--split", required=True, help="file split cần remap (JSON)")
    parser.add_argument("--dataset-root", required=True,
                        help="dataset root của máy hiện tại (vd data/raw/celeba_spoof_full)")
    parser.add_argument("--image-subdir", default="SpoofingData",
                        help="thư mục con chứa ảnh trong dataset root")
    args = parser.parse_args()

    split_path = Path(args.split)
    if not split_path.is_file():
        print(f"[FAIL] không thấy file split: {split_path}")
        return 1

    image_dir = Path(args.dataset_root) / args.image_subdir
    if not image_dir.is_dir():
        print(f"[FAIL] không thấy thư mục ảnh: {image_dir}")
        return 1
    image_dir = image_dir.resolve()

    data = json.loads(split_path.read_text(encoding="utf-8"))
    splits = data.get("splits", {})
    changed = missing = total = 0
    for name in ("train", "val", "test"):
        for sample in splits.get(name, []):
            total += 1
            old = Path(str(sample.get("path", "")))
            new = image_dir / old.name
            if str(new) != str(old):
                changed += 1
            if not new.is_file():
                missing += 1
            sample["path"] = str(new)

    if missing:
        print(f"[FAIL] {missing:,}/{total:,} ảnh không tồn tại trong {image_dir}")
        print("       -> dataset máy này thiếu ảnh hoặc khác cấu trúc; KHÔNG ghi file split.")
        return 1

    split_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] đã remap {changed:,}/{total:,} đường dẫn -> {image_dir}")
    print(f"     file split: {split_path}")
    print("     Bước tiếp theo: python -m scripts.check_data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
