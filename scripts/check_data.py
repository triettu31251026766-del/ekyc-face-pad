"""scripts/check_data.py — kiểm tra dataset + split TRƯỚC khi train (chống lệch dữ liệu giữa các máy).

Tệp này dùng để:
1. Kiểm tra dataset tồn tại và đủ số ảnh (mặc định: 244,030 ảnh của celeba_spoof_full).
2. Kiểm tra file split tồn tại + số lượng train/val/test đúng.
3. So "dấu vân tay" (SHA-256 trên subject_id + label) của split hiện tại với
   `configs/split_manifest.json` đã commit — đảm bảo MỌI MÁY dùng CÙNG một split.

Nếu bước 3 lệch => DỪNG, không train, vì kết quả sẽ không so sánh được.

Cách dùng:
    python -m scripts.check_data
    python -m scripts.check_data --config configs/full_clean.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import load_config
from src.data import load_splits, splits_fingerprint

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANIFEST_PATH = Path("configs/split_manifest.json")


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra dataset + split trước khi train")
    parser.add_argument("--config", default="configs/full_clean.yaml")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset"]["root"])
    dataset_name = config["dataset"]["name"]
    split_seed = int(config["split"].get("seed", config["seed"]))
    strategy = config["split"]["strategy"]
    split_file = Path("data/splits") / f"{dataset_name}_seed{split_seed}_{strategy}.json"

    print("=== KIỂM TRA DỮ LIỆU ===")
    print(f"config     : {args.config}")
    print(f"dataset    : {dataset_root}")
    print(f"split file : {split_file}")
    print()

    failures = 0

    # --- 1. Dataset ---
    if not dataset_root.is_dir():
        _fail(f"dataset root không tồn tại: {dataset_root}")
        print("       -> chạy: python -m scripts.download_celeba_full")
        return 1
    image_dir = dataset_root / "SpoofingData"
    if not image_dir.is_dir():
        _fail(f"không thấy thư mục ảnh: {image_dir}")
        return 1
    n_images = sum(1 for _ in image_dir.glob("*.jpg"))
    _ok(f"dataset có {n_images:,} ảnh .jpg")

    # --- 2. Split file ---
    if not split_file.is_file():
        _fail(f"KHÔNG có file split: {split_file}")
        print("       -> chạy: python -m scripts.download_celeba_full (tạo dataset + split)")
        print("       -> hoặc copy file split từ máy đã train (cùng nội dung)")
        return 1
    splits = load_splits(split_file)
    fp = splits_fingerprint(splits)
    _ok(f"split counts: train={fp['counts']['train']:,} "
        f"val={fp['counts']['val']:,} test={fp['counts']['test']:,}")

    # --- 3. So với manifest ---
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        _fail(f"không thấy manifest: {manifest_path} (không xác minh được split)")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("counts") != fp["counts"]:
        _fail(f"counts KHÁC manifest:\n"
              f"         hiện tại : {fp['counts']}\n"
              f"         manifest : {manifest.get('counts')}")
        failures += 1
    else:
        _ok("counts khớp manifest")

    if manifest.get("sha256") != fp["sha256"]:
        _fail("SHA-256 KHÁC manifest -> split hiện tại KHÔNG phải split chuẩn!")
        for name in ("train", "val", "test"):
            cur = fp["sha256"][name]
            exp = manifest.get("sha256", {}).get(name)
            if cur != exp:
                print(f"         {name}: hiện tại {cur[:16]}... | manifest {str(exp)[:16]}...")
        failures += 1
    else:
        _ok("SHA-256 khớp manifest -> split CHUẨN")

    expected_images = manifest.get("expected_total_images")
    if expected_images is not None and n_images != expected_images:
        _fail(f"số ảnh dataset ({n_images:,}) khác manifest ({expected_images:,})")
        failures += 1
    elif expected_images is not None:
        _ok(f"số ảnh khớp manifest ({expected_images:,})")

    print()
    if failures:
        print("=== KẾT LUẬN: KHÔNG ĐẠT — DỪNG, không train với dữ liệu này. ===")
        return 1
    print("=== KẾT LUẬN: ĐẠT — dataset + split hợp lệ, sẵn sàng train. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
