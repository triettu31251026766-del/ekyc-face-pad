"""scripts/reset_seed.py — xóa kết quả của một seed (theo model) để train lại từ đầu.

Tệp này dùng khi kết quả của một seed bị sai (ví dụ train nhầm split) và cần
chạy lại. Xóa đúng các file của seed đó:
    - results/checkpoints/{model}_*_seed<N>.pt
    - results/thresholds/{model}_seed<N>.json
    - results/raw/*_seed<N>*  (json/csv/log/predictions của model + lưới suy giảm)
    - results/tables/degradation_{tag}_seed<N>.csv
    - results/figures/fig_{tag}_seed<N>_*.png

Chọn model:
    --model both (mặc định) | E01 (baseline) | E07 (robust)

KHÔNG đụng tới seed/model khác. Mặc định chỉ LIỆT KÊ (dry-run); --yes để xóa thật.

Cách dùng:
    python -m scripts.reset_seed --seed 456                    # xem trước (cả 2 model)
    python -m scripts.reset_seed --seed 101 --model E07        # chỉ xóa E07 của seed 101
    python -m scripts.reset_seed --seed 456 --yes              # xóa thật
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Prefix experiment của lưới suy giảm theo model (E02-E06 = baseline, E08-E12 = robust).
DEG_PREFIXES = {
    "E01": ["E02", "E03", "E04", "E05", "E06"],
    "E07": ["E08", "E09", "E10", "E11", "E12"],
}
TAGS = {"E01": "baseline", "E07": "robust"}


def _targets(seed: int, model: str) -> list[Path]:
    models = ["E01", "E07"] if model == "both" else [model]
    found: set[Path] = set()
    for mid in models:
        tag = TAGS[mid]
        found |= set(Path("results/checkpoints").glob(f"{mid}_*_seed{seed}.pt"))
        found |= set(Path("results/thresholds").glob(f"{mid}_seed{seed}.json"))
        found |= set(Path("results/tables").glob(f"degradation_{tag}_seed{seed}.csv"))
        found |= set(Path("results/figures").glob(f"fig_{tag}_seed{seed}_*.png"))
        # Raw: file của model (E01/E07) + các thí nghiệm suy giảm tương ứng.
        for prefix in [mid, *DEG_PREFIXES[mid]]:
            found |= set(Path("results/raw").glob(f"{prefix}_*_seed{seed}*"))
    return sorted(p for p in found if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Xóa kết quả của một seed (theo model)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", choices=["both", "E01", "E07"], default="both")
    parser.add_argument("--yes", action="store_true", help="xác nhận xóa thật")
    args = parser.parse_args()

    targets = _targets(args.seed, args.model)
    if not targets:
        print(f"seed {args.seed} (model={args.model}): không có file nào để xóa.")
        return 0

    print(f"seed {args.seed} (model={args.model}): {len(targets)} file:")
    for path in targets:
        print(f"  - {path}")

    if not args.yes:
        print("\n(dry-run) chưa xóa gì. Thêm --yes để xóa thật.")
        return 0

    for path in targets:
        path.unlink()
    print(f"\n[OK] đã xóa {len(targets)} file của seed {args.seed} (model={args.model}).")
    print("Bước tiếp theo: python -m scripts.pad_app auto  (sẽ train lại phần đã xóa)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
