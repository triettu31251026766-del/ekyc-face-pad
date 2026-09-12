"""scripts/run_frozen_eval.py — tự động chạy eval clean + degradation grid với threshold đã freeze.

Tệp này dùng để (theo protocol advisor Phase 7-12):
1. Đọc threshold đã chọn trên VALIDATION (từ results/thresholds/<model>_seed<seed>.json).
2. Chạy eval_clean với threshold đã freeze -> kết quả clean test tại frozen threshold.
3. Chạy eval_degradation_grid với CÙNG threshold -> 16 điều kiện suy giảm tại frozen threshold.

Cách dùng (chạy SAU khi đã train + select_threshold cho 1 seed):
    python -m scripts.run_frozen_eval --seed 123 --tag baseline
    python -m scripts.run_frozen_eval --seed 123 --tag robust
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy eval clean + degradation grid với threshold đã freeze"
    )
    parser.add_argument("--seed", required=True, type=int,
                        help="training seed (của model đã train)")
    parser.add_argument("--tag", choices=["baseline", "robust"], required=True,
                        help="baseline (E01) hoặc robust (E07)")
    parser.add_argument("--config", default="configs/full_clean.yaml",
                        help="config nền (mặc định full_clean.yaml — thí nghiệm chính)")
    args = parser.parse_args()

    model_id = "E01" if args.tag == "baseline" else "E07"
    checkpoint = f"results/checkpoints/{model_id}_{args.tag}_seed{args.seed}.pt"
    threshold_json = Path(f"results/thresholds/{model_id}_seed{args.seed}.json")

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint không tồn tại: {checkpoint}")
    if not threshold_json.is_file():
        raise FileNotFoundError(
            f"threshold không tồn tại: {threshold_json}. "
            f"Hãy chạy experiments.select_threshold trước."
        )
    threshold = json.load(open(threshold_json, encoding="utf-8"))["threshold"]
    print(f"[{args.tag} seed {args.seed}] frozen threshold = {threshold:.4f}")

    cmds = [
        [sys.executable, "-m", "experiments.eval_clean",
         "--config", args.config, "--checkpoint", checkpoint,
         "--threshold", str(threshold)],
        [sys.executable, "-m", "experiments.eval_degradation_grid",
         "--checkpoint", checkpoint, "--tag", args.tag,
         "--threshold", str(threshold)],
    ]
    for cmd in cmds:
        print("RUN:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(f"lệnh thất bại (rc={result.returncode}): {' '.join(cmd)}")


if __name__ == "__main__":
    main()
