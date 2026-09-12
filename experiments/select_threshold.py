"""experiments/select_threshold.py — chọn threshold trên VALIDATION (min ACER).

Tệp này dùng để (theo protocol advisor, Phase 7):
- Nạp checkpoint (best validation-loss) -> dự đoán trên VALIDATION SET (KHÔNG test).
- Quét threshold, chọn threshold tối thiểu hóa ACER trên validation.
- Tie-break: threshold gần 0.5 nhất; nếu vẫn hòa thì chọn threshold nhỏ hơn.
- Lưu threshold JSON (per model/seed) — sau đó FREEZE và dùng cho clean test
  + toàn bộ 16 degradation conditions.

QUAN TRỌNG: KHÔNG được nhìn test set ở bước này.

Cách dùng:
    python -m experiments.select_threshold \
        --checkpoint results/checkpoints/E01_baseline_seed123.pt \
        --out results/thresholds/E01_seed123.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments._common import load_checkpoint, load_val_loader
from src.evaluate import evaluate_model
from src.utils import resolve_device


def select_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    """Tìm threshold min ACER trên validation.

    Returns:
        (threshold, acer, apcer, bpcer) tại threshold được chọn.
    """
    labels = np.asarray(labels)
    best_key = None
    best = None
    for t in np.arange(0.005, 0.995, 0.005):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        apcer = fn / (fn + tp) if (fn + tp) else 0.0
        bpcer = fp / (fp + tn) if (fp + tn) else 0.0
        acer = (apcer + bpcer) / 2.0
        # Tie-break: (1) gần 0.5 nhất, (2) threshold nhỏ hơn.
        key = (round(acer, 10), abs(t - 0.5), t)
        if best_key is None or key < best_key:
            best_key = key
            best = (float(t), float(acer), float(apcer), float(bpcer))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Chọn threshold trên validation set")
    parser.add_argument("--checkpoint", required=True,
                        help="đường dẫn checkpoint (best validation-loss)")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out", required=True,
                        help="đường dẫn file JSON lưu threshold")
    args = parser.parse_args()

    device = resolve_device("auto")
    model, config, _ = load_checkpoint(Path(args.checkpoint), device)
    val_loader, _ = load_val_loader(config, args.splits_dir)

    result = evaluate_model(model, val_loader, device=device, threshold=0.5)
    probs = np.asarray(result["probabilities"])
    labels = np.asarray(result["labels"])

    threshold, acer, apcer, bpcer = select_threshold(probs, labels)

    out = {
        "checkpoint": str(Path(args.checkpoint)),
        "seed": config["seed"],
        "threshold": threshold,
        "validation_acer": acer,
        "validation_apcer": apcer,
        "validation_bpcer": bpcer,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"threshold={threshold:.3f} (val ACER={acer:.4f}, APCER={apcer:.4f}, "
          f"BPCER={bpcer:.4f}) -> {out_path}")


if __name__ == "__main__":
    main()
