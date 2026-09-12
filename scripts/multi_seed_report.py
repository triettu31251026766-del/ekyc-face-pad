"""scripts/multi_seed_report.py — gộp kết quả 5 seed (mean ± SD) + paired bootstrap CI.

Tệp này dùng để (theo protocol advisor Phase 13-15):
- Đọc kết quả của 5 training seed (E01 baseline + E07 robust).
- Báo cáo mean ± SD cho clean metrics, 16 degradation conditions, aggregate, worst-case.
- Paired bootstrap 95% CI (5,000 resamples) cho ΔF1 / ΔACER / ΔAUC giữa E01 và E07
  trên CÙNG fixed test samples.

CHẠY SAU khi đã train xong 10 runs + đánh giá (xem handover/PROTOCOL_THUC_HIEN.md).

Cách dùng:
    python -m scripts.multi_seed_report --seeds 123 456 789 101 202
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAW = Path("results/raw")
TABLES = Path("results/tables")
OUT = Path("results/multiseed")

CLEAN_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                 "apcer", "bpcer", "acer"]
DEG_METRICS = ["f1", "roc_auc", "apcer", "bpcer", "acer"]


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def clean_summary(seeds: list[int]) -> None:
    print("\n=== CLEAN TEST (mean ± SD over seeds, frozen threshold) ===")
    rows = []
    for exp_prefix in ["E01_baseline", "E07_robust"]:
        for metric in CLEAN_METRICS:
            vals = []
            for seed in seeds:
                f = RAW / f"{exp_prefix}_seed{seed}_clean.json"
                if not f.is_file():
                    f = RAW / f"{exp_prefix}_seed{seed}.json"
                if f.is_file():
                    v = json.load(open(f, encoding="utf-8"))[metric]
                    if v is not None:
                        vals.append(v)
            if vals:
                m, sd = _mean_sd(vals)
                rows.append({"exp": exp_prefix, "metric": metric,
                             "mean": round(m, 4), "sd": round(sd, 4)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT / "clean_multiseed.csv", index=False)


def degradation_summary(seeds: list[int]) -> None:
    print("\n=== 16 DEGRADATION (mean ± SD over seeds) ===")
    out_rows = []
    for tag in ["baseline", "robust"]:
        for seed in seeds:
            f = TABLES / f"degradation_{tag}_seed{seed}.csv"
            if not f.is_file():
                continue
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                if row["condition"] == "clean":
                    continue
                out_rows.append({
                    "tag": tag, "seed": seed,
                    "condition": row["condition"], "severity": row["severity"],
                    **{m: row[m] for m in DEG_METRICS},
                })
    if not out_rows:
        print("  CHUA CO per-seed degradation tables (chay grid truoc).")
        return
    df = pd.DataFrame(out_rows)
    grp = df.groupby(["tag", "condition", "severity"])
    summary = grp[DEG_METRICS].agg(["mean", "std"]).round(4)
    summary.to_csv(OUT / "degradation_multiseed.csv")
    # aggregate over 16 conditions per seed
    agg_rows = []
    for (tag, seed), g in df.groupby(["tag", "seed"]):
        agg_rows.append({"tag": tag, "seed": seed,
                         **{f"mean_{m}": g[m].mean() for m in DEG_METRICS}})
    agg = pd.DataFrame(agg_rows)
    print("  mean over 16 conditions (per seed):")
    print(agg.groupby("tag")[[f"mean_{m}" for m in DEG_METRICS]].agg(["mean", "std"]).round(4))
    agg.to_csv(OUT / "aggregate_multiseed.csv", index=False)


def paired_bootstrap_ci(y, p1, p2, metric, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        if metric == "auc":
            diffs[b] = roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx])
        elif metric == "f1":
            diffs[b] = _f1(y[idx], p1[idx]) - _f1(y[idx], p2[idx])
        else:
            diffs[b] = _acer(y[idx], p1[idx]) - _acer(y[idx], p2[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def _f1(y, pred):
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _acer(y, pred):
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    ap = fn / (fn + tp) if (fn + tp) else 0.0
    bp = fp / (fp + tn) if (fp + tn) else 0.0
    return (ap + bp) / 2


def bootstrap_ci(seeds: list[int]) -> None:
    print("\n=== PAIRED BOOTSTRAP 95% CI (B=5000, frozen threshold) ===")
    for seed in seeds:
        b1 = RAW / f"E01_baseline_seed{seed}_clean_predictions.csv"
        b2 = RAW / f"E07_robust_seed{seed}_clean_predictions.csv"
        if not b1.is_file() or not b2.is_file():
            print(f"  seed {seed}: thiếu clean predictions (chạy eval_clean --threshold trước)")
            continue
        df1 = pd.read_csv(b1); df2 = pd.read_csv(b2)
        y = df1["label"].values
        # prediction = đã threshold tại frozen threshold của từng model.
        pred1 = df1["prediction"].values
        pred2 = df2["prediction"].values
        # AUC là threshold-free -> dùng probability_spoof.
        prob1 = df1["probability_spoof"].values
        prob2 = df2["probability_spoof"].values
        for metric in ["f1", "acer", "auc"]:
            if metric == "auc":
                m, lo, hi = paired_bootstrap_ci(y, prob1, prob2, metric)
            else:
                m, lo, hi = paired_bootstrap_ci(y, pred1, pred2, metric)
            print(f"  seed {seed} d{metric.upper()}: {m:+.4f} [{lo:+.4f},{hi:+.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[123, 456, 789, 101, 202])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    clean_summary(args.seeds)
    degradation_summary(args.seeds)
    bootstrap_ci(args.seeds)


if __name__ == "__main__":
    main()
