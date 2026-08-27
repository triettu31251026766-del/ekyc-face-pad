"""experiments/ablation.py — thí nghiệm ablation (E09+): từng phép tăng cường riêng lẻ.

Tệp này dùng để (theo mục 31, 32 của tài liệu kỹ thuật):
Pipeline:
    run/read individual augmentation variants -> compare metrics
    -> save ablation table -> figure

Ablation: huấn luyện model robustness với TỪNG phép tăng cường riêng lẻ
(chỉ jpeg / chỉ resize / chỉ blur / chỉ noise / chỉ brightness) rồi so sánh
với baseline (không tăng cường) và robust đầy đủ (tất cả phép) — trả lời câu
hỏi phép tăng cường nào đóng góp nhiều nhất (mục 32: "E09+ Ablation").

Đầu ra:
    results/tables/ablation_table.csv
    results/figures/fig_ablation.png

Chạy:
    python experiments/ablation.py \
        --config configs/clean.yaml \
        --robustness configs/robustness.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments import train_robust
from experiments.compare_models import load_results
from src.config import load_config
from src.utils import get_experiment_logger

AUGMENTATION_NAMES = ("jpeg", "resize", "blur", "noise", "brightness")

logger = get_experiment_logger("ablation")


def run_variants(
    base_config: dict,
    robustness_config: dict,
    variants: tuple[str, ...] = AUGMENTATION_NAMES,
    splits_dir: str | Path = "data/splits",
    results_dir: str | Path = "results/raw",
    checkpoints_dir: str | Path = "results/checkpoints",
) -> list[dict]:
    """Huấn luyện các biến thể ablation: MỖI lần chỉ bật 1 phép tăng cường.

    Mỗi biến thể dùng lại đúng splits/base config của baseline (mục 24:
    so sánh công bằng). Trả về danh sách record của các biến thể đã chạy.
    """
    seed = base_config["seed"]
    records = []

    for name in variants:
        spec = robustness_config["robustness"]["augmentations"].get(name)
        if not spec:
            logger.warning(f"bỏ qua ablation '{name}': không có spec trong cấu hình robustness")
            continue

        # Chỉ bật ĐÚNG MỘT phép tăng cường với probability 1.0 (biến số duy nhất).
        single = {
            "robustness": {
                "enabled": True,
                "augmentations": {
                    other: {**other_spec, "enabled": False}
                    for other, other_spec in robustness_config["robustness"]["augmentations"].items()
                },
            }
        }
        single["robustness"]["augmentations"][name] = {**spec, "enabled": True, "probability": 1.0}

        variant_config = {**base_config, "experiment_id": f"E09_ablation_{name}_seed{seed}"}
        logger.info(f"===== ablation variant: {name} =====")
        record = train_robust.run(
            variant_config,
            single,
            splits_dir=splits_dir,
            results_dir=results_dir,
            checkpoints_dir=checkpoints_dir,
        )
        records.append(record)

    return records


def _variant_label(record: dict) -> str | None:
    """Gán nhãn biến thể cho bảng ablation:
    "jpeg"/"resize"/... cho ablation, "all" cho robust đầy đủ, "baseline" cho E01.
    """
    experiment_id = str(record.get("experiment_id", ""))
    for name in AUGMENTATION_NAMES:
        if f"ablation_{name}" in experiment_id:
            return name
    if record.get("degradation_name") != "none":
        return None  # thí nghiệm đánh giá suy giảm không thuộc bảng ablation
    if "robust" in experiment_id:
        return "all"
    if record.get("training_mode") == "clean":
        return "baseline"
    return None


def ablation_table(records: list[dict]) -> pd.DataFrame:
    """Dựng bảng ablation: mỗi biến thể 1 dòng (f1, roc_auc, apcer, bpcer, acer)."""
    rows = []
    for record in records:
        label = _variant_label(record)
        if label is None:
            continue
        rows.append(
            {
                "experiment_id": record["experiment_id"],
                "variant": label,
                "f1": record.get("f1"),
                "roc_auc": record.get("roc_auc"),
                "apcer": record.get("apcer"),
                "bpcer": record.get("bpcer"),
                "acer": record.get("acer"),
            }
        )
    return pd.DataFrame(rows, columns=[
        "experiment_id", "variant", "f1", "roc_auc", "apcer", "bpcer", "acer",
    ])


def plot_ablation(records: list[dict], out_path: str | Path) -> bool:
    """Vẽ fig_ablation.png: F1/ACER theo từng biến thể tăng cường."""
    frame = ablation_table(records)
    if frame.empty:
        return False

    # Thứ tự hiển thị cố định: baseline -> từng phép -> all.
    order = ["baseline", *AUGMENTATION_NAMES, "all"]
    frame["variant"] = pd.Categorical(frame["variant"], categories=order, ordered=True)
    frame = frame.sort_values("variant")

    labels = frame["variant"].tolist()
    x = range(len(frame))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar([pos - width / 2 for pos in x], frame["f1"], width, label="F1")
    plt.bar([pos + width / 2 for pos in x], frame["acer"], width, label="ACER")
    plt.xticks(list(x), labels)
    plt.ylim(0, 1.05)
    plt.ylabel("giá trị metric (0-1)")
    plt.title("Ablation: đóng góp của từng phép tăng cường chất lượng (mục 32)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def run(
    base_config: dict,
    robustness_config: dict,
    splits_dir: str | Path = "data/splits",
    results_dir: str | Path = "results/raw",
    checkpoints_dir: str | Path = "results/checkpoints",
    tables_dir: str | Path = "results/tables",
    figures_dir: str | Path = "results/figures",
    variants: tuple[str, ...] = AUGMENTATION_NAMES,
) -> dict:
    """Chạy ablation đầy đủ: train từng biến thể -> bảng CSV -> figure.

    Bảng bao gồm: biến thể vừa huấn luyện + baseline (E01) + robust đầy đủ
    (E07) nếu đã có record trong results_dir (không huấn luyện lại — mục 41).
    """
    variant_records = run_variants(
        base_config, robustness_config, variants=variants,
        splits_dir=splits_dir, results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
    )

    existing = load_results(results_dir)
    known_ids = {record["experiment_id"] for record in variant_records}
    records = variant_records + [record for record in existing
                                 if record["experiment_id"] not in known_ids]

    table = ablation_table(records)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "ablation_table.csv"
    table.to_csv(table_path, index=False)
    logger.info(f"bảng ablation: {table_path} ({len(table)} dòng)")

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "fig_ablation.png"
    plot_ablation(records, figure_path)

    return {"variants": len(variant_records), "table": str(table_path),
            "figure": str(figure_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="E09+: ablation từng phép tăng cường")
    parser.add_argument("--config", default="configs/clean.yaml",
                        help="đường dẫn tệp cấu hình huấn luyện cơ sở")
    parser.add_argument("--robustness", default="configs/robustness.yaml",
                        help="đường dẫn tệp cấu hình robustness")
    args = parser.parse_args()

    base_config = load_config(args.config)
    robustness_config = load_config(args.robustness)
    run(base_config, robustness_config)


if __name__ == "__main__":
    main()
