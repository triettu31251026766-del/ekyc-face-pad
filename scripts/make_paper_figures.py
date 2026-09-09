"""scripts/make_paper_figures.py — sinh các hình cho bài báo khoa học từ dữ liệu thật.

Tệp này dùng để tạo 6 hình (Fig. 1-6) phục vụ paper
"Robust Face Presentation Attack Detection for eKYC Under Image Quality
Degradations", lưu vào thư mục images/. Toàn bộ số liệu lấy từ results/ và
data/ THẬT của project — không bịa số.

Hình tạo ra:
    images/fig1_pipeline.png         sơ đồ pipeline tổng quan (vẽ minh họa)
    images/fig2_degradation_examples.png  ảnh thật: 1 live + 1 spoof qua các mức suy giảm
    images/fig3_roc_curves.png       ROC curve E01 vs E07 (từ predictions CSV)
    images/fig4_severity_curves.png  F1 + ACER theo severity (5 loại suy giảm)
    images/fig5_aggregate_bars.png   bar chart mean/worst-case ACER/F1
    images/fig6_loss_curves.png      train/val loss theo epoch

Cách dùng (chạy từ thư mục gốc dự án):
    python -m scripts.make_paper_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src.data import load_splits  # noqa: E402
from src.degradation import (  # noqa: E402
    brightness_adjustment,
    gaussian_blur,
    gaussian_noise,
    jpeg_compression,
    resize_degradation,
)

RAW = Path("results/raw")
IMG_DIR = Path("images")
IMG_DIR.mkdir(exist_ok=True)

ORDER = [
    ("jpeg", "E02", "E08", ["90", "70", "50", "30"]),
    ("resize", "E03", "E09", ["75", "50", "25"]),
    ("blur", "E04", "E10", ["light", "medium", "strong"]),
    ("noise", "E05", "E11", ["low", "medium", "high"]),
    ("brightness", "E06", "E12", ["dark", "normal", "bright"]),
]


def fig1_pipeline() -> None:
    """Fig. 1 — sơ đồ pipeline tổng quan (số liệu mới nhất: best-epoch)."""
    fig, ax = plt.subplots(figsize=(6.5, 8))
    ax.axis("off")

    boxes = [
        ("CelebA-Spoof subset\n203,215 images (70,288 live)", "tab:blue"),
        ("Subject-disjoint split, seed 123\ntrain 142,250 | val 20,322 | test 40,643", "tab:blue"),
        ("Preprocessing\nface crop (bbox + 10%)", "tab:cyan"),
        ("Resize 224x224\nImageNet normalize", "tab:cyan"),
        ("MobileNetV2\n2,225,153 params / 8.62 MB", "tab:green"),
        ("E01 Baseline\nclean + flip only", "tab:orange"),
        ("E07 Robust\n5 quality augmentations\n(p = 0.3 each)", "tab:red"),
        ("Best val-loss checkpoint\n(epoch 18, both models)", "tab:purple"),
        ("Fixed clean test\n40,643 images, threshold 0.5", "tab:purple"),
        ("16 degradation\nconditions (no retraining)", "tab:brown"),
        ("Metrics\nF1, ROC-AUC, PR-AUC\nAPCER, BPCER, ACER", "tab:gray"),
    ]
    n = len(boxes)
    ys = np.linspace(0.95, 0.05, n)
    centers = []
    for (label, color), y in zip(boxes, ys):
        ax.add_patch(mpatches.FancyBboxPatch((0.25, y - 0.045), 0.5, 0.09,
                                             boxstyle="round,pad=0.01",
                                             fc=color, ec="black", alpha=0.85))
        ax.text(0.5, y, label, ha="center", va="center", fontsize=8.5)
        centers.append((0.5, y))
    for (x1, y1), (x2, y2) in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(x2, y2 + 0.045), xytext=(x1, y1 - 0.045),
                    arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("Fig. 1. Overall pipeline", fontsize=11)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "fig1_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig2_degradation_examples() -> None:
    """Fig. 2 — ảnh live + spoof thật qua các mức suy giảm."""
    splits = load_splits("data/splits/celeba_spoof_seed123_subject_disjoint.json")
    test = splits["test"]
    live_path = next(s.path for s in test if s.label == 0)
    spoof_path = next(s.path for s in test if s.label == 1)

    variants = [
        ("clean", lambda im: im),
        ("JPEG 50", lambda im: jpeg_compression(im, 50)),
        ("JPEG 30", lambda im: jpeg_compression(im, 30)),
        ("resize 25%", lambda im: resize_degradation(im, 0.25)),
        ("blur strong", lambda im: gaussian_blur(im, 11, 3.0)),
        ("noise high", lambda im: gaussian_noise(im, 0.03, seed=123)),
        ("bright 1.4", lambda im: brightness_adjustment(im, 1.4)),
        ("dark 0.6", lambda im: brightness_adjustment(im, 0.6)),
    ]

    fig, axes = plt.subplots(2, len(variants), figsize=(12, 3.6))
    for row, (path, row_name) in enumerate(
        [(live_path, "Live (bona fide)"), (spoof_path, "Spoof")]
    ):
        base = Image.open(path).convert("RGB")
        for col, (name, fn) in enumerate(variants):
            im = fn(base)
            axes[row, col].imshow(im)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(name, fontsize=9)
        axes[row, 0].set_ylabel(row_name, fontsize=10)
    plt.suptitle("Fig. 2. Example degradations on one live and one spoof test image",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "fig2_degradation_examples.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def fig3_roc_curves() -> None:
    """Fig. 3 — ROC curves E01 vs E07 (checkpoint best-epoch) trên clean test."""
    plt.figure(figsize=(5, 4.5))
    for exp, label in [("E01_baseline_seed123_bestepoch", "Baseline (E01, best-epoch)"),
                       ("E07_robust_seed123_bestepoch", "Robust (E07, best-epoch)")]:
        df = pd.read_csv(RAW / f"{exp}_predictions.csv")
        y = df["label"].values
        p = df["probability_spoof"].values
        fpr, tpr, _ = roc_curve(y, p)
        auc_v = roc_auc_score(y, p)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc_v:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fig. 3. ROC curves on the clean 40,643-image test set")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "fig3_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def _read_grid(tag: str) -> pd.DataFrame:
    """Đọc bảng lưới suy giảm mới nhất (clean + 16 điều kiện) từ results/tables."""
    return pd.read_csv(Path("results/tables") / f"degradation_{tag}.csv")


def _grid_row(grid: pd.DataFrame, condition: str, severity: str | None) -> pd.Series:
    """Lấy hàng metric của grid theo condition/severity (severity=None = clean)."""
    frame = grid[grid["condition"] == condition]
    if severity is None:
        return frame.iloc[0]
    return frame[frame["severity"].astype(str) == severity].iloc[0]


def fig4_severity_curves() -> None:
    """Fig. 4 — F1 + ACER theo severity, E01 vs E07, 5 loại suy giảm (số mới)."""
    base = _read_grid("baseline")
    rob = _read_grid("robust")
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), sharey=True)
    for ax, (cond, _, _, sevs) in zip(axes, ORDER):
        xs = ["clean"] + sevs
        f1_b, f1_r, ac_b, ac_r = [], [], [], []
        for sev in [None] + sevs:
            b = _grid_row(base, "clean" if sev is None else cond, None if sev is None else sev)
            r = _grid_row(rob, "clean" if sev is None else cond, None if sev is None else sev)
            f1_b.append(b["f1"]); f1_r.append(r["f1"])
            ac_b.append(b["acer"]); ac_r.append(r["acer"])
        x = np.arange(len(xs))
        ax.plot(x, f1_b, "-o", color="tab:blue", label="E01 F1", lw=1.6, ms=3)
        ax.plot(x, f1_r, "-o", color="tab:red", label="E07 F1", lw=1.6, ms=3)
        ax.plot(x, ac_b, "--s", color="tab:blue", label="E01 ACER", lw=1.4, ms=3)
        ax.plot(x, ac_r, "--s", color="tab:red", label="E07 ACER", lw=1.4, ms=3)
        ax.set_xticks(x)
        ax.set_xticklabels(xs, rotation=45, fontsize=8)
        ax.set_title(cond, fontsize=10)
        ax.grid(alpha=0.3)
        if cond == "jpeg":
            ax.set_ylabel("score", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    plt.suptitle("Fig. 4. F1 and ACER versus severity (baseline vs robust)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(IMG_DIR / "fig4_severity_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def fig5_aggregate_bars() -> None:
    """Fig. 5 — bar chart tổng hợp mean/worst-case của E01 vs E07 (số mới)."""
    base = _read_grid("baseline")
    rob = _read_grid("robust")
    b = base[base["condition"] != "clean"]
    r = rob[rob["condition"] != "clean"]

    metrics = [
        ("Mean ACER", b["acer"].mean(), r["acer"].mean(), "lower is better"),
        ("Worst ACER", b["acer"].max(), r["acer"].max(), "lower is better"),
        ("Worst F1", b["f1"].min(), r["f1"].min(), "higher is better"),
    ]
    x = np.arange(len(metrics))
    w = 0.35
    plt.figure(figsize=(6, 3.8))
    plt.bar(x - w / 2, [m[1] for m in metrics], w, label="Baseline (E01)",
            color="tab:blue")
    plt.bar(x + w / 2, [m[2] for m in metrics], w, label="Robust (E07)",
            color="tab:red")
    for i, (name, v1, v2, _) in enumerate(metrics):
        plt.text(i - w / 2, v1 + 0.015, f"{v1:.4f}", ha="center", fontsize=8)
        plt.text(i + w / 2, v2 + 0.015, f"{v2:.4f}", ha="center", fontsize=8)
    plt.xticks(x, [m[0] for m in metrics], fontsize=9)
    plt.ylabel("value")
    plt.title("Fig. 5. Aggregate degradation performance (16 conditions)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "fig5_aggregate_bars.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def fig6_loss_curves() -> None:
    """Fig. 6 — train/val loss theo epoch của E01 và E07."""
    b = json.load(open(RAW / "E01_baseline_seed123.json", encoding="utf-8"))
    r = json.load(open(RAW / "E07_robust_seed123.json", encoding="utf-8"))
    epochs = list(range(1, 21))
    plt.figure(figsize=(6.5, 4))
    plt.plot(epochs, [e["train_loss"] for e in b["train_history"]], "-",
             color="tab:blue", label="E01 train", lw=1.6)
    plt.plot(epochs, [e["val_loss"] for e in b["train_history"]], "--",
             color="tab:blue", label="E01 val", lw=1.6)
    plt.plot(epochs, [e["train_loss"] for e in r["train_history"]], "-",
             color="tab:red", label="E07 train", lw=1.6)
    plt.plot(epochs, [e["val_loss"] for e in r["train_history"]], "--",
             color="tab:red", label="E07 val", lw=1.6)
    plt.plot([18, 18], [0.02, 0.24], "k:", lw=1, alpha=0.6)
    plt.text(18.3, 0.24, "best val epoch 18\nE01 0.0586 | E07 0.0681", fontsize=8)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Fig. 6. Training and validation loss per epoch")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "fig6_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


CODE_EVAL = '''\
def run(config, checkpoint_path, splits_dir="data/splits", results_dir="results/raw"):
    """Evaluate a checkpoint on the quality-degraded test set (deterministic)."""
    start = time.time()
    set_seed(config["seed"])

    degradation = config["degradation"]
    degradation_name = degradation["name"]
    # Explicit degradation parameters are recorded in the result (reproducibility).
    parameters = {k: v for k, v in degradation.items() if k != "name"}

    # Load the checkpoint (NO retraining) and move the model to the target device.
    model, checkpoint_config, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))
    device = resolve_device(checkpoint_config["device"]["name"])
    model.to(device)

    seed = checkpoint_config["seed"]
    experiment_id = config.get("experiment_id",
                               f"{degradation_name}{_short_params(parameters)}_seed{seed}")

    # Reuse the exact splits saved during training (the same fixed test set).
    strategy = checkpoint_config["split"]["strategy"]
    splits_file = (Path(splits_dir)
                   / f"{checkpoint_config['dataset']['name']}_seed{seed}_{strategy}.json")
    splits = load_splits(splits_file)

    # Deterministic evaluation: apply the degradation BEFORE the standard transform.
    eval_transform = T.Compose([
        T.Lambda(lambda image: apply_degradation_config(image, config)),
        build_eval_transform(checkpoint_config),
    ])
    test_loader = DataLoader(
        PADDataset(splits["test"], transform=eval_transform),
        batch_size=checkpoint_config["training"]["batch_size"],
        shuffle=False,   # mandatory for deterministic evaluation
        num_workers=0,
    )

    threshold = checkpoint_config["evaluation"]["threshold"]
    eval_result = evaluate_model(model, test_loader, device=device, threshold=threshold)

    # Save the full record: JSON + CSV + per-sample predictions.
    runtime = round(time.time() - start, 2)
    record = finalize(
        experiment_id, checkpoint_config, model, eval_result,
        runtime_seconds=runtime, results_dir=results_dir,
        record_extras={
            "degradation_name": degradation_name,
            "degradation_parameters": parameters,
        },
    )
    return record
'''

CODE_AUGMENT = '''\
def apply_training_quality_augmentation(image, config):
    """Apply random quality-degradation augmentations to ONE training image.

    Each enabled augmentation is applied independently with its own
    probability (0.3 in the main experiment); the severity is sampled
    uniformly from the configured range. Augmentations are considered in
    a fixed order so that results are reproducible under the same seed.
    """
    robustness = _require_robustness(config)
    if not robustness.get("enabled", False):
        return image

    augmentations = robustness.get("augmentations", {})
    for name in ("jpeg", "resize", "blur", "noise", "brightness", "crop"):
        spec = augmentations.get(name)
        if not spec or not spec.get("enabled", False):
            continue

        probability = spec.get("probability", DEFAULT_PROBABILITY)
        if random.random() >= probability:
            continue  # this augmentation is skipped for this sample

        image = _apply_one(name, image, spec)   # severity ~ U[configured range]

    return image
'''


def _render_code(code: str, out_path: Path, font_size: int = 22) -> None:
    """Vẽ ảnh code với syntax highlighting (pygments) lên nền sáng."""
    from pygments import lex
    from pygments.lexers import PythonLexer
    from pygments.styles import get_style_by_name

    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    font_path = Path("C:/Windows/Fonts/consola.ttf")
    if not font_path.exists():
        font_path = Path("C:/Windows/Fonts/cour.ttf")
    font = ImageFont.truetype(str(font_path), font_size)

    style = get_style_by_name("friendly")
    tokens = list(lex(code, PythonLexer()))

    def color_for(ttype):
        c = style.style_for_token(ttype).get("color")
        if not c:
            return "#000000"
        if not c.startswith("#"):
            return "#" + c
        return c

    def is_bold(ttype):
        return bool(style.style_for_token(ttype).get("bold"))

    line_height = font_size + 7
    pad = 26

    # Phân tokens thành các dòng.
    lines = [[]]
    for ttype, text in tokens:
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                lines.append([])
            if part:
                lines[-1].append((ttype, part))

    # Đo kích thước.
    probe = PILImage.new("RGB", (10, 10))
    draw_probe = ImageDraw.Draw(probe)
    max_w = 0
    for line in lines:
        w = 0
        for ttype, text in line:
            f = font
            try:
                if is_bold(ttype):
                    f = ImageFont.truetype(str(font_path), font_size)
            except Exception:
                pass
            w += draw_probe.textlength(text, font=f)
        max_w = max(max_w, w)

    width = int(max_w) + pad * 2
    height = line_height * len(lines) + pad * 2
    img = PILImage.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    y = pad
    for line in lines:
        x = pad
        for ttype, text in line:
            draw.text((x, y), text, font=font, fill=color_for(ttype))
            x += draw.textlength(text, font=font)
        y += line_height

    img.save(out_path)


def fig7_eval_algorithm() -> None:
    """Fig. 7 — ảnh code thật (eval_degradation.py) với comment tiếng Anh."""
    _render_code(CODE_EVAL, IMG_DIR / "fig7_eval_algorithm.png")


def fig8_augmentation_algorithm() -> None:
    """Fig. 8 — ảnh code thật (robustness.py) với comment tiếng Anh."""
    _render_code(CODE_AUGMENT, IMG_DIR / "fig8_augmentation_algorithm.png")


def main() -> None:
    """Sinh đủ 8 hình vào thư mục images/."""
    fig1_pipeline()
    fig2_degradation_examples()
    fig3_roc_curves()
    fig4_severity_curves()
    fig5_aggregate_bars()
    fig6_loss_curves()
    fig7_eval_algorithm()
    fig8_augmentation_algorithm()
    for p in sorted(IMG_DIR.glob("fig*.png")):
        print("created:", p)


if __name__ == "__main__":
    main()
