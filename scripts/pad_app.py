"""scripts/pad_app.py — terminal app điều phối toàn bộ thí nghiệm Face PAD.

Tệp này là giao diện dòng lệnh (CLI) để người dùng THEO DÕI VÀ ĐIỀU KHIỂN toàn bộ
protocol 5 seed theo feedback advisor (xem handover/PROTOCOL_THUC_HIEN.md):

    train (E01/E07) -> chọn threshold trên validation -> eval frozen
    (clean + 16 degradation) -> gộp báo cáo (mean ± SD + paired bootstrap B=5000).

Có 2 cách dùng:

1) TƯƠNG TÁC (menu):
        python -m scripts.pad_app

2) LỆNH TRỰC TIẾP (không menu, tiện cho automation):
        python -m scripts.pad_app status
        python -m scripts.pad_app train --seed 123 --model E01
        python -m scripts.pad_app threshold --seed 123
        python -m scripts.pad_app eval --seed 123
        python -m scripts.pad_app report
        python -m scripts.pad_app auto          # chạy mọi bước còn thiếu (5 seed)

Mọi lệnh con được stream output trực tiếp ra console và lưu log vào results/logs/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Dòng thanh tiến độ tqdm có dạng "... 0%|    | 0/3176 [...]".
_PROGRESS_RE = re.compile(r"\d+%\|")

ROOT = Path(__file__).resolve().parent.parent

# Đảm bảo console Windows ghi được tiếng Việt (tránh UnicodeEncodeError cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SEEDS = [123, 456, 789, 101, 202]
MODELS = {
    "E01": {"tag": "baseline", "label": "baseline"},
    "E07": {"tag": "robust", "label": "robust"},
}
CONFIG = "configs/full_clean.yaml"
ROBUSTNESS_CONFIG = "configs/robustness.yaml"


# --------------------------------------------------------------------------- #
# đường dẫn artifacts
# --------------------------------------------------------------------------- #
def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def ckpt_path(seed: int, model_id: str) -> Path:
    return _p("results", "checkpoints", f"{model_id}_{MODELS[model_id]['tag']}_seed{seed}.pt")


def thr_path(seed: int, model_id: str) -> Path:
    return _p("results", "thresholds", f"{model_id}_seed{seed}.json")


def clean_path(seed: int, model_id: str) -> Path:
    return _p("results", "raw", f"{model_id}_{MODELS[model_id]['tag']}_seed{seed}_clean.json")


def table_path(seed: int, tag: str) -> Path:
    return _p("results", "tables", f"degradation_{tag}_seed{seed}.csv")


# --------------------------------------------------------------------------- #
# tiện ích
# --------------------------------------------------------------------------- #
def read_json(path: Path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def _ckpt_info(seed: int, model_id: str) -> str:
    """Đọc dataset name + seed lưu trong checkpoint (không tải model weights)."""
    p = ckpt_path(seed, model_id)
    if not p.is_file():
        return " - "
    try:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        name = cfg.get("dataset", {}).get("name", "?")
        ck_seed = ck.get("seed", cfg.get("seed", "?"))
        return f"OK({name},s{ck_seed})"
    except Exception:
        return "OK(?)"


def _train_done(seed: int, model_id: str) -> bool:
    """Train được coi là XONG chỉ khi chạy hết epochs + có kết quả cuối.

    Checkpoint bị lưu dở (Ctrl+C giữa chừng) KHÔNG được coi là xong — nếu không,
    auto sẽ bỏ qua train và eval lên model chưa train xong.
    """
    tag = MODELS[model_id]["tag"]
    final_json = _p("results", "raw", f"{model_id}_{tag}_seed{seed}.json")
    if final_json.is_file():
        return True
    ck = ckpt_path(seed, model_id)
    if not ck.is_file():
        return False
    try:
        import torch
        d = torch.load(ck, map_location="cpu", weights_only=False)
        epochs = int(d.get("config", {}).get("training", {}).get("epochs", 0))
        return epochs > 0 and int(d.get("epoch", 0)) >= epochs
    except Exception:
        return False


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    return env


def run_cmd(cmd: list[str]) -> int:
    """Chạy lệnh con, stream stdout ra console và ghi log vào results/logs/."""
    label = "_".join(str(a) for a in cmd[1:])
    label = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)[:90]
    log_dir = _p("results", "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}.log"

    print("\n" + "=" * 72)
    print("> " + " ".join(cmd))
    print("  log: " + str(log_path))
    print("=" * 72)

    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=_env(),
    )
    showing_bar = False
    last_bar = None
    buf = b""

    with open(log_path, "w", encoding="utf-8") as fh:
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            buf += chunk
            # Cắt theo \n HOẶC \r để giữ nguyên tín hiệu refresh của tqdm
            # (tqdm ghi "\r" + thanh mỗi lần cập nhật, KHÔNG có \n).
            while True:
                pos_n = buf.find(b"\n")
                pos_r = buf.find(b"\r")
                cand = [p for p in (pos_n, pos_r) if p >= 0]
                if not cand:
                    break
                pos = min(cand)
                seg = buf[:pos].decode("utf-8", "replace")
                buf = buf[pos + 1:]
                is_nl = pos == pos_n

                if not seg and not is_nl:
                    # Đoạn rỗng giữa 2 lần \r (tqdm refresh liên tiếp) — bỏ qua.
                    continue
                if _PROGRESS_RE.search(seg):
                    # Thanh tiến độ: ghi đè 1 dòng trên console; log chỉ giữ
                    # dòng cuối của mỗi đợt refresh (tránh log phình).
                    sys.stdout.write("\r" + seg)
                    sys.stdout.flush()
                    showing_bar = True
                    last_bar = seg
                    if is_nl:
                        sys.stdout.write("\n")
                        showing_bar = False
                else:
                    if showing_bar:
                        sys.stdout.write("\n")
                        showing_bar = False
                    if last_bar is not None:
                        fh.write(last_bar + "\n")
                        fh.flush()
                        last_bar = None
                    sys.stdout.write(seg + "\n")
                    sys.stdout.flush()
                    fh.write(seg + "\n")
                    fh.flush()

        # Xử lý phần còn dư không kết thúc bằng \n.
        if buf:
            seg = buf.decode("utf-8", "replace")
            if _PROGRESS_RE.search(seg):
                sys.stdout.write("\r" + seg)
                last_bar = seg
            else:
                if showing_bar:
                    sys.stdout.write("\n")
                sys.stdout.write(seg + "\n")
                fh.write(seg + "\n")
        if last_bar is not None:
            fh.write(last_bar + "\n")
    proc.wait()
    print(f"[exit code {proc.returncode}]")
    return proc.returncode


# --------------------------------------------------------------------------- #
# các bước pipeline
# --------------------------------------------------------------------------- #
def step_train(seed: int, model_id: str) -> int:
    if model_id == "E01":
        cmd = [sys.executable, "-m", "experiments.train_baseline",
               "--config", CONFIG, "--seed", str(seed)]
    else:
        cmd = [sys.executable, "-m", "experiments.train_robust",
               "--config", CONFIG, "--robustness", ROBUSTNESS_CONFIG,
               "--seed", str(seed)]
    return run_cmd(cmd)


def step_threshold(seed: int, model_id: str) -> int:
    cmd = [sys.executable, "-m", "experiments.select_threshold",
           "--checkpoint", str(ckpt_path(seed, model_id)),
           "--out", str(thr_path(seed, model_id))]
    return run_cmd(cmd)


def step_eval(seed: int, tag: str) -> int:
    cmd = [sys.executable, "-m", "scripts.run_frozen_eval",
           "--seed", str(seed), "--tag", tag]
    return run_cmd(cmd)


def step_report() -> int:
    cmd = [sys.executable, "-m", "scripts.multi_seed_report",
           "--seeds"] + [str(s) for s in SEEDS]
    return run_cmd(cmd)


# --------------------------------------------------------------------------- #
# trạng thái
# --------------------------------------------------------------------------- #
def _mark(path: Path) -> str:
    return "OK" if path.is_file() else " - "


def _fmt_clean(seed: int, model_id: str) -> str:
    d = read_json(clean_path(seed, model_id))
    if d is None:
        d = read_json(_p("results", "raw",
                         f"{model_id}_{MODELS[model_id]['tag']}_seed{seed}.json"))
    if d is None:
        return "chưa có"
    f1 = d.get("f1"); auc = d.get("roc_auc")
    return f"F1={f1:.3f}/AUC={auc:.3f}" if f1 is not None else "chưa có"


def _fmt_thr(seed: int, model_id: str) -> str:
    d = read_json(thr_path(seed, model_id))
    if d is None:
        return "chưa chọn"
    return f"thr={d['threshold']:.3f}"


def do_status() -> None:
    print("\n=== TRẠNG THÁI THÍ NGHIỆM (5 seed) ===")
    print(f"config: {CONFIG} | robustness: {ROBUSTNESS_CONFIG}")
    done = 0
    for seed in SEEDS:
        e01_ck = _ckpt_info(seed, "E01")
        e07_ck = _ckpt_info(seed, "E07")
        e01_t = _fmt_thr(seed, "E01")
        e07_t = _fmt_thr(seed, "E07")
        e01_c = _fmt_clean(seed, "E01")
        e07_c = _fmt_clean(seed, "E07")
        b_tab = _mark(table_path(seed, "baseline"))
        r_tab = _mark(table_path(seed, "robust"))

        complete = all(p.is_file() for p in [
            ckpt_path(seed, "E01"), ckpt_path(seed, "E07"),
            thr_path(seed, "E01"), thr_path(seed, "E07"),
            clean_path(seed, "E01"), clean_path(seed, "E07"),
            table_path(seed, "baseline"), table_path(seed, "robust"),
        ])
        done += int(complete)

        print(f"\n  SEED {seed} {'[HOAN TAT]' if complete else ''}")
        print(f"    E01 baseline : ckpt={e01_ck}  {e01_t}  clean: {e01_c}  deg={b_tab}")
        print(f"    E07 robust   : ckpt={e07_ck}  {e07_t}  clean: {e07_c}  deg={r_tab}")

    print(f"\n  Tiến độ: {done}/5 seed hoàn tất đầy đủ.")
    rep = _p("results", "multiseed")
    if any(rep.glob("*.csv")):
        print(f"  Báo cáo gộp: {rep}")
    else:
        print("  Báo cáo gộp: chưa có (chạy 'report' sau khi có đủ kết quả).")


# --------------------------------------------------------------------------- #
# pipeline cho 1 seed
# --------------------------------------------------------------------------- #
def pipeline_seed(seed: int, skip_done: bool = True) -> int:
    steps = []
    if not (skip_done and _train_done(seed, "E01")):
        steps.append(("train E01", lambda: step_train(seed, "E01")))
    if not (skip_done and _train_done(seed, "E07")):
        steps.append(("train E07", lambda: step_train(seed, "E07")))
    if not (skip_done and thr_path(seed, "E01").is_file()):
        steps.append(("threshold E01", lambda: step_threshold(seed, "E01")))
    if not (skip_done and thr_path(seed, "E07").is_file()):
        steps.append(("threshold E07", lambda: step_threshold(seed, "E07")))
    if not (skip_done and clean_path(seed, "E01").is_file() and table_path(seed, "baseline").is_file()):
        steps.append(("eval baseline", lambda: step_eval(seed, "baseline")))
    if not (skip_done and clean_path(seed, "E07").is_file() and table_path(seed, "robust").is_file()):
        steps.append(("eval robust", lambda: step_eval(seed, "robust")))

    if not steps:
        print(f"seed {seed}: mọi bước đã hoàn tất.")
        return 0

    print(f"\n[seed {seed}] Các bước sẽ chạy: {', '.join(s for s, _ in steps)}")
    for name, fn in steps:
        print(f"\n>>> seed {seed}: {name}")
        rc = fn()
        if rc != 0:
            print(f"!!! Bước '{name}' thất bại (rc={rc}), dừng pipeline seed {seed}.")
            return rc
    return 0


# --------------------------------------------------------------------------- #
# các hành động tương tác
# --------------------------------------------------------------------------- #
def _choose(options: list[tuple], title: str):
    """Chọn một mục bằng số thứ tự: in danh sách, nhận input 1..n."""
    print(f"\n  {title}")
    for i, (_, label) in enumerate(options, start=1):
        print(f"    [{i}] {label}")
    while True:
        v = input("  chọn (số)> ").strip()
        if v.isdigit() and 1 <= int(v) <= len(options):
            return options[int(v) - 1][0]
        print(f"  nhập số từ 1 đến {len(options)}.")


def _choose_seed() -> int:
    options = [(s, f"seed {s}") for s in SEEDS]
    return _choose(options, "Chọn seed:")


def _choose_model() -> str:
    options = [(mid, f"{MODELS[mid]['label']} ({mid})") for mid in MODELS]
    return _choose(options, "Chọn model:")


def do_train_prompt() -> None:
    do_status()
    seed = _choose_seed()
    model = _choose_model()
    rc = step_train(seed, model)
    print(">>> train xong (rc=%d)" % rc)


def do_threshold_prompt() -> None:
    do_status()
    seed = _choose_seed()
    model = _choose_model()
    rc = step_threshold(seed, model)
    print(">>> threshold xong (rc=%d)" % rc)


def do_eval_prompt() -> None:
    do_status()
    seed = _choose_seed()
    tag = _choose([("baseline", "baseline (E01)"), ("robust", "robust (E07)")],
                  "Chọn tag:")
    rc = step_eval(seed, tag)
    print(">>> eval xong (rc=%d)" % rc)


def do_pipeline_prompt() -> None:
    do_status()
    seed = _choose_seed()
    rc = pipeline_seed(seed, skip_done=True)
    print(">>> pipeline seed %d xong (rc=%d)" % (seed, rc))


def do_auto() -> None:
    do_status()
    for seed in SEEDS:
        rc = pipeline_seed(seed, skip_done=True)
        if rc != 0:
            print(f"!!! Dừng auto vì seed {seed} thất bại.")
            return
    do_report()


def do_report() -> None:
    rc = step_report()
    print(">>> report xong (rc=%d)" % rc)


MENU = [
    ("1", "Xem trạng thái", do_status),
    ("2", "Train (E01/E07) một seed", do_train_prompt),
    ("3", "Chọn threshold một seed", do_threshold_prompt),
    ("4", "Eval frozen (clean + 16 degradation) một seed", do_eval_prompt),
    ("5", "Chạy toàn bộ pipeline một seed", do_pipeline_prompt),
    ("6", "Gộp báo cáo 5 seed", do_report),
    ("7", "AUTO: chạy mọi bước còn thiếu (5 seed)", do_auto),
    ("0", "Thoát", None),
]


def interactive() -> None:
    print("\n" + "=" * 72)
    print("  FACE PAD — ĐIỀU KHIỂN THÍ NGHIỆM 5 SEED")
    print("=" * 72)
    while True:
        print("\n---- MENU ----")
        for key, desc, _ in MENU:
            print(f"  {key}. {desc}")
        choice = input("  chọn> ").strip()
        for key, _, fn in MENU:
            if choice == key:
                if fn is None:
                    print("Tạm biệt.")
                    return
                fn()
                break
        else:
            print("  lựa chọn không hợp lệ.")


# --------------------------------------------------------------------------- #
# CLI trực tiếp (không menu)
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal app điều phối thí nghiệm Face PAD")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="xem trạng thái")
    p_train = sub.add_parser("train", help="train một model cho một seed")
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--model", choices=["E01", "E07"], required=True)
    p_thr = sub.add_parser("threshold", help="chọn threshold trên validation")
    p_thr.add_argument("--seed", type=int, required=True)
    p_thr.add_argument("--model", choices=["E01", "E07"], required=True)
    p_eval = sub.add_parser("eval", help="eval frozen (clean + degradation)")
    p_eval.add_argument("--seed", type=int, required=True)
    p_eval.add_argument("--tag", choices=["baseline", "robust"], required=True)
    sub.add_parser("report", help="gộp báo cáo 5 seed")
    sub.add_parser("auto", help="chạy mọi bước còn thiếu")
    p_pipe = sub.add_parser("pipeline", help="pipeline 1 seed")
    p_pipe.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command is None:
        interactive()
        return

    if args.command == "status":
        do_status()
    elif args.command == "train":
        rc = step_train(args.seed, args.model)
        sys.exit(rc)
    elif args.command == "threshold":
        rc = step_threshold(args.seed, args.model)
        sys.exit(rc)
    elif args.command == "eval":
        rc = step_eval(args.seed, args.tag)
        sys.exit(rc)
    elif args.command == "pipeline":
        sys.exit(pipeline_seed(args.seed, skip_done=True))
    elif args.command == "report":
        sys.exit(step_report())
    elif args.command == "auto":
        do_auto()


if __name__ == "__main__":
    main()
