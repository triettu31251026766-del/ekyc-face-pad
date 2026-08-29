# eKYC Face PAD

Phát hiện tấn công trình diện khuôn mặt (Face Presentation Attack Detection)
cho facial eKYC — đồ án môn học.

**BẮT ĐẦU TỪ ĐÂY:**
- Tài liệu kỹ thuật: `documents/TECHNICAL_DOCUMENTATION_eKYC_FACE_PAD.md`
- Guide sinh viên: `documents/Student_Guide_eKYC_Face_PAD_Project.docx`
- **Trạng thái + việc tiếp theo: `documents/PROGRESS.md`** ← mở file này đầu tiên
- Lệnh chạy từng thí nghiệm: `documents/HUONG_DAN_TRAINING_MODEL_eKYC_FACE_PAD.md`
- Hướng dẫn cho AI coding agent: `AGENTS.md`

Luồng dự án: `Dataset → Preprocessing → PAD Baseline → Clean Evaluation → Quality Degradation → Robustness Training → Comparison → Report`

## Trạng thái hiện tại (tóm tắt)

- Code base: **HOÀN CHỈNH** — 14 module `src/`, experiments, 193 test pass.
- **Giai đoạn pilot 18k: XONG** → kết quả lưu tại `results/pilot_18k/`
  (E01 F1 .931, E07 F1 .929, E20 fine-tune webcam: P(spoof) live 0.89 → 0.017).
- **Giai đoạn chính (FULL CelebA-Spoof 625k): ĐANG LÀM** — xem `documents/PROGRESS.md` mục 7.

## Cài đặt & kiểm thử (máy mới)

```bash
python -m pip install -r requirements.txt
# Nếu có GPU NVIDIA:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # phải ra True nếu có GPU
python -m pytest tests/ -q    # 193 test pass, 1 skip (skip khi có CUDA)
```

## Chạy thí nghiệm (chi tiết + giải thích: `documents/HUONG_DAN_TRAINING_MODEL_eKYC_FACE_PAD.md`)

```bash
python -m experiments.train_baseline --config configs/clean.yaml
python -m experiments.eval_degradation --config configs/degradation_jpeg.yaml --checkpoint results/checkpoints/E01_baseline_seed123.pt
python -m experiments.train_robust --config configs/clean.yaml --robustness configs/robustness.yaml
python -m experiments.compare_models
python -m experiments.ablation --config configs/clean.yaml --robustness configs/robustness.yaml
python -m experiments.run_all --config configs/clean.yaml --robustness configs/robustness.yaml
```

> LƯU Ý: luôn chạy bằng `python -m experiments.X` (KHÔNG `python experiments/X.py`).
> Kết quả pilot 18k nằm trong `results/pilot_18k/` — KHÔNG xóa.

## Dataset (tùy giai đoạn)

- **Pilot 18k**: `data/raw/celeba_spoof/` — tải lại bằng `python -m scripts.download_dataset`
  (mirror HuggingFace, đã crop mặt theo bbox).
- **FULL (giai đoạn chính)**: giải nén 74 phần zip vào `data/raw/celeba_spoof_full/`
  (thư mục ảnh `SpoofingData/` + `train_list.txt`).

## Công cụ camera

```bash
python -m scripts.collect_webcam_data --mode live --count 300   # thu ảnh mặt thật
python -m scripts.collect_webcam_data --mode spoof --count 200  # thu ảnh giả mạo
python -m experiments.finetune_webcam                          # fine-tune với ảnh webcam
python -m scripts.camera_demo                                  # demo camera trực tiếp
```

## Definition of Done (mục 48 tài liệu)

- [x] Dataset loads successfully (module `src/data.py`)
- [x] Labels are correct and consistent (0 = bona_fide, 1 = spoof)
- [x] Train/test split is reproducible (subject_disjoint, lưu + tái sử dụng splits)
- [x] Dataset class works (`src/dataset.py`)
- [x] Clean preprocessing works (`src/transforms.py`, eval transform tất định)
- [x] MobileNetV2 baseline trains (`src/model.py` + `src/train.py`)
- [x] Clean evaluation works (`src/evaluate.py`, thí nghiệm E01)
- [x] F1/AUC are calculated (`src/metrics.py`)
- [x] APCER/BPCER/ACER are calculated (`src/metrics.py`, quy ước positive = spoof)
- [x] JPEG / Resize / Blur / Noise / Brightness degradation works
- [x] Degradation evaluation is deterministic (mục 13)
- [x] Robustness augmentation works (`src/robustness.py` + crop augmentation)
- [x] Baseline vs robust comparison works (`experiments/compare_models.py`, không retrain)
- [x] Results are automatically saved (JSON + CSV + predictions, mục 29-30, 39)
- [x] Figures are generated from saved results (mục 40, không bịa số liệu)
- [x] Unit tests pass (193 test)
- [x] At least one complete experiment can be reproduced from a config file
      (pilot 18k: E01/E07 đã chạy trên dữ liệu thật — `results/pilot_18k/`)

## Cấu trúc

```text
AGENTS.md        hướng dẫn bắt buộc cho AI coding agent
configs/         cấu hình YAML: base, clean, degradation_*.yaml, robustness
data/            raw (dataset), splits (splits đã lưu), webcam_data (ảnh thu thập)
documents/       đặc tả, guide sinh viên, PROGRESS.md, hướng dẫn training
src/             14 module cốt lõi (config, data, dataset, transforms, model,
                 train, metrics, evaluate, inference, video, degradation,
                 robustness, reproducibility, utils)
experiments/     train_baseline (E01), eval_clean, eval_degradation (E02..),
                 train_robust (E07), compare_models, ablation (E09+), run_all,
                 finetune_webcam (E20), _common
scripts/         download_dataset, camera_demo, collect_webcam_data + model YuNet
results/         raw, tables, figures, checkpoints (giai đoạn chính)
results/pilot_18k/  kết quả giai đoạn pilot (raw + checkpoints + camera_debug)
tests/           193 test + smoke end-to-end
notebooks/       khám phá & phân tích (Jupyter)
```
