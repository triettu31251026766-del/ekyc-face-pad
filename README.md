# eKYC Face PAD

Phát hiện tấn công trình diện khuôn mặt (Face Presentation Attack Detection)
cho facial eKYC — đồ án môn học. Đặc tả đầy đủ: `documents/TECHNICAL_DOCUMENTATION_eKYC_FACE_PAD.md`.

Luồng dự án: `Dataset → Preprocessing → PAD Baseline → Clean Evaluation → Quality Degradation → Robustness Training → Comparison → Report`

## Cài đặt & kiểm thử

```bash
python -m venv env
env\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest tests/          # 191 test
```

## Chạy thí nghiệm (mục 32 tài liệu)

```bash
# Toàn bộ chuỗi: E01 -> suy giảm -> E07 robust -> suy giảm robust -> ablation -> bảng/figure
python experiments/run_all.py --config configs/clean.yaml --robustness configs/robustness.yaml

# Hoặc từng bước:
python experiments/train_baseline.py --config configs/clean.yaml
python experiments/eval_degradation.py --config configs/degradation_jpeg.yaml --checkpoint results/checkpoints/E01_baseline_seed42.pt
python experiments/train_robust.py --config configs/clean.yaml --robustness configs/robustness.yaml
python experiments/compare_models.py
python experiments/ablation.py --config configs/clean.yaml --robustness configs/robustness.yaml
```

> Trước khi chạy, đặt dataset (CelebA-Spoof) vào `data/raw/celeba_spoof` gồm
> thư mục ảnh `SpoofingData/` và tệp danh sách `train_list.txt`.

## Definition of Done (mục 48 tài liệu)

- [x] Dataset loads successfully (module `src/data.py`, test bằng dataset tổng hợp)
- [x] Labels are correct and consistent (0 = bona_fide, 1 = spoof)
- [x] Train/test split is reproducible (subject_disjoint, lưu + tái sử dụng splits)
- [x] Dataset class works (`src/dataset.py`)
- [x] Clean preprocessing works (`src/transforms.py`, eval transform tất định)
- [x] MobileNetV2 baseline trains (`src/model.py` + `src/train.py`, smoke test với custom_cnn)
- [x] Clean evaluation works (`src/evaluate.py`, thí nghiệm E01)
- [x] F1/AUC are calculated (`src/metrics.py`)
- [x] APCER/BPCER/ACER are calculated (`src/metrics.py`, quy ước positive = spoof)
- [x] JPEG degradation works
- [x] Resize degradation works
- [x] Blur degradation works
- [x] Noise degradation works
- [x] Brightness degradation works
- [x] Degradation evaluation is deterministic (mục 13, noise có seed từ config)
- [x] Robustness augmentation works (`src/robustness.py`, tái sử dụng `degradation.py`)
- [x] Baseline vs robust comparison works (`experiments/compare_models.py`, không retrain)
- [x] Results are automatically saved (JSON + CSV + predictions, mục 29-30, 39)
- [x] Figures are generated from saved results (mục 40, không bịa số liệu)
- [x] Unit tests pass (191 test, `python -m pytest tests/`)
- [ ] At least one complete experiment can be reproduced from a config file
      (chờ tải dataset CelebA-Spoof thật vào `data/raw/celeba_spoof` — pipeline
      đã sẵn sàng, chưa chạy trên dữ liệu thật)

## Cấu trúc

```text
configs/       cấu hình YAML: base, clean, degradation_*.yaml, robustness
data/          raw (dataset), splits (splits đã lưu), processed
src/           14 module cốt lõi (config, data, dataset, transforms, model,
               train, metrics, evaluate, inference, video, degradation,
               robustness, reproducibility, utils)
experiments/   điều phối thí nghiệm: train_baseline (E01), eval_clean,
               eval_degradation (E02..), train_robust (E07), compare_models,
               ablation (E09+), run_all
results/       raw (JSON/CSV/log), tables, figures, checkpoints
tests/         191 unit test + smoke test end-to-end
notebooks/     khám phá & phân tích (Jupyter)
```
