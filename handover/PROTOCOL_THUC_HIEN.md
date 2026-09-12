# PROTOCOL THỰC HIỆN — RE-RUN THEO FEEDBACK ADVISOR (29 PHASE)

> File này ghi lại từng bước đã làm / chưa làm. Nguyên tắc tối cao: KHÔNG bịa số liệu,
> KHÔNG giả lập, KHÔNG claim novelty, KHÔNG dùng test set để chọn thành phần pipeline.
> Các lệnh CẦN TRAINING sẽ do người dùng chạy rồi gửi lại kết quả.

## A. KẾT QUẢ AUDIT (Phase 1, 3, 4 — KHÔNG cần training)

### Phase 1 — Dataset & Split
- Source: CelebA-Spoof mirror HuggingFace `Ar4ikov/celebA_spoof` (tải bằng
  `scripts/download_celeba_full.py`).
- Metadata có sẵn: Filepath (ảnh), Bbox, Class (live/spoof) — KHÔNG có subject ID thật.
- Split hiện tại: `data/splits/celeba_spoof_seed123_subject_disjoint.json`
  (train 142,250 / val 20,322 / test 40,643; tổng 203,215; live 70,288 / spoof 132,927).
- **KẾT LUẬN PHASE 1 (QUAN TRỌNG)**: subject_id được gán từ tên file ảnh
  (`{idx:06d}.jpg` → subject_id = stem = chỉ số ảnh, xem `download_celeba_full.py`
  dòng 149 và 246). => MỖI ẢNH LÀ 1 "SUBJECT" RIÊNG. => Split KHÔNG phải
  subject-disjoint thật; đây là **IMAGE-LEVEL SPLIT** (held-out images).
  => KHÔNG được dùng: "subject-disjoint", "unseen subjects", "unseen-person
  generalization". Thay bằng "image-level split / held-out image split".

### Phase 3 — Model Initialization
- `build_model()` mặc định `pretrained=False`; `_common.py` gọi
  `build_model(config["model"]["name"], num_classes=1)` KHÔNG truyền pretrained.
  => **MobileNetV2 được TRAIN FROM SCRATCH** (không pretrained ImageNet).

### Phase 4 — Freeze Training Config (đã khớp spec)
- MobileNetV2, 224x224, Adam lr 1e-4 wd 1e-5, batch 64, 20 epochs,
  BCEWithLogitsLoss không class weighting, baseline = RandomHorizontalFlip,
  E07 = 5 augmentation p=0.3 (jpeg U[50,90], resize U[0.5,1.0], blur sigma U[0.5,2.0],
  noise U[0.005,0.03], brightness U[0.7,1.3]), crop TẮT. => KHỚP spec.

### Phase 11 — Paired degradation (KHÔNG cần thay đổi, đã đúng)
- Noise ở EVAL dùng seed cố định: `apply_degradation_config` →
  `gaussian_noise(..., seed=config["seed"])` (config seed = 123 cho cả E01/E07).
  JPEG/resize/blur/brightness là hàm tất định. => E01 và E07 dùng CÙNG degradation
  realization. (Có thể pre-generate để chắc chắn, nhưng hiện đã thoả mãn.)

## B. CODE ĐÃ/ĐANG SỬA (để chạy multi-seed + threshold + bootstrap)

1. `_common.py`:
   - [x] Tách split seed khỏi training seed (`split_seed = config["split"].get("seed", seed)`
     trong `train_and_evaluate`, `load_test_loader`, `load_val_loader`).
   - [x] Thêm `load_val_loader(...)` (để chọn threshold trên val).
2. `experiments/train_baseline.py` + `train_robust.py`: [x] thêm `--seed` override
   (training seed; split seed luôn lấy `split.seed` = 123).
3. `configs/full_clean.yaml`: [x] thêm `split.seed: 123` (THÍ NGHIỆM CHÍNH 200k dùng
   config này — xem NHAT_KY_THI_NGHIEM.md mục 4; `clean.yaml` là pilot 18k).
4. `experiments/select_threshold.py`: [x] chọn threshold trên VALIDATION (min ACER,
   tie-break gần 0.5, grid 0.005), lưu JSON vào `results/thresholds/`.
5. eval scripts: [x] `eval_clean.py` + `eval_degradation_grid.py` nhận `--threshold <frozen>`.
   `eval_degradation_grid.py` cũng ghi bảng/bảng hình theo `*_seed{seed}`.
6. `scripts/multi_seed_report.py`: [x] gộp 5 seed -> mean ± SD (clean + 16 degradation
   + aggregate) + paired bootstrap B=5000 (ΔF1/ΔACER/ΔAUC).
   (Thay cho multi_seed_summary.py + bootstrap_ci.py riêng lẻ.)

=> 193 test vẫn pass sau các thay đổi.

## C. DANH SÁCH 5 SEED
123, 456, 789, 101, 202 (cố định).

## C2. LỆNH CHẠY (người dùng chạy — training trên máy có GPU)

```cmd
set PYTHONUTF8=1

:: 1) TRAIN 10 runs (5 seed x {E01, E07}) — mỗi run ~20h (GPU)
python -m experiments.train_baseline --config configs/full_clean.yaml --seed 123
python -m experiments.train_robust --config configs/full_clean.yaml --robustness configs/robustness.yaml --seed 123
:: ... lặp lại với --seed 456 789 101 202 ...

:: 2) CHỌN threshold trên VALIDATION (min ACER) — nhanh (inference val ~20k)
python -m experiments.select_threshold --checkpoint results/checkpoints/E01_baseline_seed123.pt --out results/thresholds/E01_seed123.json
python -m experiments.select_threshold --checkpoint results/checkpoints/E07_robust_seed123.pt --out results/thresholds/E07_seed123.json
:: ... lặp lại 5 seed ...

:: 3) EVAL clean + 16 degradation tại threshold đã FREEZE (tự đọc JSON threshold)
python -m scripts.run_frozen_eval --seed 123 --tag baseline
python -m scripts.run_frozen_eval --seed 123 --tag robust
:: ... lặp lại 5 seed ...

:: 4) GỘP 5 seed -> mean ± SD + paired bootstrap B=5000
python -m scripts.multi_seed_report --seeds 123 456 789 101 202
```

Đầu ra:
- `results/thresholds/E01_seed<seed>.json`, `E07_seed<seed>.json`
- `results/raw/E01_baseline_seed<seed>_clean.json` (clean @ frozen threshold)
- `results/tables/degradation_baseline_seed<seed>.csv`,
  `degradation_robust_seed<seed>.csv` (16 điều kiện @ frozen threshold)
- `results/multiseed/*.csv` (mean ± SD + bootstrap)

## D. AUDIT CHECKLIST (mục XXIX) — trạng thái
A. image-level (KHÔNG subject-disjoint)  | B. KHÔNG có genuine subject ID
C. không thấy train/test leakage (image-level) | D. trained from scratch
E..R: CHƯA (chờ chạy training) — NOT VERIFIED
