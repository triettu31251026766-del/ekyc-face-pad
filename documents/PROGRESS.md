# PROGRESS.md — Trạng thái dự án eKYC Face PAD

> **File này là "sổ nhật ký" của dự án.** Mỗi lần làm xong một việc PHẢI cập
> nhật tick `[x]` + ghi kết quả vào đây. Sinh viên kế tiếp / AI agent mở file
> này ĐẦU TIÊN để biết đang ở đâu và làm tiếp việc gì.
> Ngày cập nhật gần nhất: 29/08/2026 | Deadline đồ án: 10/09/2026.

---

## CÂY TIẾN ĐỘ (1 dòng, tick việc đã xong)

```
[x] 1. Môi trường (requirements + torch CUDA) -> [x] 2. Skeleton repo theo tài liệu -> [x] 3. src/ 14 module + tests (193 pass) -> [x] 4. PILOT 18k (dataset -> E01 baseline F1 .931 -> E07 robust F1 .929 -> degradation config) -> [x] 5. Camera demo + phát hiện domain shift (Test A/B/C/D) -> [x] 6. E20 fine-tune webcam (P(spoof) live 0.89 -> 0.017) -> [ ] 7. P0: FULL CelebA-Spoof (tải 74 phần -> giải nén -> EDA -> split khóa test -> train 200k) -> [ ] 8. P0: degradation + robust + comparison + bảng/figure -> [ ] 9. P1: data-scale ablation + webcam nhiều người -> [ ] 10. P2: báo cáo cuối
```

## CHI TIẾT TỪNG MỤC

```
[x] 1. Môi trường
    [x] requirements.txt (torch, torchvision, numpy, pandas, sklearn, pillow, pyyaml, matplotlib, tqdm, pytest, ...)
    [x] torch 2.13.0+cu126 (CUDA, RTX 3050) — cài: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    [x] smoke test môi trường PASS

[x] 2. Skeleton repo
    [x] cấu trúc thư mục đúng mục 4 tài liệu (configs/, data/, src/, experiments/, results/, notebooks/, tests/)
    [x] pyproject.toml (pytest config), .gitignore

[x] 3. Code base (HOÀN CHỈNH — không cần viết thêm module nào)
    [x] src/: config, data, dataset, transforms, degradation, model, train, metrics, evaluate, inference, video, robustness, reproducibility, utils
    [x] experiments/: train_baseline, train_robust, eval_clean, eval_degradation, compare_models, ablation, run_all, _common, finetune_webcam
    [x] scripts/: download_dataset (tải 18k HF mirror), camera_demo, collect_webcam_data, face_detection_yunet_2023mar.onnx
    [x] tests/: 193 test pass, 1 skip (skip có chủ đích khi có CUDA)

[x] 4. PILOT 18k — HOÀN THÀNH (kết quả: results/pilot_18k/)
    [x] dataset 18k từ HuggingFace (Camilotabares1/celebA_spoof_sample_split, crop theo bbox + 10% lề)
        -> data/raw/celeba_spoof/ (18.000 ảnh: live 5.941, spoof 12.059) + train_list.txt
    [x] split subject_disjoint seed 123 -> data/splits/celeba_spoof_seed123_subject_disjoint.json (train 12.600 / val 1.800 / test 3.600)
    [x] E01 baseline (MobileNetV2, 20 epoch): F1 .931, ROC-AUC .955, ACER .107, APCER .069, BPCER .146
    [x] E07 robust (aug: jpeg/resize/blur/noise/brightness/crop): F1 .929, ROC-AUC .956, ACER .110
    [x] checkpoint: E01, E07, E20 (trong results/pilot_18k/checkpoints/)

[x] 5. Camera demo + PHÁT HIỆN DOMAIN SHIFT (quan trọng cho báo cáo!)
    [x] scripts/camera_demo.py: YuNet detect + crop 10% + model + khung xanh/đỏ + % + temporal smoothing
    [x] Test C: mặt webcam thật bị gọi spoof 93-100% (E01)
    [x] Test B: ảnh dataset live -> P(spoof) 0.0-0.07 (pipeline inference ĐÚNG, không có bug)
    [x] Test D: model CỰC NHẠY với mức crop (ảnh full -> spoof .81; crop 10% -> live .10)
    [x] Kết luận: domain shift (webcam khác thiết bị thu của dataset) + model overfit tỉ lệ crop

[x] 6. E20 fine-tune webcam
    [x] scripts/collect_webcam_data.py: thu 500 ảnh webcam (live+spoof) -> data/webcam_data/
    [x] experiments/finetune_webcam.py: fine-tune 3 epoch từ E07 (lr 5e-5)
    [x] Kết quả: P(spoof) trên mặt webcam live 0.89 -> 0.017 (bằng chứng domain adaptation)

[ ] 7. P0 — FULL CelebA-Spoof (GIAI ĐOẠN CHÍNH, ĐANG LÀM)
    [ ] 7.1 Tải full: 74 phần zip (~74GB) từ Google Drive + train_list.txt (người có dataset tải giúp team)
    [ ] 7.2 Giải nén vào data/raw/celeba_spoof_full/ (cần ~150GB trống)
    [ ] 7.3 EDA: scripts/eda_dataset.py -> dataset_report.json (subject, live/spoof, spoof type, illumination, environment)
    [ ] 7.4 Subject-disjoint split + KHÓA test set ~20k (scripts/prepare_full_dataset.py, dùng src/data.py create_splits)
    [ ] 7.5 Train subset có kiểm soát: ~200k (cân bằng label, trải subject) — config mới (dataset root mới)
    [ ] 7.6 E01 main baseline (MobileNetV2 224, 20 epoch)
    [ ] 7.7 Degradation eval (jpeg/resize/blur/noise/brightness) trên model E01
    [ ] 7.8 E07 main robust -> so sánh baseline vs robust (compare_models)

[ ] 8. P0 — Kết quả
    [ ] bảng so sánh + figures (results/tables/, results/figures/)
    [ ] failure analysis (ảnh sai: FP/FN, theo spoof type)

[ ] 9. P1 — Nên làm nếu còn thời gian
    [ ] data-scale ablation: 10k / 25k / 50k / 100k / 200k (cùng protocol)
    [ ] webcam evaluation nhiều người (không phải chỉ người train)
    [ ] ablation từng augmentation (đã có experiments/ablation.py)

[ ] 10. P2 — Báo cáo cuối (trước 10/09)
    [ ] viết báo cáo từ results/ (tables + figures)
    [ ] đưa câu chuyện pilot -> full vào báo cáo (domain shift, data scale)
```

---

## NHẬT KÝ CÁC PHÁT HIỆN QUAN TRỌNG (dùng cho báo cáo)

| # | Phát hiện | Bằng chứng | Ý nghĩa |
|---|---|---|---|
| 1 | Model F1 .93 trên test vẫn **fail webcam thật** (gọi spoof 93-100%) | Test C (camera log) | Domain gap giữa CelebA-Spoof và webcam thật |
| 2 | Inference pipeline **không có bug** | Test B (ảnh dataset live -> P(spoof) 0.0-0.07) | Vấn đề nằm ở dữ liệu, không phải code |
| 3 | Model **rất nhạy với mức crop** | Test D (full -> .81 spoof; crop 10% -> .10 live) | Model overfit tỉ lệ mặt/ảnh của dataset |
| 4 | 500 ảnh webcam fine-tune kéo P(spoof) live **0.89 -> 0.017** | E20 (results/pilot_18k/checkpoints/E20) | Domain adaptation với ít dữ liệu đích rất hiệu quả |

## QUY ƯỚC THÍ NGHIỆM (bắt buộc giữ nguyên)

- seed = 123 | model = mobilenet_v2 | image_size = 224 | threshold = 0.5 | lr = 1e-4 | batch 64 | 20 epochs
- Nhãn: 0 = bona_fide, 1 = spoof. Metric: F1, ROC-AUC, PR-AUC, APCER, BPCER, ACER (positive = spoof)
- Chỉ đổi 1 biến khi so sánh (baseline vs robust: chỉ khác chiến lược huấn luyện)
- KHÔNG chọn threshold trên test set (dùng val nếu cần) — tránh leak

## HƯỚNG DẪN MÁY MỚI (sinh viên nhận repo, chạy máy riêng)

1. `python -m pip install -r requirements.txt`
2. Cài torch CUDA nếu có GPU (nếu không thì chạy CPU, chậm hơn):
   `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps`
3. Dataset: tùy việc đang làm
   - Pilot (18k): `python -m scripts.download_dataset` (tự tải từ HuggingFace)
   - Main (full): chép `data/raw/celeba_spoof_full/` từ ổ cứng của team (KHÔNG tải lại 74GB)
4. Kiểm tra: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
5. `python -m pytest tests/ -q` — phải 193 pass
6. Xem PROGRESS.md để biết việc tiếp theo; lệnh cụ thể trong documents/HUONG_DAN_TRAINING_MODEL_eKYC_FACE_PAD.md
7. Tùy máy: giảm `training.batch_size` nếu OOM; điều chỉnh `training.num_workers`

## THƯ MỤC QUAN TRỌNG

```text
results/pilot_18k/raw/          kết quả pilot (E01, E07 json/csv/predictions/log)
results/pilot_18k/checkpoints/  E01, E07, E20 .pt
results/pilot_18k/camera_debug/ crop webcam lưu khi gỡ lỗi (Test A)
results/raw/ + results/checkpoints/   dùng cho GIAI ĐOẠN CHÍNH (full dataset)
data/raw/celeba_spoof/          dataset pilot 18k (HF mirror, đã crop)
data/raw/celeba_spoof_full/     dataset FULL (tải về + giải nén vào đây)
data/splits/                    split đã lưu (pilot: celeba_spoof_seed123_...json)
data/webcam_data/               500 ảnh webcam thu thập (live + spoof)
scripts/                        công cụ: download_dataset, camera_demo, collect_webcam_data
```
