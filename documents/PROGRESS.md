# PROGRESS.md — Trạng thái dự án eKYC Face PAD

> **File này là "sổ nhật ký" của dự án.** Mỗi lần làm xong một việc PHẢI cập
> nhật tick `[x]` + ghi kết quả vào đây. Sinh viên kế tiếp / AI agent mở file
> này ĐẦU TIÊN để biết đang ở đâu và làm tiếp việc gì.
> Ngày cập nhật gần nhất: 29/08/2026 | Deadline đồ án: 10/09/2026.

---

## CÂY TIẾN ĐỘ (1 dòng, tick việc đã xong)

```
[x] 1. Môi trường (requirements + torch CUDA) -> [x] 2. Skeleton repo theo tài liệu -> [x] 3. src/ 14 module + tests (193 pass) -> [x] 4. PILOT 18k (dataset -> E01 baseline F1 .931 -> E07 robust F1 .929 -> degradation config) -> [x] 5. Camera demo + phát hiện domain shift (Test A/B/C/D) -> [x] 6. E20 fine-tune webcam (P(spoof) live 0.89 -> 0.017) -> [x] 7. Script tải full ~200k + config full_clean + lưới suy giảm (eval_degradation_grid) -> [x] 8. Tải 200k + train E01 baseline + đánh giá lưới suy giảm (E02-E06) -> [x] 9. Train E07 robust 200k (crop TẮT, p=0.3) -> [x] 10. Đánh giá robust (E08-E12) + SO SÁNH baseline vs robust (robust thắng ở noise: ACER 0.433->0.157) -> [ ] 11. P1: data-scale + ablation + webcam nhiều người -> [x] 12. P2: BÀI BÁO (Paper_Robust_Face_PAD_for_eKYC.docx: 12 bảng + 8 hình + 15 refs thật + peer-review audit + đã sửa corrections) -> [ ] 13. Nộp báo cáo cuối (trước 10/09)
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

[x] 7. P0 — GIAI ĐOẠN CHÍNH (FULL CelebA-Spoof ~200k, KHÔNG tải 74GB)
    [x] 7.1 scripts/download_celeba_full.py: tải 45 train shard (~200k) + 22 test shard (~20k KHÓA) + 4 valid shard (~20k) từ mirror HF Ar4ikov/celebA_spoof (~24GB, log từng shard + ETA)
    [x] 7.2 configs/full_clean.yaml (dataset celeba_spoof_full, seed 123, MobileNetV2 224, 20 epochs)
    [x] 7.3 experiments/eval_degradation_grid.py: lưới suy giảm đầy đủ 16 điều kiện (jpeg 90/70/50/30, resize 75/50/25, blur light/medium/strong, noise low/medium/high, brightness dark/normal/bright) + bảng CSV + 5 biểu đồ
    [x] 7.4 Smoke test: grid chạy OK trên checkpoint pilot (bảng suy giảm đã đo)
    [x] 7.5 ĐÃ TẢI: python -m scripts.download_celeba_full (203.215 train / 20.042 test / 20.773 val)
    [x] 7.6 ĐÃ TRAIN: python -m experiments.train_baseline --config configs/full_clean.yaml (E01: F1 .825, AUC .963, ACER .157, 19.7h)

[x] 8. ĐÁNH GIÁ BASELINE TRƯỚC — XONG (kết quả: documents/NHAT_KY_THI_NGHIEM.md mục 5.2)
    [x] kiểm tra checkpoint + config E01 full
    [x] python -m experiments.eval_degradation_grid --checkpoint results/checkpoints/E01_baseline_seed123.pt --tag baseline
    [x] bảng results/tables/degradation_baseline.csv + biểu đồ results/figures/
    [x] phân tích: baseline yếu nhất ở NOISE (F1 .27 ở high), sau đó JPEG/brightness

[x] 9. E07 ROBUST 200k — XONG (01/09)
    [x] config robustness.yaml: crop TẮT, 5 augmentation p=0.3 (P(ảnh sạch)~17%)
    [x] train_robust --config configs/full_clean.yaml --robustness configs/robustness.yaml (20.6h)
    [x] eval_degradation_grid --checkpoint results/checkpoints/E07_robust_seed123.pt --tag robust
    [x] SO SÁNH (mục 5.3 nhật ký): robust cải thiện mạnh nhất ở noise
        (noise high: F1 .27->.83, ACER .433->.157, AUC .780->.952);
        clean không bị hy sinh (F1 +.026, AUC -.002); 2 điều kiện F1 tụt nhẹ
        (blur strong -.008, resize25 -.001) dù AUC tăng

[ ] 10. P1 — Nên làm nếu còn thời gian
    [ ] data-scale: 18k (pilot) vs 50k vs 100k vs 200k (cùng protocol)
    [ ] webcam evaluation nhiều người (baseline 200k vs robust 200k)
    [ ] ablation từng augmentation (experiments/ablation.py)

[x] 11. P2 — BÀI BÁO KHOA HỌC (đã xong draft + audit)
    [x] Paper_Robust_Face_PAD_for_eKYC.docx: ~5.000 từ, 12 bảng, 8 hình, 15 refs thật
    [x] 8 hình sinh từ dữ liệu thật -> images/ (script: scripts/make_paper_figures.py)
        fig1 pipeline, fig2 ảnh suy giảm thật, fig3 ROC, fig4 severity, fig5 aggregate,
        fig6 loss, fig7-8 ảnh code màu (comment tiếng Anh)
    [x] Peer-review + research audit: 0 lỗi số liệu, 14/14 refs verified (Crossref/arXiv),
        không novelty giả; corrections đã sửa (subject-disjoint wording, "exceeds",
        bỏ moiré, thêm nuance abstract)
    [ ] Điền tên tác giả thay [Author Names] trước khi nộp
    [ ] failure case analysis (FP/FN) — tùy chọn nếu còn thời gian
```

---

## NHẬT KÝ CÁC PHÁT HIỆN QUAN TRỌNG (dùng cho báo cáo)

| # | Phát hiện | Bằng chứng | Ý nghĩa |
|---|---|---|---|
| 1 | Model F1 .93 trên test vẫn **fail webcam thật** (gọi spoof 93-100%) | Test C (camera log) | Domain gap giữa CelebA-Spoof và webcam thật |
| 2 | Inference pipeline **không có bug** | Test B (ảnh dataset live -> P(spoof) 0.0-0.07) | Vấn đề nằm ở dữ liệu, không phải code |
| 3 | Model **rất nhạy với mức crop** | Test D (full -> .81 spoof; crop 10% -> .10 live) | Model overfit tỉ lệ mặt/ảnh của dataset |
| 4 | 500 ảnh webcam fine-tune kéo P(spoof) live **0.89 -> 0.017** | E20 (results/pilot_18k/checkpoints/E20) | Domain adaptation với ít dữ liệu đích rất hiệu quả |
| 5 | F1@0.5 có thể CHE GIẤU suy giảm: baseline resize/blur giữ F1 ~.84 nhưng AUC sụp .96->.82/.78, BPCER tăng tới .57 | bảng 5.2 nhật ký | Phải báo đủ AUC/APCER/BPCER/ACER |
| 6 | **Robust training cải thiện rõ nhất ở noise** (điểm yếu nhất của baseline): noise high F1 .27->.83, ACER .433->.157, AUC +.17 | bảng 5.3 nhật ký | Quality augmentation có hiệu quả, không hy sinh clean (F1 +.026) |

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
