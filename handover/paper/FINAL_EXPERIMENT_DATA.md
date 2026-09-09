# FINAL EXPERIMENT DATA — eKYC Face PAD (200k)

> Gói dữ liệu thí nghiệm CUỐI CÙNG (mới nhất) phục vụ VIẾT LẠI báo cáo khoa học.
> Mọi con số dưới đây đọc trực tiếp từ file kết quả trên đĩa — không bịa, không suy diễn.
> Ngày tổng hợp: 07/09/2026.

## CẬP NHẬT 07/09 (tối) — HOÀN TẤT GÓI SỐ LIỆU BEST-EPOCH + VIẾT LẠI PAPER

- [x] Đã chạy `python -m scripts.eval_best_epoch_clean` → sinh đầy đủ metric + predictions
      cho checkpoint best-epoch (epoch 18), KHÔNG ghi đè file epoch-20 cũ:
      - `results/raw/E01_baseline_seed123_bestepoch.json` / `_predictions.csv`
      - `results/raw/E07_robust_seed123_bestepoch.json` / `_predictions.csv`
      - Kết quả KHỚP 100% hàng clean của bảng lưới suy giảm.
- [x] **PAPER ĐÃ VIẾT LẠI BẰNG BỘ B (best-epoch)**:
      `handover/paper/Paper_Robust_Face_PAD_for_eKYC.docx`
      (bản cũ số liệu 01/09 đã backup ra ngoài repo — không còn trong project).
- [x] 8 hình paper đã vẽ lại từ dữ liệu mới (`scripts/make_paper_figures.py` đã sửa:
      fix đường dẫn splits, fig1 text mới, fig3 ROC từ predictions best-epoch,
      fig4/fig5 đọc thẳng `results/tables/degradation_*.csv`, fig6 chú thích val loss mới)
      → `images/fig1-8.png` và `handover/images/fig1-8.png` đều là bản MỚI.
- [x] Đã xóa file cũ gây nhầm lẫn trong `results/raw/`: grid_baseline_stdout/stderr.txt
      + 13 log thừa từ pytest (E_test, E_rep_a/b, ablation, compare_models, run_all,
      jpeg50, noise0.03, E03_blur3, E08_robust_jpeg70, E09_ablation_jpeg, E09_robust_blur3, E01_test).
- [x] **QUYẾT ĐỊNH CUỐI CÙNG: mọi số liệu trong paper dùng BỘ B (checkpoint best-epoch,
      epoch 18)** — nhất quán với lưới suy giảm, là chuẩn khoa học (chọn model trên val).
      Bộ A (epoch 20) chỉ còn dùng cho `train_history`/runtime (loss curves, training time).
- [x] 07/09 đêm: RÚT GỌN PAPER VỀ 11 TRANG (format journal 1 cột, TNR 11pt) — giữ nguyên
      toàn bộ số liệu (re-audit 481 checks = 0 lỗi), refs [1]-[15], Tables I-XI, Figs 1-6.

---


## 1. EXPERIMENT SUMMARY

| Hạng mục | E01 Baseline | E07 Robust |
|---|---|---|
| Mã thí nghiệm | `E01_baseline_seed123` | `E07_robust_seed123` |
| File kết quả | `results/raw/E01_baseline_seed123.json` (05/09 17:34) | `results/raw/E07_robust_seed123.json` (07/09 17:19) |
| Checkpoint | `results/checkpoints/E01_baseline_seed123.pt` | `results/checkpoints/E07_robust_seed123.pt` |
| Checkpoint giữ epoch | **18** (best val_loss = 0.058560) | **18** (best val_loss = 0.068120) |
| Training time | 18 111.03 s ≈ **5h02m** | 34 404.42 s ≈ **9h33m** |
| Dataset | CelebA-Spoof mirror subset (root `data/raw/celeba_spoof_full`) | giống E01 |
| Số mẫu | total 203 215 = bona_fide 70 288 + spoof 132 927 (spoof_ratio 0.6541) | giống E01 |
| Split | subject_disjoint, seed 123: train 142 250 / val 20 322 / test **40 643** | giống E01 (cùng file splits) |
| Model | mobilenet_v2, 2 225 153 params, 8.6188 MB, input 224×224 ImageNet | giống E01 |
| Training config | epochs 20, batch 64, Adam lr 1e-4, wd 1e-5, num_workers 0, BCEWithLogitsLoss (pos_weight=false) | giống E01 + robustness (5 augmentation p=0.3, crop TẮT) |
| Threshold | **0.5** (khóa) | **0.5** (khóa) |
| Seed | 123 | 123 |
| Environment | Python 3.14.7, torch 2.14.0+cu130, torchvision 0.29.0+cu130, numpy 2.5.2, Windows 11, NVIDIA GeForce RTX 5060 Laptop GPU, git `296643e` | giống E01 |

Chi tiết khối robustness của E07 (lưu trong checkpoint):
jpeg quality [50,90] p=0.3 · resize scale [0.5,1.0] p=0.3 · blur sigma [0.5,2.0] p=0.3 · noise std [0.005,0.03] p=0.3 · brightness factor [0.7,1.3] p=0.3 · crop TẮT.

> Lưu ý: trường `"dataset"` trong JSON kết quả ghi `"celeba_spoof"` (tên cấu hình lưu trong
> checkpoint), nhưng `dataset.root` = `data/raw/celeba_spoof_full` — đây là dataset FULL 200k,
> KHÔNG phải pilot 18k. Số mẫu trong log khẳng định: total 203 215, test 40 643.

---

## 2. FINAL METRICS — clean test (40 643 ảnh, threshold 0.5)

Quy ước nhãn: **0 = bona_fide, 1 = spoof; positive = spoof**.
APCER = spoof bị bỏ lọt (FN/(FN+TP)); BPCER = bona_fide bị gọi nhầm (FP/(FP+TN)); ACER = trung bình 2 lỗi.

### Bộ A — model epoch 20 (đánh giá trong bộ nhớ cuối training; từ JSON kết quả)

| Metric | E01 Baseline | E07 Robust |
|---|---|---|
| Accuracy | 0.977684 | 0.976700 |
| Precision | 0.983419 | 0.985221 |
| Recall | 0.982496 | 0.979115 |
| F1 | 0.982957 | 0.982159 |
| ROC-AUC | 0.997219 | 0.997093 |
| PR-AUC | 0.998373 | 0.998354 |
| APCER | 0.017504 | 0.020885 |
| BPCER | 0.031453 | 0.027887 |
| ACER | 0.024479 | 0.024386 |

**Confusion matrix** (hàng = nhãn thật, cột = dự đoán; tính lại từ `*_predictions.csv` — khớp 100% với JSON):

| E01 | Dự đoán bona_fide | Dự đoán spoof | | E07 | Dự đoán bona_fide | Dự đoán spoof |
|---|---|---|---|---|---|---|
| Thật bona_fide | TN = 13 580 | FP = 441 | | Thật bona_fide | TN = 13 630 | FP = 391 |
| Thật spoof | FN = 466 | TP = 26 156 | | Thật spoof | FN = 556 | TP = 26 066 |

### Bộ B — checkpoint BEST-EPOCH (epoch 18; hàng `clean` của bảng lưới suy giảm)

| Metric | E01 Baseline | E07 Robust |
|---|---|---|
| F1 | 0.986776 | 0.983057 |
| ROC-AUC | 0.997767 | 0.996944 |
| PR-AUC | 0.998667 | 0.998255 |
| APCER | 0.009128 | 0.014875 |
| BPCER | 0.033093 | 0.036231 |
| ACER | 0.021111 | 0.025553 |
| Accuracy | 0.982605 | 0.977758 |

> KHÔNG có confusion matrix riêng cho Bộ B: lần đánh giá clean của grid KHÔNG lưu predictions CSV.
> Confusion matrix ở trên chỉ có cho Bộ A (epoch 20).

---

## 3. DEGRADATION RESULTS — lưới 16 điều kiện, E01 vs E07

Nguồn: `results/tables/degradation_baseline.csv` (06/09) + `degradation_robust.csv` (07/09).
Đánh giá từ checkpoint best-epoch (epoch 18), cùng test set 40 643 ảnh, threshold 0.5, suy giảm TẤT ĐỊNH.
Định nghĩa mức suy giảm (từ `experiments/eval_degradation_grid.py` SEVERITY_GRID):
jpeg q=90/70/50/30 · resize scale 0.75/0.50/0.25 · blur light (k3,σ0.6) / medium (k7,σ1.8) / strong (k11,σ3.0) · noise std 0.005/0.015/0.03 · brightness factor 0.6/1.0/1.4.

| Condition | Severity | E01 F1 | E07 F1 | ΔF1 | E01 AUC | E07 AUC | ΔAUC | E01 PR-AUC | E07 PR-AUC | ΔPR-AUC | E01 APCER | E07 APCER | E01 BPCER | E07 BPCER | E01 ACER | E07 ACER | ΔACER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | – | 0.9868 | 0.9831 | −0.0037 | 0.9978 | 0.9969 | −0.0008 | 0.9987 | 0.9983 | −0.0004 | 0.0091 | 0.0149 | 0.0331 | 0.0362 | 0.0211 | 0.0256 | +0.0044 |
| jpeg | 90 | 0.9866 | 0.9830 | −0.0036 | 0.9977 | 0.9969 | −0.0008 | 0.9987 | 0.9982 | −0.0004 | 0.0095 | 0.0151 | 0.0332 | 0.0362 | 0.0213 | 0.0256 | +0.0043 |
| jpeg | 70 | 0.9747 | 0.9795 | +0.0049 | 0.9957 | 0.9962 | +0.0006 | 0.9974 | 0.9978 | +0.0004 | 0.0379 | 0.0234 | 0.0230 | 0.0331 | 0.0304 | 0.0282 | −0.0022 |
| jpeg | 50 | 0.9507 | 0.9753 | +0.0246 | 0.9932 | 0.9955 | +0.0023 | 0.9959 | 0.9974 | +0.0015 | 0.0853 | 0.0332 | 0.0184 | 0.0300 | 0.0518 | 0.0316 | −0.0202 |
| jpeg | 30 | 0.8911 | 0.9674 | +0.0763 | 0.9890 | 0.9943 | +0.0053 | 0.9934 | 0.9967 | +0.0034 | 0.1915 | 0.0501 | 0.0118 | 0.0265 | 0.1016 | 0.0383 | −0.0633 |
| resize | 75 | 0.9679 | 0.9806 | +0.0127 | 0.9944 | 0.9961 | +0.0017 | 0.9968 | 0.9978 | +0.0011 | 0.0053 | 0.0123 | 0.1153 | 0.0509 | 0.0603 | 0.0316 | −0.0287 |
| resize | 50 | 0.9339 | 0.9783 | +0.0444 | 0.9849 | 0.9955 | +0.0107 | 0.9915 | 0.9975 | +0.0060 | 0.0060 | 0.0122 | 0.2558 | 0.0600 | 0.1309 | 0.0361 | −0.0948 |
| resize | 25 | 0.8672 | 0.9704 | +0.1032 | 0.9405 | 0.9922 | +0.0517 | 0.9664 | 0.9956 | +0.0292 | 0.0078 | 0.0178 | 0.5622 | 0.0802 | 0.2850 | 0.0490 | −0.2360 |
| blur | light | 0.9785 | 0.9815 | +0.0030 | 0.9964 | 0.9964 | 0.0000 | 0.9979 | 0.9980 | +0.0001 | 0.0054 | 0.0130 | 0.0725 | 0.0460 | 0.0390 | 0.0295 | −0.0095 |
| blur | medium | 0.8967 | 0.9757 | +0.0791 | 0.9646 | 0.9947 | +0.0301 | 0.9802 | 0.9970 | +0.0168 | 0.0067 | 0.0146 | 0.4218 | 0.0653 | 0.2143 | 0.0400 | −0.1743 |
| blur | strong | 0.8489 | 0.9677 | +0.1188 | 0.9161 | 0.9906 | +0.0745 | 0.9523 | 0.9947 | +0.0424 | 0.0089 | 0.0221 | 0.6532 | 0.0821 | 0.3310 | 0.0521 | −0.2789 |
| noise | low | 0.9849 | 0.9832 | −0.0017 | 0.9972 | 0.9969 | −0.0003 | 0.9983 | 0.9983 | −0.0001 | 0.0146 | 0.0146 | 0.0000 | 0.0295 | 0.0363 | 0.0221 | 0.0255 | +0.0034 |
| noise | medium | 0.9283 | 0.9828 | +0.0545 | 0.9869 | 0.9968 | +0.0099 | 0.9925 | 0.9982 | +0.0056 | 0.1242 | 0.0157 | 0.0212 | 0.0357 | 0.0727 | 0.0257 | −0.0470 |
| noise | high | 0.8487 | 0.9781 | +0.1294 | 0.9638 | 0.9957 | +0.0319 | 0.9799 | 0.9976 | +0.0177 | 0.2532 | 0.0251 | 0.0248 | 0.0352 | 0.1390 | 0.0302 | −0.1089 |
| brightness | dark | 0.9190 | 0.9446 | +0.0255 | 0.9628 | 0.9805 | +0.0177 | 0.9788 | 0.9890 | +0.0102 | 0.0261 | 0.0184 | 0.2762 | 0.1839 | 0.1512 | 0.1012 | −0.0500 |
| brightness | normal | 0.9868 | 0.9831 | −0.0037 | 0.9978 | 0.9969 | −0.0008 | 0.9987 | 0.9983 | −0.0004 | 0.0091 | 0.0149 | 0.0331 | 0.0362 | 0.0211 | 0.0256 | +0.0044 |
| brightness | bright | 0.7985 | 0.9571 | +0.1586 | 0.9744 | 0.9876 | +0.0131 | 0.9848 | 0.9926 | +0.0078 | 0.3300 | 0.0578 | 0.0157 | 0.0505 | 0.1728 | 0.0542 | −0.1187 |

Δ = E07 − E01. ΔACER âm = robust tốt hơn; ΔF1 dương = robust tốt hơn.
File JSON gốc từng điều kiện: `results/raw/E02_{jpeg|resize|blur|noise|brightness}*_seed123.json` (baseline) và `E08..E12_*_seed123.json` (robust).

---

## 4. AGGREGATE RESULTS

**Chính (16 điều kiện suy giảm, KHÔNG gồm hàng clean; gồm brightness normal):**

| Chỉ số | E01 Baseline | E07 Robust |
|---|---|---|
| Mean F1 | 0.922635 | 0.974256 |
| Mean ROC-AUC | 0.978465 | 0.993941 |
| Mean PR-AUC | 0.987710 | 0.996549 |
| Mean APCER | 0.070099 | 0.022510 |
| Mean BPCER | 0.160478 | 0.055519 |
| Mean ACER | 0.115289 | 0.039014 |
| Worst-case F1 | 0.798478 (brightness bright) | 0.944555 (brightness dark) |
| Worst-case ACER | 0.331014 (blur strong) | 0.101153 (brightness dark) |

**Phụ (17 hàng gồm cả clean):**

| Chỉ số | E01 Baseline | E07 Robust |
|---|---|---|
| Mean F1 | 0.926408 | 0.974774 |
| Mean ACER | 0.109749 | 0.038223 |
| Worst-case F1 | 0.798478 | 0.944555 |
| Worst-case ACER | 0.331014 | 0.101153 |

---

## 5. TRAINING CURVES — train_loss / val_loss từng epoch

Nguồn: trường `train_history` trong JSON kết quả (E01: `results/raw/E01_baseline_seed123.json`, E07: `E07_robust_seed123.json`).

| Epoch | E01 train | E01 val | E07 train | E07 val |
|---|---|---|---|---|
| 1 | 0.275718 | 0.206510 | 0.321514 | 0.226664 |
| 2 | 0.182722 | 0.148745 | 0.232550 | 0.164363 |
| 3 | 0.140734 | 0.125960 | 0.192001 | 0.143393 |
| 4 | 0.116349 | 0.112974 | 0.167837 | 0.134275 |
| 5 | 0.099160 | 0.108852 | 0.148633 | 0.132910 |
| 6 | 0.086470 | 0.090847 | 0.135628 | 0.103025 |
| 7 | 0.075919 | 0.078624 | 0.123754 | 0.100407 |
| 8 | 0.067544 | 0.075583 | 0.114346 | 0.089919 |
| 9 | 0.060123 | 0.069614 | 0.105899 | 0.084532 |
| 10 | 0.054999 | 0.073080 | 0.100478 | 0.096559 |
| 11 | 0.048647 | 0.067766 | 0.095183 | 0.087379 |
| 12 | 0.044076 | 0.063929 | 0.087240 | 0.085235 |
| 13 | 0.040548 | 0.071396 | 0.082040 | 0.091784 |
| 14 | 0.036187 | 0.067326 | 0.078234 | 0.092131 |
| 15 | 0.034159 | 0.062784 | 0.075555 | 0.071530 |
| 16 | 0.030985 | 0.065384 | 0.071837 | 0.071564 |
| 17 | 0.028193 | 0.063596 | 0.067468 | 0.073313 |
| 18 | 0.026481 | **0.058560** (best) | 0.065136 | **0.068120** (best) |
| 19 | 0.025827 | 0.060059 | 0.061023 | 0.073539 |
| 20 | 0.023351 | 0.063627 | 0.059229 | 0.071880 |

Checkpoint đã lưu cho cả 2 model = **epoch 18** (epoch có val_loss nhỏ nhất).

---

## 6. FIGURES / FILES

### results/figures/ (sinh từ dữ liệu MỚI bởi `eval_degradation_grid.py` — mỗi hình: trục x = các mức severity [clean, ...], 2 đường F1 (chấm tròn) và ACER (chấm vuông), ylim [0,1])

| File | Sinh lúc | Nội dung |
|---|---|---|
| `fig_baseline_jpeg.png` | 06/09 | F1/ACER của E01 theo JPEG quality (clean, 90, 70, 50, 30) |
| `fig_baseline_resize.png` | 06/09 | F1/ACER của E01 theo resize scale (clean, 75, 50, 25) |
| `fig_baseline_blur.png` | 06/09 | F1/ACER của E01 theo blur (clean, light, medium, strong) |
| `fig_baseline_noise.png` | 06/09 | F1/ACER của E01 theo noise (clean, low, medium, high) |
| `fig_baseline_brightness.png` | 06/09 | F1/ACER của E01 theo brightness (clean, dark, normal, bright) |
| `fig_robust_jpeg.png` | 07/09 | F1/ACER của E07 theo JPEG quality |
| `fig_robust_resize.png` | 07/09 | F1/ACER của E07 theo resize scale |
| `fig_robust_blur.png` | 07/09 | F1/ACER của E07 theo blur |
| `fig_robust_noise.png` | 07/09 | F1/ACER của E07 theo noise |
| `fig_robust_brightness.png` | 07/09 | F1/ACER của E07 theo brightness |

### handover/images/fig1–fig8.png — BẢN CŨ (01/09, số liệu CŨ — KHÔNG dùng cho báo cáo mới)

Sinh bởi `scripts/make_paper_figures.py` từ JSON kết quả CŨ (F1 .825/.850):
fig1 pipeline · fig2 ảnh suy giảm mẫu · fig3 ROC E01 vs E07 · fig4 F1/ACER theo severity · fig5 bar mean/worst-case · fig6 loss curves · fig7-8 ảnh code màu.
→ Cần chạy lại script để sinh hình với số MỚI (xem mục 9 — script hiện có 1 đường dẫn hỏng).

---

## 7. REPRODUCIBILITY

| Hạng mục | Giá trị / lệnh |
|---|---|
| Dataset | CelebA-Spoof mirror subset, root `data/raw/celeba_spoof_full` (203 215 ảnh) |
| Split | subject_disjoint, seed 123; file KHÓA `data/splits/celeba_spoof_seed123_subject_disjoint.json` (train 142 250 / val 20 322 / test 40 643) |
| Seed | 123 (toàn dự án) |
| Model | mobilenet_v2 (2 225 153 params, 8.6188 MB), input 224×224, normalize ImageNet |
| Threshold | 0.5 — khóa cho MỌI thí nghiệm |
| Git commit (code lúc train + eval) | `296643e1ec9194e4cb5d4c6023fa5e3fd574e40f` — "save the best epoch" (04/09/2026). Cả 2 JSON kết quả đều ghi đúng commit này. |
| Train E01 | `python -m experiments.train_baseline --config <config gốc: name=celeba_spoof, root=data/raw/celeba_spoof_full, num_workers=0>` (config chính xác lưu trong checkpoint) |
| Train E07 | `python -m experiments.train_robust --config <như E01> --robustness configs/robustness.yaml` |
| Grid baseline | `python -m experiments.eval_degradation_grid --checkpoint results/checkpoints/E01_baseline_seed123.pt --tag baseline` |
| Grid robust | `python -m experiments.eval_degradation_grid --checkpoint results/checkpoints/E07_robust_seed123.pt --tag robust` |
| Định nghĩa lưới | `SEVERITY_GRID` trong `experiments/eval_degradation_grid.py`: jpeg q 90/70/50/30 · resize 0.75/0.50/0.25 · blur (k3,0.6)/(k7,1.8)/(k11,3.0) · noise 0.005/0.015/0.03 · brightness 0.6/1.0/1.4 — suy giảm TẤT ĐỊNH, áp TRƯỚC transform chuẩn |
| Không retrain khi đánh giá suy giảm | đúng (mục 23 tài liệu) — grid chỉ load checkpoint |
| Lưu ý working tree | hiện có sửa chưa commit: `configs/*.yaml` num_workers → 0 (chỉ ảnh hưởng tốc độ tải dữ liệu, KHÔNG ảnh hưởng số liệu) + `documents/PROGRESS.md` |

---

## 8. RAW DATA PATHS (các file quan trọng)

```
KẾT QUẢ CHÍNH:
results/raw/E01_baseline_seed123.json            metric + train_history E01 (Bộ A)
results/raw/E01_baseline_seed123.csv             1 dòng tóm tắt
results/raw/E01_baseline_seed123.log             log train E01 (05/09, epoch-by-epoch, thời gian từng epoch)
results/raw/E01_baseline_seed123_predictions.csv 40 643 dòng: path, subject_id, attack_type, label, probability_spoof, prediction, correct
results/raw/E07_robust_seed123.json              metric + train_history E07 (Bộ A)
results/raw/E07_robust_seed123.csv
results/raw/E07_robust_seed123.log               log train E07 (07/09)
results/raw/E07_robust_seed123_predictions.csv   40 643 dòng (cấu trúc như trên)

DEGRADATION (16 điều kiện x 2 model = 32 bộ):
results/raw/E02_{jpeg90,jpeg70,jpeg50,jpeg30}_seed123.json (+ .csv/.log/_predictions.csv)
results/raw/E03_{resize75,resize50,resize25}_seed123.json  ...
results/raw/E04_{blurlight,blurmedium,blurstrong}_seed123.json ...
results/raw/E05_{noiselow,noisemedium,noisehigh}_seed123.json ...
results/raw/E06_{brightnessdark,brightnessnormal,brightnessbright}_seed123.json ...
results/raw/E08..E12_{...}_seed123.json          (tương ứng cho robust E07)
results/tables/degradation_baseline.csv          bảng tổng hợp E01: condition,severity,f1,roc_auc,pr_auc,apcer,bpcer,acer,accuracy (17 dòng)
results/tables/degradation_robust.csv            bảng tổng hợp E07
results/raw/grid_baseline_stdout.txt             stdout lần chạy grid baseline

CHECKPOINT:
results/checkpoints/E01_baseline_seed123.pt      best epoch 18 (val_loss 0.058560) + config + optimizer state
results/checkpoints/E07_robust_seed123.pt        best epoch 18 (val_loss 0.068120) + config (gồm khối robustness) + optimizer state

FIGURES:
results/figures/fig_baseline_{jpeg,resize,blur,noise,brightness}.png
results/figures/fig_robust_{jpeg,resize,blur,noise,brightness}.png

SPLITS + DATASET:
data/splits/celeba_spoof_seed123_subject_disjoint.json   splits 200k (KHÓA)
data/raw/celeba_spoof_full/                              dataset 200k

SCRIPT NGUỒN SỐ LIỆU:
experiments/eval_degradation_grid.py   (định nghĩa lưới + sinh tables/figures)
experiments/train_baseline.py, train_robust.py, _common.py   (logic train + best-epoch)
scripts/make_paper_figures.py          (sinh fig1-fig8 của paper — bản cũ)
```

---

## 9. CHECK FOR MISSING DATA (cần cho báo cáo nhưng project CHƯA có)

1. **Confusion matrix của checkpoint best-epoch (epoch 18)**: chưa có — grid clean row không lưu predictions. Predictions CSV hiện có thuộc model epoch 20 (Bộ A). Nếu báo cáo dùng Bộ B thì phải chạy lại đánh giá clean có lưu predictions.
2. **Hình paper mới**: `handover/images/fig1–fig8.png` là số liệu CŨ (01/09). `scripts/make_paper_figures.py` phải chạy lại để sinh hình với số mới; LƯU Ý script đang trỏ `data/splits/celeba_spoof_full_seed123_subject_disjoint.json` (KHÔNG tồn tại — file thật là `celeba_spoof_seed123_subject_disjoint.json`) nên fig2 sẽ lỗi ngay khi chạy.
3. **Fig tổng hợp mới**: chưa có ROC-curve, loss-curve, aggregate bar cho số liệu mới (chỉ có 10 fig theo từng loại suy giảm trong `results/figures/`).
4. **Số liệu phụ P1** (tùy chọn nhưng bài cũ có thể đã nhắc): data-scale 18k vs 200k, ablation từng augmentation, webcam eval cho model 200k, latency/model-size — tất cả vẫn "chưa đo" với model mới.
5. **Nhất quán tên dataset trong báo cáo**: JSON ghi `"celeba_spoof"` nhưng dữ liệu thật là `celeba_spoof_full` (200k) — tránh nhầm với pilot 18k.
6. **Chọn 1 bộ clean metrics** (Bộ A epoch-20 vs Bộ B epoch-18) — đang tồn tại song song; báo cáo phải nêu rõ dùng bộ nào.
7. **Tài liệu chưa cập nhật số mới**: `handover/documents/NHAT_KY_THI_NGHIEM.md` mục 5.1–5.3 vẫn ghi số CŨ (F1 .825, AUC .963, ACER .157...); `handover/paper/Paper_Robust_Face_PAD_for_eKYC.docx` dùng toàn bộ số cũ.
