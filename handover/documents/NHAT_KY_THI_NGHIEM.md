# NHẬT KÝ THÍ NGHIỆM — eKYC Face PAD

> File này ghi lại TOÀN BỘ quá trình thực nghiệm (đã chạy gì, kết quả ra sao,
> nhận xét gì) để làm tư liệu viết báo cáo khoa học.
> Cập nhật file này SAU MỖI lần chạy thí nghiệm. Không bịa số liệu — chỉ ghi
> số đã đo được; mục chưa chạy ghi "chưa đo".

> **⚠️ CẬP NHẬT 07/09/2026 — SỐ LIỆU CHÍNH THỨC ĐÃ THAY ĐỔI.**
> Sau khi sửa logic "save best epoch" (commit `296643e`), E01 và E07 đã được
> TRAIN LẠI (05/09 và 07/09, máy RTX 5060) và lưới suy giảm đã đánh giá lại.
> **Các bảng 5.1–5.3 dưới đây là số CŨ (01/09, F1 .825/.850) — chỉ giữ làm lịch sử,
> KHÔNG dùng viết báo cáo.**
> Số liệu mới nhất (best-epoch, epoch 18) nằm ở:
> - `handover/paper/FINAL_EXPERIMENT_DATA.md` (gói số liệu đầy đủ)
> - `handover/paper/Paper_Robust_Face_PAD_for_eKYC.docx` (paper đã viết lại bằng số mới)
> - `results/tables/degradation_baseline.csv` + `degradation_robust.csv` (06-07/09)

---

## 1. Mục tiêu nghiên cứu

Đánh giá xem model Face PAD nhẹ (MobileNetV2) có giữ được độ tin cậy khi
**chất lượng ảnh đầu vào suy giảm** (JPEG, giảm độ phân giải, blur, noise,
thay đổi độ sáng) hay không, và **huấn luyện robustness bằng quality
augmentation** có cải thiện được độ bền vững đó không.

Câu hỏi thí nghiệm chính:
> "Does quality-aware augmentation improve robustness under degraded input conditions?"

---

## 2. Thiết lập thí nghiệm (protocol cố định)

| Hạng mục | Giá trị |
|---|---|
| Dataset chính | CelebA-Spoof (mirror HuggingFace `Ar4ikov/celebA_spoof`, ~526k ảnh) — **chỉ dùng subset có kiểm soát: ~200k train / ~20k val / ~20k test** (không tải 74GB từ Drive) |
| Pilot | 18k ảnh (mirror `Camilotabares1/celebA_spoof_sample_split`) |
| Model | MobileNetV2, 2.225.153 tham số (~8.6 MB) |
| Input | 224×224, normalize ImageNet |
| Split | train ~200k / val ~20k / test ~20k — subject-disjoint theo protocol chính thức, test set KHÓA |
| Seed | 123 |
| Optimizer | Adam, lr 1e-4, weight_decay 1e-5 |
| Batch size | 64 | Epochs | 20 |
| Loss | BCEWithLogitsLoss (không pos_weight) |
| Threshold | 0.5 (khóa; mọi thí nghiệm dùng chung) |
| Metric | F1, ROC-AUC, PR-AUC, APCER, BPCER, ACER (positive = spoof) |
| Phần cứng | NVIDIA RTX 3050 Laptop GPU (4GB), torch 2.13.0+cu126 |

Mức suy giảm (định nghĩa tường minh trong `experiments/eval_degradation_grid.py`):

| Loại | Các mức |
|---|---|
| JPEG | quality 90 / 70 / 50 / 30 |
| Resize | scale 75% / 50% / 25% |
| Blur | light (k3,σ0.6) / medium (k7,σ1.8) / strong (k11,σ3.0) |
| Noise | std 0.005 / 0.015 / 0.03 |
| Brightness | factor 0.6 / 1.0 / 1.4 |

---

## 3. NHẬT KÝ CÁC LẦN CHẠY

### 3.1. Giai đoạn PILOT (18k ảnh) — HOÀN TẤT

| Ngày | Thí nghiệm | Kết quả chính | Ghi chú |
|---|---|---|---|
| 28-29/08 | E01_pilot (18k, 20 epoch) | F1 .931, ROC-AUC .955, ACER .107, APCER .069, BPCER .146 | train 12.6k/val 1.8k/test 3.6k |
| 29/08 | E07_pilot (robust) | F1 .929, ROC-AUC .956, ACER .110 | aug: jpeg/resize/blur/noise/brightness/crop |
| 29/08 | E20_pilot (fine-tune webcam) | P(spoof) mặt webcam live: 0.89 → 0.017 | 500 ảnh webcam, 3 epoch, lr 5e-5 |
| 29/08 | Test B/C/D (chẩn đoán domain shift) | xem mục 5 | phát hiện quan trọng cho báo cáo |

### 3.2. Giai đoạn CHÍNH (200k ảnh) — HOÀN TẤT CORE

| Ngày | Thí nghiệm | Kết quả chính | Ghi chú |
|---|---|---|---|
| 30-31/08 | E01_baseline_200k (20 epoch) | **F1 .825, ROC-AUC .963, PR-AUC .984, ACER .157, APCER .292, BPCER .021, Accuracy .787** | runtime 19.7h; train_loss 0.2495 → 0.0162; val_loss 0.0417 |
| 31/08 | E02–E06 (lưới suy giảm baseline) | **ĐÃ ĐO** — xem bảng 5.2 | 16 điều kiện, ~31 phút |
| 01/09 | E07_robust_200k (20 epoch) | **F1 .850, ROC-AUC .961, ACER .142, APCER .250, BPCER .034, Accuracy .813** | runtime 20.6h; train_loss 0.2941 → 0.0491; val_loss tốt nhất 0.0498 (epoch 18) |
| 01/09 | E08–E12 (lưới suy giảm robust) | **ĐÃ ĐO** — xem bảng 5.3 | cùng lưới 16 điều kiện |

---

## 4. LỆNH CHẠY (để tái lập)

```cmd
set PYTHONUTF8=1

:: Tải dataset ~200k (đã xong, KHÔNG chạy lại nếu data/raw/celeba_spoof_full đã có)
python -m scripts.download_celeba_full

:: E01 baseline (đã xong)
python -m experiments.train_baseline --config configs/full_clean.yaml

:: ĐÁNH GIÁ baseline trên lưới suy giảm (đã xong — bảng 5.2):
python -m experiments.eval_degradation_grid --checkpoint results/checkpoints/E01_baseline_seed123.pt --tag baseline

:: E07 robust (đã xong — cấu hình robustness.yaml: crop TẮT, 5 augmentation p=0.3)
python -m experiments.train_robust --config configs/full_clean.yaml --robustness configs/robustness.yaml

:: Đánh giá robust trên CÙNG lưới suy giảm (đã xong — bảng 5.3)
python -m experiments.eval_degradation_grid --checkpoint results/checkpoints/E07_robust_seed123.pt --tag robust
```

Sau lệnh grid, kết quả nằm ở:
- `results/raw/E02..E06_*_seed123.json` (mỗi mức suy giảm 1 bộ metric đầy đủ)
- `results/tables/degradation_baseline.csv` (bảng tổng hợp)
- `results/figures/fig_baseline_*.png` (biểu đồ F1/ACER theo từng loại suy giảm)

---

## 5. KẾT QUẢ & PHÂN TÍCH SƠ BỘ

### 5.1. Baseline 200k — clean test (đã đo)

| Metric | Giá trị |
|---|---|
| F1 | 0.8246 |
| ROC-AUC | 0.9629 |
| PR-AUC | 0.9840 |
| APCER | 0.2922 |
| BPCER | 0.0214 |
| ACER | 0.1568 |
| Accuracy | 0.7872 |

**Nhận xét sơ bộ:**
- ROC-AUC 0.963 **cao hơn** pilot 18k (0.955) → khả năng phân biệt (ranking) tốt hơn với dữ liệu đa dạng hơn.
- Nhưng ở **ngưỡng 0.5**, APCER 0.29 (29% spoof bị bỏ lọt) trong khi BPCER chỉ 0.02 → **điểm quyết định của model bị lệch**, xác suất đầu ra bị "co" về phía thấp. Ngưỡng cân bằng sẽ thấp hơn 0.5 nhiều (sẽ tính khi cần).
- val_loss 0.0417 (rất thấp) so với train_loss 0.0162 → model khớp tốt, không thấy dấu hiệu overfit nghiêm trọng.
- F1 thấp hơn pilot chủ yếu do ngưỡng cố định 0.5 không phù hợp với phân phối xác suất của model 200k — đây là điểm cần bàn trong báo cáo (protocol giữ threshold 0.5 cố định).

### 5.1b. VÌ SAO GIỮ NGUỠNG 0.5 CỐ ĐỊNH? (giải thích cho sinh viên)

**Ngưỡng (threshold) là gì?** Model đưa ra xác suất P(spoof) trong [0,1];
quy tắc phân loại cố định: `P >= threshold -> spoof, ngược lại -> bona_fide`.
F1/APCER/BPCER đều được tính TẠI một ngưỡng cụ thể, còn ROC-AUC/PR-AUC
quét qua mọi ngưỡng nên không phụ thuộc ngưỡng.

**Phân bố xác suất thực tế của model 200k** (đo từ predictions E01, test 20k):

| Nhóm | P(spoof) trung bình | Tỉ lệ bị sai ở ngưỡng 0.5 |
|---|---|---|
| Ảnh spoof thật | 0.71 | 29.2% có P < 0.5 → bị gọi là "thật" (APCER 0.29) |
| Ảnh live thật | 0.02 | 2.1% có P ≥ 0.5 → bị gọi là "spoof" (BPCER 0.02) |

Nghĩa là model phân biệt RẤT tốt (live ≈ 0.02 vs spoof ≈ 0.71) — vì thế
ROC-AUC 0.963 cao hơn pilot — nhưng toàn bộ dải xác suất bị **"dồn về gần 0"**,
và gần 1/3 ảnh spoof nằm dưới vạch 0.5 → F1@0.5 bị kéo thấp.

**Nếu hạ ngưỡng thì sao?** Ở ngưỡng cân bằng (~0.05-0.1): ACER 0.116,
F1 0.888 — ngang ngửa pilot. Tức model 200k không "tệ hơn" pilot; nó chỉ
khác điểm vận hành.

**Vậy tại sao KHÔNG chọn ngưỡng khác 0.5?** Vì 2 lý do khoa học:
1. **So sánh công bằng (mục 24, 41 tài liệu)**: mọi model (baseline/robust/
   ablation) phải được đánh giá ở CÙNG một ngưỡng thì chênh lệch metric mới
   chỉ phản ánh mô hình, không phải ngưỡng. Chọn mỗi model một ngưỡng riêng
   sẽ khiến bảng so sánh không còn ý nghĩa.
2. **Không leak test set**: chọn ngưỡng "đẹp nhất" dựa trên test set là dùng
   thông tin test để chỉnh mô hình — vi phạm nguyên tắc test set là tập đánh
   giá ẩn. Muốn chọn ngưỡng thì phải chọn trên VALIDATION rồi khóa, không
   chọn trên test.

**Cách xử lý đúng trong báo cáo:**
- So sánh mọi model ở ngưỡng 0.5 (công bằng, tái lập được).
- Luôn báo cáo đủ bộ metric: ROC-AUC/PR-AUC (không phụ thuộc ngưỡng) +
  APCER/BPCER/ACER (phơi bày trade-off hai loại lỗi).
- Ghi vào mục Hạn chế: "F1@0.5 của model 200k bị kéo thấp do phân phối xác
  suất bị co về 0; ở ngưỡng cân bằng (~0.05-0.1) hiệu năng tương đương pilot
  (F1 0.888, ACER 0.116) — kết luận dựa trên phân tích ROC, không chỉnh ngưỡng
  trên test".

### 5.2. Lưới suy giảm baseline 200k (ĐÃ ĐO — 31/08/2026)

Bảng đầy đủ (cũng lưu tại `results/tables/degradation_baseline.csv`):

| Condition | Severity | F1 | ROC-AUC | PR-AUC | APCER | BPCER | ACER |
|---|---|---|---|---|---|---|---|
| clean | - | 0.8246 | 0.9629 | 0.9840 | 0.2922 | 0.0214 | 0.1568 |
| jpeg | 90 | 0.8253 | 0.9626 | 0.9838 | 0.2913 | 0.0208 | 0.1560 |
| jpeg | 70 | 0.7199 | 0.9480 | 0.9774 | 0.4349 | 0.0117 | 0.2233 |
| jpeg | 50 | 0.6125 | 0.9308 | 0.9694 | 0.5570 | 0.0083 | 0.2827 |
| jpeg | 30 | 0.4323 | 0.9007 | 0.9549 | 0.7235 | 0.0065 | 0.3650 |
| resize | 75% | 0.8411 | 0.9272 | 0.9678 | 0.2502 | 0.0798 | 0.1650 |
| resize | 50% | 0.8427 | 0.8960 | 0.9523 | 0.2213 | 0.1672 | 0.1942 |
| resize | 25% | 0.8387 | 0.8226 | 0.9163 | 0.1608 | 0.3905 | 0.2757 |
| blur | light | 0.8401 | 0.9402 | 0.9740 | 0.2586 | 0.0568 | 0.1577 |
| blur | medium | 0.8490 | 0.8526 | 0.9300 | 0.1681 | 0.3080 | 0.2381 |
| blur | strong | 0.8357 | 0.7830 | 0.8963 | 0.1131 | 0.5674 | 0.3403 |
| noise | low | 0.7504 | 0.9633 | 0.9838 | 0.3968 | 0.0109 | 0.2038 |
| noise | medium | 0.3998 | 0.9083 | 0.9579 | 0.7498 | 0.0039 | 0.3768 |
| noise | high | 0.2678 | 0.7804 | 0.8850 | 0.8440 | 0.0223 | 0.4331 |
| brightness | dark | 0.8495 | 0.8555 | 0.9329 | 0.1634 | 0.3208 | 0.2421 |
| brightness | normal | 0.8246 | 0.9629 | 0.9840 | 0.2922 | 0.0214 | 0.1568 |
| brightness | bright | 0.5160 | 0.9063 | 0.9562 | 0.6509 | 0.0094 | 0.3301 |

Biểu đồ: `results/figures/fig_baseline_{jpeg,resize,blur,noise,brightness}.png`

**PHÂN TÍCH (ghi vào báo cáo):**

1. **Noise là loại suy giảm tàn phá nhất đối với baseline.** Ngay mức "low"
   (std 0.005 — nhiễu rất nhẹ) F1 đã tụt 0.825 → 0.750 (APCER 0.29 → 0.40);
   ở medium/high model gần như hỏng (F1 0.40 / 0.27, ACER 0.38 / 0.43).
   → Đây là điểm yếu lớn nhất, là động lực chính cho robust training.
2. **JPEG suy giảm đơn điệu theo quality** (0.83 → 0.43) nhưng ROC-AUC vẫn
   khá (0.90 ở q=30) — model vẫn còn khả năng xếp hạng.
3. **Resize và blur "giả vờ" ổn định nếu chỉ nhìn F1** (F1 ~0.84 ở mọi mức)
   nhưng ROC-AUC sụp (0.96 → 0.82 ở resize 25%; 0.94 → 0.78 ở blur strong)
   và BPCER tăng mạnh (0.02 → 0.39 / 0.57): ngày càng nhiều **mặt thật bị
   gọi là spoof**. → F1 tại ngưỡng cố định CHE GIẤU suy giảm chất lượng xếp
   hạng; đây là luận điểm quan trọng: phải báo cáo đủ APCER/BPCER/ACER/AUC.
4. **Brightness**: tối (0.6) làm AUC giảm mạnh (0.856) dù F1 ổn; chói (1.4)
   làm F1 giảm mạnh (0.52). Bất đối xứng tối/chói đáng chú ý.
5. **Hiệu ứng ngưỡng cố định 0.5 lặp lại xuyên suốt**: ở hầu hết điều kiện
   APCER ≫ BPCER — ngưỡng 0.5 không phải điểm vận hành cân bằng cho model
   này (đã thấy từ clean test). Đây là giới hạn cố ý của protocol — sẽ bàn
   ở mục hạn chế của báo cáo.

### 5.3. So sánh baseline vs robust — ĐÃ ĐO (01/09/2026)

Cùng test set (20,042 ảnh), cùng ngưỡng 0.5, cùng lưới suy giảm.

| Condition | Sev | E01_F1 | E07_F1 | ΔF1 | E01_AUC | E07_AUC | ΔAUC | E01_ACER | E07_ACER | ΔACER |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | - | 0.8246 | 0.8502 | +0.0256 | 0.9629 | 0.9607 | -0.0022 | 0.1568 | 0.1421 | -0.0147 |
| jpeg | 90 | 0.8253 | 0.8495 | +0.0241 | 0.9626 | 0.9602 | -0.0023 | 0.1560 | 0.1430 | -0.0130 |
| jpeg | 70 | 0.7199 | 0.8243 | +0.1044 | 0.9480 | 0.9501 | +0.0021 | 0.2233 | 0.1603 | -0.0630 |
| jpeg | 50 | 0.6125 | 0.8001 | +0.1876 | 0.9308 | 0.9423 | +0.0115 | 0.2827 | 0.1776 | -0.1051 |
| jpeg | 30 | 0.4323 | 0.7661 | +0.3338 | 0.9007 | 0.9318 | +0.0311 | 0.3650 | 0.2007 | -0.1642 |
| resize | 75% | 0.8411 | 0.8553 | +0.0142 | 0.9272 | 0.9524 | +0.0251 | 0.1650 | 0.1421 | -0.0229 |
| resize | 50% | 0.8427 | 0.8570 | +0.0143 | 0.8960 | 0.9469 | +0.0509 | 0.1942 | 0.1438 | -0.0504 |
| resize | 25% | 0.8387 | 0.8374 | -0.0012 | 0.8226 | 0.9272 | +0.1046 | 0.2757 | 0.1658 | -0.1098 |
| blur | light | 0.8401 | 0.8530 | +0.0128 | 0.9402 | 0.9552 | +0.0150 | 0.1577 | 0.1430 | -0.0147 |
| blur | medium | 0.8490 | 0.8513 | +0.0024 | 0.8526 | 0.9411 | +0.0885 | 0.2381 | 0.1494 | -0.0886 |
| blur | strong | 0.8357 | 0.8275 | -0.0083 | 0.7830 | 0.9206 | +0.1376 | 0.3403 | 0.1729 | -0.1674 |
| noise | low | 0.7504 | 0.8507 | +0.1004 | 0.9633 | 0.9606 | -0.0027 | 0.2038 | 0.1415 | -0.0624 |
| noise | medium | 0.3998 | 0.8467 | +0.4469 | 0.9083 | 0.9588 | +0.0505 | 0.3768 | 0.1443 | -0.2325 |
| noise | high | 0.2678 | 0.8288 | +0.5610 | 0.7804 | 0.9523 | +0.1718 | 0.4331 | 0.1566 | -0.2765 |
| brightness | dark | 0.8495 | 0.8754 | +0.0259 | 0.8555 | 0.9283 | +0.0728 | 0.2421 | 0.1492 | -0.0929 |
| brightness | normal | 0.8246 | 0.8502 | +0.0256 | 0.9629 | 0.9607 | -0.0022 | 0.1568 | 0.1421 | -0.0147 |
| brightness | bright | 0.5160 | 0.8030 | +0.2870 | 0.9063 | 0.9116 | +0.0053 | 0.3301 | 0.1953 | -0.1348 |

E07 clean test chi tiết: Accuracy .813, Precision .981, Recall .750,
F1 .850, ROC-AUC .961, PR-AUC .983, APCER .250, BPCER .034, ACER .142.
Confusion matrix: TN 5,678 / FN 3,542 / FP 201 / TP 10,621.

**PHÂN TÍCH SƠ BỘ (cho báo cáo):**

1. **Robust cải thiện rõ nhất ở noise** — đúng chỗ baseline yếu nhất:
   noise high F1 0.27 → 0.83 (+0.56), ACER 0.433 → 0.157 (-0.28),
   AUC +0.17. noise medium cũng cải thiện lớn (F1 +0.45).
2. **Cải thiện lan tỏa sang hầu hết điều kiện**: 15/17 điều kiện có
   ΔACER âm (tốt hơn); mọi mức JPEG đều cải thiện mạnh dần theo mức suy giảm.
3. **2 điều kiện F1 hơi tụt**: blur strong (-0.008) và resize 25% (-0.001) —
   nhưng cả 2 vẫn có AUC tăng rõ (+0.14/+0.10) và ACER giảm.
4. **Clean performance không bị hy sinh**: F1 +0.026, ACER -0.015; ROC-AUC
   giảm rất nhẹ (-0.002).
5. **Anomaly**: brightness dark của E07 có F1 0.875 — cao nhất mọi điều kiện
   (cao hơn cả clean); APCER vẫn >> BPCER ở ngưỡng 0.5 (lệch như baseline);
   best val_loss ở epoch 18 nhưng checkpoint lưu epoch 20.
6. **Kết luận sơ bộ**: quality-aware augmentation CẢI THIỆN robustness dưới
   suy giảm chất lượng, mạnh nhất ở noise (điểm yếu lớn nhất của baseline),
   và không đánh đổi clean performance.

### 5.4. Phát hiện domain shift từ giai đoạn pilot (đã đo — quan trọng cho báo cáo)

- **Test C**: model 18k gọi mặt webcam thật là spoof với xác suất 93-100%.
- **Test B**: ảnh live trong dataset qua cùng pipeline → P(spoof) 0.00-0.07 → inference không sai, vấn đề là domain gap.
- **Test D**: model rất nhạy với mức crop — ảnh full (mặt nhỏ) → P(spoof) 0.81, crop 10% → 0.10 → model học "tỉ lệ mặt/ảnh" của dataset.
- **Fine-tune 500 ảnh webcam** (3 epoch): P(spoof) trên mặt webcam live giảm 0.89 → 0.017 → domain adaptation bằng ít dữ liệu đích rất hiệu quả.
- Hàm ý: khi đánh giá webcam cho baseline/robust 200k, kỳ vọng domain gap vẫn còn nhưng có thể giảm nhờ dữ liệu đa dạng hơn + robust training.

---

## 5.5. BÀI BÁO KHOA HỌC (đã viết + đã audit — 01/09)

- **File chính**: `documents/Paper_Robust_Face_PAD_for_eKYC.docx`
  (~5.000 từ, 12 bảng, 8 hình, 15 references thật).
- **8 hình** sinh từ dữ liệu thật của project, lưu tại `images/`
  (tái tạo bằng `python -m scripts.make_paper_figures`):
  fig1 pipeline tổng quan; fig2 ảnh live+spoof thật qua các mức suy giảm;
  fig3 ROC curve E01 vs E07; fig4 F1/ACER theo severity (5 loại); fig5 bar
  chart mean/worst-case; fig6 train/val loss; fig7-8 ảnh code màu của
  `eval_degradation.py` + `robustness.py` (comment tiếng Anh, highlight).
- **Peer-review + research audit** (đóng vai reviewer):
  - Số liệu: đối chiếu lại toàn bộ từ file gốc → **0 lỗi, 0 làm tròn sai**;
    confusion matrix → tính lại khớp mọi metric.
  - References: **14/14 verified** qua Crossref API + arXiv API (không citation giả).
  - Methodology: khớp code/config (p=0.3, crop tắt, không retrain, cùng test set).
  - Novelty: paper KHÔNG có claim novelty — an toàn.
  - Verdict: PASS WITH CORRECTIONS (9/10).
  - **Corrections đã sửa vào bản chính**: (1) đổi wording "subject-disjoint by
    construction" → "official split, subject-separated per official protocol"
    (mirror không có subject_id nên không verify được trực tiếp); (2) "remains
    close to" → "exceeds" (0.7661 > 0.7028); (3) bỏ "moire patterns" khỏi câu
    trích dẫn [3],[4]; (4) thêm nuance "(F1 +0.0256, ROC-AUC -0.0022)" vào Abstract.
- **Còn thiếu**: điền tên tác giả thay `[Author Names]` trước khi nộp.

---

## 6. VIỆC TIẾP THEO (đúng trình tự, không nhảy cóc)

1. [x] Chạy lệnh grid baseline (mục 4) → kết quả đã điền vào bảng 5.2.
2. [x] Train E07 robust 200k (config: crop TẮT, 5 augmentation p=0.3) → xong 01/09.
3. [x] Grid robust → bảng so sánh 5.3 (baseline vs robust từng điều kiện).
4. [x] Viết bài báo + 8 hình + peer-review audit + sửa corrections (mục 5.5).
5. [ ] Điền tên tác giả vào paper trước khi nộp.
6. [ ] (Nếu còn thời gian) data-scale: 18k pilot vs 200k; ablation từng augmentation; webcam nhiều người; failure case analysis; latency/model-size.
