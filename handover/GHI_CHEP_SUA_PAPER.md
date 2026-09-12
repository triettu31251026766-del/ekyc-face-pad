# GHI CHÉP — SỬA PAPER THEO 12 ĐIỂM CỦA THẦY + TEMPLATE LNCS

## Template
- `documents/Template-word.docm` = **Springer LNCS** (Lecture Notes in Computer Science).
- Format: 10pt TNR body, title 14pt bold centered, authors + affiliations (footnote).
- "Abstract." (bold lead-in, 150-250 từ), "Keywords." (bold lead-in).
- Heading cấp 1: "1 Introduction" 12pt bold; cấp 2: "2.1 ..." 10pt bold;
  cấp 3: run-in bold; cấp 4: run-in italic.
- Table caption ĐẶT TRÊN bảng ("Table 1. ..."); Figure caption ĐẶT DƯỚI ("Fig. 1. ...").
- References: kiểu LNCS "Author, F.: Title. ..." + citation [n].

## DỮ LIỆU HIỆN TẠI (Bộ B — best-epoch epoch 18, đã committed)
Nguồn: handover/paper/FINAL_EXPERIMENT_DATA.md + results/raw/*.json + results/tables/*.csv
- Split: train 142,250 / val 20,322 / test 40,643 (tổng 203,215), seed 123.
- Clean (best-epoch): E01 F1 .9868 AUC .9978 ACER .0211; E07 F1 .9831 AUC .9969 ACER .0256.
- Aggregates (16 điều kiện): mean F1 .9226→.9743; mean ACER .1153→.0390;
  worst F1 .7985→.9446; worst ACER .3310→.1012.
- Checkpoint: epoch 18 (val E01 .0586 / E07 .0681). Model từ SCRATCH (build_model pretrained=False).
- Hardware: RTX 5060 Laptop 8GB, torch 2.14.0+cu130, Python 3.14.7, num_workers 0.

## PHÁT HIỆN QUAN TRỌNG KHI AUDIT LẠI
1. **subject_id = chỉ số ảnh** (filename {idx:06d}.jpg -> stem). Mỗi ảnh là 1 "subject" riêng.
   => "subject-disjoint" ĐÚNG về kỹ thuật (0 overlap) nhưng KHÔNG phải phân tách theo NGƯỜI THẬT.
   Mirror không cung cấp subject id thật. => Phải hạ claim: split là image-level (random),
   đo generalization trên ảnh chưa thấy, KHÔNG phải người chưa thấy.
2. Model MobileNetV2 train từ SCRATCH (không pretrained ImageNet) — phải ghi rõ.
3. Single seed 123 -> phải hạ claim + đưa single-seed vào Limitations.
4. Threshold 0.5 cố định (fixed-threshold protocol) -> ghi rõ, không hàm ý tối ưu.
5. In-range vs out-of-range degradation (severity extrapolation):
   - jpeg train [50,90] vs test 30 => 30 out-of-range
   - resize train [0.5,1.0] vs test 0.25 => 0.25 out-of-range
   - blur train sigma [0.5,2.0] vs test 3.0 => out-of-range
   - noise train [0.005,0.03] vs test 0.005/0.015/0.03 => all in-range
   - brightness train [0.7,1.3] vs test 0.6/1.4 => 0.6 và 1.4 out-of-range
   => Điểm MẠNH: robust vẫn cải thiện ở mức ngoài training range (extrapolation).
6. Clean performance GIẢM nhẹ (F1 .9868→.9831, ACER .0211→.0256) => viết là "small clean-performance
   cost, substantial robustness gain" (trade-off), KHÔNG "essentially preserved".

## KẾ HOẠCH SỬA (12 điểm thầy -> hành động)
1. Multi-seed: KHÔNG chạy được (không có GPU/data trên máy này). => Hạ claim thành single-seed
   controlled experiment + ghi vào Limitations. KHÔNG bịa multi-seed.
2. Threshold: giữ 0.5, ghi rõ "fixed-threshold protocol; 0.5 không phải operating point tối ưu".
3. In/out-of-range: thêm bảng/subsection phân biệt, nêu strength (extrapolation).
4. Clean wording: "small clean cost + substantial robustness gain".
5. Uncertainty: TÍNH paired bootstrap 95% CI cho ΔF1/ΔACER/ΔAUC (clean + 16 điều kiện)
   từ *_predictions.csv (có sẵn, committed). Ghi rõ bootstrap ≠ multi-seed.
6. Subject-disjoint: hạ claim (image-level split; mirror không có subject id thật).
7. Scope: "synthetic corruption robustness under controlled degradations relevant to eKYC".
8. Lightweight = kiến trúc (param count/size), KHÔNG claim latency.
9. Related Work: thêm refs thật về robust PAD / domain generalization / cross-dataset / efficient FAS
   (verify web). Ứng viên: Sun CVPR2023 "Rethinking DG for FAS", Jia CVPR2019, Shao CVPR2020
   "Single-side DG", CDCN CVPR2020, PatchNet CVPR2022, "Assessing Efficient FAS" CVPR2024.
10. Table VI (16 điều kiện): giữ cột F1, ROC-AUC, APCER, BPCER, ACER (bỏ delta + PR-AUC);
    full table -> supplementary/repo.
11. Bỏ Table X (loss 20 epoch), giữ Fig 6; text: "min val loss at epoch 18".
12. Model init: "trained from scratch".

## REFS HIỆN CÓ (15) — giữ nguyên, chuyển sang LNCS style
[1] Ramachandra & Busch; [2] Patel; [3] Määttä; [4] Chingovska; [5] CelebA-Spoof;
[6] OULU-NPU; [7] Liu; [8] George & Marcel; [9] MobileNetV2; [10] MobileNets;
[11] Hendrycks & Dietterich; [12] Shorten; [13] AutoAugment; [14] Challenge; [15] ImageNet.

## CON SỐ CẦN TÍNH THÊM (để điền paper)
- Subject counts + intersections (đã xác nhận subject_id = image index).
- Bootstrap 95% CI cho ΔF1/ΔACER/ΔAUC (clean + 16 điều kiện).
