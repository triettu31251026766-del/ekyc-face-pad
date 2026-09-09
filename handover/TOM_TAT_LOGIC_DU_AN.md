# TÓM TẮT LOGIC TỔNG THỂ DỰ ÁN — eKYC Face PAD

> File này giải thích LOGIC của toàn bộ dự án (không phải code đầy đủ) để
> mọi người đọc là hiểu được dự án chạy thế nào. Đường dẫn trong ngoặc là
> file thật trong repo — mở file đó để xem code chi tiết.

---

## 1. Mục tiêu

Trả lời 2 câu hỏi trên CelebA-Spoof (~200k ảnh):
1. Model PAD nhẹ (MobileNetV2) suy giảm thế nào khi chất lượng ảnh đầu vào giảm
   (JPEG / resize / blur / noise / brightness)?
2. Huấn luyện bằng quality-aware augmentation (E07 "Robust") có cải thiện
   độ bền vững đó không, và có hy sinh hiệu năng clean không?

---

## 2. Luồng dữ liệu tổng thể

```
CelebA-Spoof (~526k ảnh trên HuggingFace)
        │ tải subset: 45 train shard + 22 test shard + 4 valid shard
        │ (scripts/download_celeba_full.py)
        ▼
data/raw/celeba_spoof_full/SpoofingData/  (ảnh crop mặt bbox + 10% lề)
        │ + split chính thức của dataset (subject tách biệt theo protocol gốc)
        │ lưu: data/splits/celeba_spoof_full_seed123_subject_disjoint.json
        ▼
HUẤN LUYỆN (src/train.py + experiments/_common.py)
        ├── E01 Baseline : ảnh sạch + RandomHorizontalFlip
        └── E07 Robust   : + 5 quality augmentation, mỗi phép p=0.3
                           (src/robustness.py gọi src/degradation.py)
        ▼
MobileNetV2 (src/model.py) — 2.225.153 tham số, 1 logit đầu ra
        ▼
ĐÁNH GIÁ (src/evaluate.py + experiments/eval_degradation_grid.py)
        ├── clean test 20.042 ảnh, threshold 0.5
        └── 16 điều kiện suy giảm TẤT ĐỊNH, KHÔNG retrain
        ▼
Metrics: F1, ROC-AUC, PR-AUC, APCER, BPCER, ACER (src/metrics.py)
        ▼
Bảng + biểu đồ → results/tables/, results/figures/ → bài báo
```

---

## 3. Logic từng khối (mở file trong repo để xem code)

| Khối | File | Logic chính |
|---|---|---|
| Cấu hình | `src/config.py` + `configs/*.yaml` | Đọc + validate YAML (3 loại: training / degradation / robustness). Mọi biến thí nghiệm nằm ở config, KHÔNG hard-code |
| Dữ liệu | `src/data.py` | Đọc `train_list.txt` → chuẩn hóa Sample(path, label, subject_id); nhãn 0=live, 1=spoof; tạo/lưu/nạp splits |
| Dataset | `src/dataset.py` | PyTorch Dataset trả dict {image, label, path}; áp transform |
| Tiền xử lý | `src/transforms.py` | resize 224 → ToTensor → normalize ImageNet. Train có flip; eval TẤT ĐỊNH (không augmentation) |
| Suy giảm | `src/degradation.py` | 5 hàm: jpeg_compression, resize_degradation, gaussian_blur, gaussian_noise, brightness_adjustment — tham số tường minh, tất định |
| Robustness | `src/robustness.py` | Với mỗi ảnh train: duyệt 5 phép theo thứ tự cố định; mỗi phép áp ĐỘC LẬP với p=0.3 (config), severity ~ uniform(range). Nhiều phép CÓ THỂ chồng nhau; P(ảnh sạch) = 0.7^5 ≈ 17%. Crop đang TẮT |
| Model | `src/model.py` | MobileNetV2, classifier → 1 logit; loss = BCEWithLogitsLoss (sigmoid nằm trong loss) |
| Train | `src/train.py` | Vòng lặp epoch tổng quát + callback lưu checkpoint mỗi epoch (lưu ĐÈ cùng file → chỉ giữ epoch cuối) |
| Evaluate | `src/evaluate.py` | Duyệt test loader → xác suất spoof → metrics tại threshold cố định → lưu predictions CSV |
| Metrics | `src/metrics.py` | Quy ước positive = spoof. APCER = % spoof bị gọi là thật; BPCER = % thật bị gọi là spoof; ACER = trung bình 2 cái |
| Điều phối | `experiments/_common.py` | Pipeline chung cho E01/E07: dataset → splits → transforms → train → clean eval → lưu JSON/CSV. E01/E07 KHÁC NHAU DUY NHẤT ở train transform |
| Lưới suy giảm | `experiments/eval_degradation_grid.py` | SEVERITY_GRID 16 mức (jpeg 90/70/50/30; resize 75/50/25%; blur light/medium/strong; noise low/medium/high; brightness dark/normal/bright); chạy `eval_degradation.run` từng mức; gộp bảng + vẽ hình |
| Fine-tune webcam | `experiments/finetune_webcam.py` | (pilot) trộn ảnh webcam + mẫu dataset, 3 epoch lr 5e-5 |

---

## 4. Logic so sánh công bằng E01 vs E07

Mọi thứ GIỐNG NHAU: cùng split file, seed 123, MobileNetV2 224, 20 epochs,
batch 64, Adam lr 1e-4 + wd 1e-5, BCEWithLogitsLoss, threshold 0.5, cùng test
set 20.042 ảnh, cùng lưới suy giảm, cùng code tính metric.

KHÁC NHAU DUY NHẤT: E01 train ảnh sạch; E07 train ảnh có quality augmentation
(5 phép × p=0.3, crop tắt).

→ Chênh lệch kết quả chỉ do chiến lược huấn luyện (đúng yêu cầu mục 24/41 tài liệu).

---

## 5. Các quyết định thiết kế quan trọng (vì sao làm vậy)

1. **Seed 123 toàn dự án** — tái lập được mọi thí nghiệm.
2. **Threshold 0.5 cố định** — so sánh công bằng mọi model; KHÔNG chọn ngưỡng
   trên test set (tránh leak). Hệ quả đã biết: APCER ≫ BPCER ở model 200k
   (được ghi rõ trong mục Hạn chế của paper).
3. **p=0.3 mỗi augmentation** — giữ ~17% ảnh sạch mỗi epoch, trung bình 1.5
   phép/ảnh, tránh stacking quá nặng.
4. **Crop tắt trong thí nghiệm chính** — bộ augmentation khớp đúng 5 loại
   suy giảm của lưới đánh giá (crop chỉ dùng cho demo camera ở pilot).
5. **Dùng split chính thức của mirror** — mirror không có subject_id nên
   không tự chia lại được; split gốc của CelebA-Spoof được thiết kế để tách
   subject giữa các tập.
6. **num_workers: train 4 / eval 0** — train nhanh; eval tất định (thứ tự
   mẫu cố định).
7. **Không class weighting** dù train lệch spoof:live ≈ 2:1 — giữ protocol
   đơn giản; hạn chế này đã ghi trong paper.

---

## 6. Cách mở code thật để đối chiếu

```text
src/robustness.py            dòng ~93-133 : hàm apply_training_quality_augmentation
experiments/eval_degradation.py  dòng ~41-118 : hàm run (stress-test tất định)
experiments/_common.py       train_and_evaluate : pipeline chung E01/E07
experiments/eval_degradation_grid.py : SEVERITY_GRID (16 mức) + main loop
configs/robustness.yaml      : 5 augmentation, probability 0.3, crop enabled false
configs/full_clean.yaml      : toàn bộ hyperparameters thí nghiệm chính
```

(2 đoạn code trên cũng đã được render thành ảnh màu: images/fig7_*.png,
images/fig8_*.png trong folder này và trong bài báo.)
