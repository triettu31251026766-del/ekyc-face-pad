# HƯỚNG DẪN TRAINING MODEL eKYC FACE PAD

File này hướng dẫn chạy **TOÀN BỘ** quy trình huấn luyện + đánh giá model PAD từ đầu đến cuối, theo đúng thứ tự thí nghiệm mục 32 của tài liệu kỹ thuật (`documents/TECHNICAL_DOCUMENTATION_eKYC_FACE_PAD.md`).

> **CẬP NHẬT 29/08/2026**: Giai đoạn pilot 18k đã HOÀN TẤT — kết quả nằm trong
> `results/pilot_18k/` (E01 F1 .931, E07 F1 .929, E20 fine-tune webcam). Giai
> đoạn chính (FULL CelebA-Spoof) đang làm — xem `documents/PROGRESS.md`.
> Các lệnh dưới đây dùng chung cho cả 2 giai đoạn; checkpoint trong ví dụ là
> của giai đoạn chính (`results/checkpoints/...`), checkpoint pilot nằm ở
> `results/pilot_18k/checkpoints/...`.

**SEED TOÀN DỰ ÁN: 123**

- Mỗi thí nghiệm bắt buộc dùng **CÙNG seed 123** và **CÙNG splits** để so sánh công bằng (mục 24, 41 tài liệu).
- Tên file kết quả sẽ có dạng `E01_baseline_seed123...`

---

## 1. YÊU CẦU MÔI TRƯỜNG

- Windows + GPU NVIDIA (driver hỗ trợ CUDA >= 12.6)
- Python 3.11+ (dùng `python` GLOBAL, **KHÔNG activate `.venv`** vì `.venv` đang cài torch bản CPU, không dùng được GPU)

### Kiểm tra GPU nhanh

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Phải ra:

```text
2.13.0+cu126 True
```

---

## 2. CÀI THƯ VIỆN (chỉ cần nếu chưa cài)

```bash
python -m pip install -r requirements.txt
```

Nếu torch là bản CPU (`cuda.is_available() = False`) thì cài bản CUDA:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
```

### Giải thích

- `requirements.txt`: cài các thư viện chung (torch, torchvision, numpy, pandas, scikit-learn, pillow, pyyaml, matplotlib, tqdm, pytest, ...)
- Lệnh thứ 2: thay torch/torchvision bản CPU bằng bản CUDA 12.6 (~2.5GB) để train trên GPU RTX.

---

## 3. CHUẨN BỊ DATASET

Đã có sẵn (18.000 ảnh, crop mặt):

```text
data/raw/celeba_spoof/SpoofingData/   <- ảnh
data/raw/celeba_spoof/train_list.txt  <- danh sách ảnh + nhãn
                                      (0 = bona_fide, 1 = spoof)
```

Nếu máy khác chưa có dataset thì tải về:

```bash
python scripts/download_dataset.py
```

### Giải thích

Script tải subset CelebA-Spoof (18k ảnh) từ HuggingFace, crop khuôn mặt theo bounding box, sinh `train_list.txt` đúng format mà `src/data.py` đọc được.

**KHÔNG cần tải full 74GB từ Google Drive.**

---

## 4. KIỂM TRA TRƯỚC KHI CHẠY (tùy chọn)

```bash
python -m pytest tests/ -q
```

### Giải thích

Chạy 191 unit test + smoke test end-to-end để chắc chắn code không hỏng trước khi train thật.

---

## 5. BƯỚC 1 - HUẤN LUYỆN BASELINE (E01)

```cmd
set PYTHONUTF8=1
python -m experiments.train_baseline --config configs/clean.yaml
```

PowerShell thay dòng đầu bằng:

```powershell
$env:PYTHONUTF8 = 1
```

### Giải thích từng phần

- `set PYTHONUTF8=1`: cho CMD hiển thị tiếng Việt không lỗi encoding.
- `python -m experiments.train_baseline`: **BẮT BUỘC** chạy dạng module (có `-m`), vì script import `from src...` và `from experiments...`; nếu chạy `python experiments/xxx.py` sẽ lỗi `ModuleNotFoundError`.
- `--config configs/clean.yaml`: cấu hình MobileNetV2, ảnh 224x224, 20 epoch, batch 64, lr 0.0001, seed 123, device auto (tự chọn GPU nếu có, không thì CPU).

### Script làm gì (mục 31 tài liệu)

```text
nạp config -> set seed -> nạp dataset -> tạo split subject_disjoint
-> build MobileNetV2 -> train 20 epoch -> lưu checkpoint mỗi epoch
-> đánh giá trên test set SẠCH -> lưu kết quả JSON + CSV.
```

### Đầu ra khi xong

```text
results/raw/E01_baseline_seed123.json
results/raw/E01_baseline_seed123.csv
results/raw/E01_baseline_seed123_predictions.csv
results/checkpoints/E01_baseline_seed123.pt
data/splits/celeba_spoof_seed123_subject_disjoint.json
```

`results/checkpoints/E01_baseline_seed123.pt` cần cho bước 6.

`data/splits/celeba_spoof_seed123_subject_disjoint.json` được tái sử dụng.

### Thời gian

~15-40 phút trên GPU (RTX 3050).

### Theo dõi tiến độ

Mở tab CMD/PowerShell thứ 2:

```powershell
powershell -Command "Get-Content results/raw/E01_baseline_seed123.log -Wait -Tail 15"
```

### Giải thích

Xem log trực tiếp. Trong terminal chạy train có thanh tiến độ từng batch (tqdm) + mỗi epoch in 1 dòng:

```text
epoch 1/20: train_loss=0.5123 | val_loss=0.4512
```

Dùng `Ctrl+C` để dừng giữa chừng nếu cần.

---

## 6. BƯỚC 2 - ĐÁNH GIÁ SUY GIẢM CHẤT LƯỢNG (E02-E06)

**KHÔNG train lại - chỉ dùng checkpoint của E01, mục 23 tài liệu.**

```bash
python -m experiments.eval_degradation --config configs/degradation_jpeg.yaml   --checkpoint results/checkpoints/E01_baseline_seed123.pt
python -m experiments.eval_degradation --config configs/degradation_resize.yaml --checkpoint results/checkpoints/E01_baseline_seed123.pt
python -m experiments.eval_degradation --config configs/degradation_blur.yaml   --checkpoint results/checkpoints/E01_baseline_seed123.pt
python -m experiments.eval_degradation --config configs/degradation_noise.yaml  --checkpoint results/checkpoints/E01_baseline_seed123.pt
```

### Giải thích

Mỗi lệnh áp MỘT kiểu suy giảm chất lượng nhất định (JPEG quality 50 / resize 0.5 / blur 7+2.0 / noise std 0.03) lên test set rồi đánh giá model baseline - đo xem model "yếu" thế nào khi đầu vào kém chất lượng.

Kết quả lưu `results/raw/E0x_*_seed123.*`

**Không được train lại model trong bước này.**

---

## 7. BƯỚC 3 - HUẤN LUYỆN ROBUST (E07)

```bash
python -m experiments.train_robust --config configs/clean.yaml --robustness configs/robustness.yaml
```

### Giải thích

Train model thứ 2, trong lúc train mỗi ảnh được tăng cường chất lượng **NGẪU NHIÊN** (JPEG/resize/blur/noise/brightness theo các khoảng tham số trong `configs/robustness.yaml`).

Mọi tham số khác (model, lr, epochs, batch size, split, seed) **GIỮ NGUYÊN** để so sánh công bằng với baseline (mục 24).

Đầu ra: `E07_robust_seed123.*` và checkpoint `E07_robust_seed123.pt`.

---

## 8. BƯỚC 4 - SO SÁNH BASELINE vs ROBUST + SINH HÌNH

```bash
python -m experiments.compare_models
```

### Giải thích

Đọc các file kết quả đã lưu trong `results/raw`, dựng bảng so sánh (`results/tables/`) và vẽ hình (`results/figures/`) từ dữ liệu **THẬT** - không train lại, không tự nhập số liệu (mục 40).

---

## 9. BƯỚC 5 - ABLATION (E09+)

```bash
python -m experiments.ablation --config configs/clean.yaml --robustness configs/robustness.yaml
```

### Giải thích

Chạy/đọc từng biến thể tăng cường riêng lẻ (chỉ JPEG, chỉ blur, ...) để biết biến nào giúp model bền vững nhất, lưu bảng ablation vào `results/tables/`.

---

## 10. CHẠY TẤT CẢ MỘT LƯỢT (TÙY CHỌN)

```bash
python -m experiments.run_all --config configs/clean.yaml --robustness configs/robustness.yaml
```

### Giải thích

Tự động chạy cả chuỗi theo đúng thứ tự mục 32:

```text
E01 -> đánh giá suy giảm -> E07 robust -> so sánh -> ablation.
```

**Chỉ nên chạy khi E01 đã thành công.**

---

## 11. KẾT QUẢ NẰM Ở ĐÂU

```text
results/raw/          JSON + CSV metric + predictions (mục 29-30, 39)
results/tables/       bảng so sánh, bảng ablation
results/figures/      hình vẽ dùng cho báo cáo
results/checkpoints/  trọng số model (.pt)
data/splits/          splits đã lưu (tái lập được)
```

---

## 12. XỬ LÝ SỰ CỐ THƯỜNG GẶP

### "ModuleNotFoundError: No module named 'experiments'"

-> Phải chạy:

```bash
python -m experiments.xxx
```

Không được:

```bash
python experiments/xxx.py
```

### GPU hết bộ nhớ (OOM / CUDA out of memory)

-> Giảm `training.batch_size` xuống 32 trong `configs/clean.yaml`.

### Lỗi tiếng Việt loạn trong CMD

-> Chạy:

```cmd
set PYTHONUTF8=1
```

PowerShell:

```powershell
$env:PYTHONUTF8 = 1
```

### DataLoader chậm / lỗi spawn worker

-> Giảm hoặc đặt `training.num_workers = 0` trong config.

### torch.cuda.is_available() = False

-> Cài lại torch bản CUDA (xem mục 2).

### Không in gì trong lúc train

-> Bình thường nếu bạn đang dùng bản code cũ (chưa có tqdm); code hiện tại đã có thanh tiến độ + log mỗi epoch.

---

## GHI CHÚ QUAN TRỌNG (mục 41 + Rule 6 tài liệu)

Khi so sánh baseline vs robust **PHẢI GIỮ NGUYÊN**:

- dataset
- splits
- model
- learning rate
- epochs
- batch size
- threshold
- seed

**Biến duy nhất được thay đổi là CHIẾN LƯỢC HUẤN LUYỆN.**

**KHÔNG được tự sửa số liệu metric trong báo cáo (Rule 5).**

---

## 13. THỬ MODEL BẰNG WEBCAM (XEM THỬ TRỰC TIẾP)

Chỉ cần **train xong E01 baseline** là chạy được — không cần đợi robust/compare.

```cmd
python -m scripts.camera_demo
```

Tùy chọn khác:

```cmd
python -m scripts.camera_demo --checkpoint results/pilot_18k/checkpoints/E20_webcam_finetune_seed123.pt   :: dùng model fine-tune webcam (tốt nhất cho demo)
python -m scripts.camera_demo --device cpu                                              :: không dùng GPU
python -m scripts.camera_demo --camera 1                                                :: chọn camera khác
```

### Giải thích

- Nạp checkpoint, mở webcam (1280x720, cửa sổ hiển thị lớn 960x720).
- Tự phát hiện khuôn mặt (Haar cascade, file `scripts/haarcascade_frontalface_default.xml`) và **khoanh khung từng khuôn mặt**: **XANH = bona_fide**, **ĐỎ = spoof**, kèm xác suất ngay trên khung.
- Mỗi khuôn mặt được cắt ra, đưa qua đúng eval transform (resize → tensor → normalize) rồi dự đoán riêng.
- Nhấn `q` hoặc `ESC` để thoát.
- KHÔNG chạy camera cùng lúc với huấn luyện (chia sẻ chung VRAM GPU).
