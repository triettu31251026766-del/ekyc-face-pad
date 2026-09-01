# AGENTS.md — Hướng dẫn bắt buộc cho AI coding agent

File này để MỌI AI agent (DeepSeek, Claude, GPT, ...) đọc ĐẦU TIÊN khi làm việc
trên repo này. Làm sai quy ước ở đây là lỗi nghiêm trọng.

## 1. Đọc tài liệu theo đúng thứ tự (BẮT BUỘC)

1. `documents/TECHNICAL_DOCUMENTATION_eKYC_FACE_PAD.md` — đặc tả kỹ thuật (mục 43-44 có rule dành riêng cho AI)
2. `documents/Student_Guide_eKYC_Face_PAD_Project.docx` — guide tổng quan của đồ án (mở bằng python-docx nếu cần)
3. `documents/PROGRESS.md` — **trạng thái hiện tại: đã làm gì, đang làm gì, làm tiếp gì**. Đây là file sống, phải cập nhật sau mỗi việc làm.
4. `documents/HUONG_DAN_TRAINING_MODEL_eKYC_FACE_PAD.md` — lệnh chạy cụ thể từng thí nghiệm.

## 2. Quy ước BẮT BUỘC của repo

- **Seed toàn dự án: `123`** — không đổi seed khi chưa được yêu cầu (Rule 6).
- **Nhãn: `0 = bona_fide`, `1 = spoof`** (mục 6 tài liệu); metric quy ước **positive = spoof**.
- **Mọi file code mới PHẢI có docstring tiếng Việt ở đầu file** mô tả: file là gì, dùng để làm gì, cách dùng.
- **Không bịa số liệu** (Rule 5): chưa chạy thí nghiệm thì ghi "chưa đo".
- **Chạy script bằng `python -m experiments.X`** (KHÔNG `python experiments/X.py` — sẽ lỗi import).
- Sau khi sửa code: **chạy `python -m pytest tests/ -q`** và phải pass (Rule 4).
- Chỉ sửa module được giao, không viết lại file không liên quan (Rule 3).
- Không tự ý đổi: seed, split, model, lr, epochs, batch size, threshold, degradation severity (Rule 6).
- Compare baseline vs robust PHẢI giữ nguyên mọi thứ trừ chiến lược huấn luyện (mục 24, 41).

## 3. Trạng thái hiện tại (tóm tắt — chi tiết xem PROGRESS.md)

- Code base: HOÀN CHỈNH (14 module src + experiments + 193 test pass).
- **Giai đoạn pilot 18k: XONG** — kết quả nằm trong `results/pilot_18k/`.
- **Giai đoạn chính (full CelebA-Spoof): ĐANG LÀM** — xem phần P0 trong PROGRESS.md.
- Kết quả pilot KHÔNG được xóa (dùng làm baseline so sánh + câu chuyện domain shift).

## 4. Trạng thái + việc đang chờ (ưu tiên cao nhất)

- **Giai đoạn chính 200k: CORE ĐÃ XONG** (xem `documents/NHAT_KY_THI_NGHIEM.md`):
  - E01 baseline (F1 .825, AUC .963) + lưới suy giảm E02-E06 ✅
  - E07 robust (F1 .850, AUC .961) + lưới suy giảm E08-E12 ✅
  - So sánh: robust cải thiện mạnh nhất ở noise, clean không bị hy sinh ✅
- **Việc tiếp theo (P1 — tùy chọn, làm nếu còn thời gian trước 10/09)**:
  1. Failure case analysis (FP/FN từ predictions CSV)
  2. Data-scale: 18k pilot vs 200k
  3. Ablation từng augmentation (`experiments/ablation.py`)
  4. Webcam đánh giá nhiều người (baseline vs robust 200k)
  5. Latency/model-size
- **BÀI BÁO ĐÃ XONG**: `documents/Paper_Robust_Face_PAD_for_eKYC.docx` (~5.000 từ,
  12 bảng, 8 hình trong `images/`, 15 refs thật, đã qua peer-review audit và sửa
  corrections). Còn thiếu duy nhất: điền tên tác giả.
- **Bàn giao**: thư mục `handover/` chứa documents + paper + images + code cốt lõi
  để gửi cho sinh viên validate.

## 5. Lưu ý máy khác nhau (sinh viên chạy máy riêng)

- Đường dẫn dataset tuyệt đối KHÔNG được hard-code trong config — dùng đường dẫn tương đối `data/raw/...`.
- `training.num_workers`: tăng/giảm tùy máy (config, không sửa code).
- `training.batch_size`: giảm nếu VRAM nhỏ (OOM).
- GPU: `device: auto` trong config tự chọn cuda/cpu.
- Đừng đưa `data/`, `results/`, `.venv/` vào git (đã có .gitignore).
