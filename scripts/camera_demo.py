"""scripts/camera_demo.py — thử nghiệm model PAD trực tiếp bằng webcam.

Tệp này dùng để:
- Nạp checkpoint đã huấn luyện (mặc định: E01 baseline seed 123).
- Mở webcam, phát hiện khuôn mặt bằng YuNet (cv2.FaceDetectorYN, model
  face_detection_yunet_2023mar.onnx từ OpenCV Zoo); với mỗi khuôn mặt:
  cắt vùng mặt -> tiền xử lý đúng eval transform -> model -> xác suất spoof.
- Khoanh khung quanh từng khuôn mặt: XANH = bona_fide, ĐỎ = spoof,
  kèm nhãn + phần trăm (0-100%) ngay trên khung. Mỗi khuôn mặt được đánh giá
  ĐỘC LẬP (không còn gộp/làm mịn giữa các mặt hay giữa các khung hình).
- GỠ LỖI DOMAIN SHIFT: in log xác suất ra console liên tục (mỗi --log-every
  khung hình) và lưu ảnh face_crop vào --save-dir để soi xem model thực sự
  nhận ảnh gì. Nhấn 's' để chụp lưu ngay khung hình hiện tại.

Lưu ý:
- opencv-python 5.x đã BỎ Haar cascade nên dùng YuNet; file model nằm ở
  scripts/face_detection_yunet_2023mar.onnx (tải từ repo chính thức
  https://github.com/opencv/opencv_zoo).
- KHÔNG chạy đồng thời với huấn luyện (chia sẻ chung VRAM của GPU).
- Ngưỡng mặc định là 0.5, khớp ngưỡng chính thức của thí nghiệm (config).
  Nếu muốn cân bằng APCER=BPCER cho riêng webcam, tự đo lại và truyền
  --threshold tương ứng — đừng suy ra ngưỡng đó từ predictions offline
  (E01) rồi hard-code vào đây, vì domain khác nhau (ảnh dataset vs webcam).

Cách dùng (chạy từ thư mục gốc dự án):
    python -m scripts.camera_demo
    python -m scripts.camera_demo --checkpoint results/checkpoints/E07_robust_seed123.pt
    python -m scripts.camera_demo --threshold 0.5 --save-dir results/camera_debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

from experiments._common import load_checkpoint
from src.transforms import build_eval_transform
from src.utils import resolve_device

GREEN = (0, 220, 0)
RED = (0, 0, 230)
WHITE = (255, 255, 255)

WINDOW_NAME = "eKYC Face PAD (q/ESC thoat, s chup anh)"
DISPLAY_WIDTH = 1060
DISPLAY_HEIGHT = 720

YUNET_MODEL = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh: checkpoint, camera, device, threshold, debug."""
    parser = argparse.ArgumentParser(description="Chạy thử model PAD bằng webcam")
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/E01_baseline_seed123.pt",
        help="đường dẫn checkpoint đã huấn luyện",
    )
    parser.add_argument("--camera", default="0",
                        help="chỉ số camera (0, 1, ...) HOẶC URL MJPEG. "
                             "Ví dụ DroidCam: http://192.168.1.5:4747/video")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="ngưỡng xác suất spoof (mặc định 0.5, khớp ngưỡng thí nghiệm)",
    )
    parser.add_argument(
        "--save-dir", default="results/camera_debug",
        help="thư mục lưu ảnh face_crop để gỡ lỗi (mặc định results/camera_debug)",
    )
    parser.add_argument(
        "--log-every", type=int, default=15,
        help="in log xác suất + lưu crop sau mỗi N khung hình (mặc định 15)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.10,
        help="tỉ lệ mở rộng khung mặt quanh bbox YuNet (mặc định 0.10 khớp "
             "training; model rất nhạy với crop nên nên thử 0.0/0.05/0.15/0.25)",
    )
    return parser.parse_args()


def predict_face(model: torch.nn.Module, face_bgr, transform, device: torch.device
                 ) -> float:
    """Dự đoán xác suất spoof cho một vùng ảnh khuôn mặt (BGR) đã cắt.

    Returns:
        Xác suất spoof (0.0 - 1.0).
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        prob = float(torch.sigmoid(logit).item())
    return prob


def expand_face_box(x: int, y: int, w: int, h: int, frame_w: int,
                    frame_h: int, ratio: float = 0.10) -> tuple[int, int, int, int]:
    """Mở rộng khung mặt thêm tỉ lệ ratio (10%) quanh khuôn mặt.

    Khớp với cách crop dataset lúc tải về (bbox + 10% lề trong
    scripts/download_dataset.py) để giảm mismatch giữa training và inference.
    """
    pad_w = int(w * ratio)
    pad_h = int(h * ratio)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame_w, x + w + pad_w)
    y2 = min(frame_h, y + h + pad_h)
    return x1, y1, x2, y2


def resize_display(frame, width: int = DISPLAY_WIDTH, height: int = DISPLAY_HEIGHT):
    """Phóng to khung hình để hiển thị (giữ nguyên tỉ lệ)."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    if scale <= 1.0:
        return frame
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def open_capture(source: str):
    """Mở nguồn camera: chỉ số (DSHOW -> fallback mặc định) hoặc URL MJPEG.

    URL MJPEG dùng cho camera ảo như DroidCam (http://IP:4747/video).
    """
    if str(source).isdigit():
        capture = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture = cv2.VideoCapture(int(source))
        if capture.isOpened():
            # MJPG + DirectShow: tranh hien tuong man hinh xanh le/sai mau
            # (camera YUYV bi doc nham kenh) tren Windows.
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        return capture
    return cv2.VideoCapture(str(source))


def main() -> None:
    """Mở webcam, phát hiện khuôn mặt, dự đoán, vẽ khung + log + lưu crop."""
    args = parse_args()

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint khong ton tai: {checkpoint_path}")

    model, config, _ = load_checkpoint(checkpoint_path, device)
    threshold = args.threshold
    transform = build_eval_transform(config)
    model.eval()

    if not YUNET_MODEL.is_file():
        raise FileNotFoundError(
            f"Khong tim thay model YuNet: {YUNET_MODEL}. Tai ve tu: {YUNET_URL}"
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    capture = open_capture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Khong mo duoc camera: {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # YuNet cần biết chính xác kích thước khung hình đầu vào.
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("Khong doc duoc khung hinh tu camera")
    frame_h, frame_w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (frame_w, frame_h),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print(f"Model: {config['model']['name']} | threshold={threshold} | "
          f"device={device} | camera {frame_w}x{frame_h}")
    print(f"Luu crop vao: {save_dir} (moi {args.log_every} khung; phim 's' chup ngay)")
    print("Webcam dang chay. Nhan 'q' hoac ESC de thoat.")

    frame_idx = 0
    while True:
        frame = cv2.flip(frame, 1)
        frame_idx += 1
        do_debug = frame_idx % args.log_every == 0

        _, faces = detector.detect(frame)
        if faces is None:
            faces = []

        if len(faces) == 0:
            cv2.putText(frame, "Khong tim thay khuon mat", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
            if do_debug:
                print(f"[frame {frame_idx}] khong co mat")

        else:
            log_parts = []
            for i, face in enumerate(faces):
                x, y, w, h = (int(round(v)) for v in face[:4])
                x1, y1, x2, y2 = expand_face_box(x, y, w, h, frame_w, frame_h,
                                                 ratio=args.margin)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                prob = predict_face(model, face_crop, transform, device)
                is_spoof = prob >= threshold
                color = RED if is_spoof else GREEN
                label = f"{'spoof' if is_spoof else 'bona_fide'} {prob * 100:.0f}%"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                text_y = y1 - 12 if y1 - 12 > 20 else y2 + 25
                cv2.putText(frame, label, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if do_debug:
                    tag = "spoof" if is_spoof else "live"
                    crop_file = save_dir / f"frame{frame_idx:05d}_face{i}_{tag}_{int(prob * 100):03d}.jpg"
                    cv2.imwrite(str(crop_file), face_crop)
                    log_parts.append(f"face{i}={prob:.2f}({tag})")

            if do_debug:
                print(f"[frame {frame_idx}] {len(faces)} mat: {', '.join(log_parts)}")
                view_file = save_dir / f"frame{frame_idx:05d}_view.jpg"
                cv2.imwrite(str(view_file), frame)

        display = resize_display(frame)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            manual_file = save_dir / f"frame{frame_idx:05d}_manual.jpg"
            cv2.imwrite(str(manual_file), frame)
            print(f"[frame {frame_idx}] da chup: {manual_file}")

        ok, frame = capture.read()
        if not ok:
            print("Mat tin hieu camera.")
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()