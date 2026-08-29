"""scripts/collect_webcam_data.py — thu thập ảnh webcam để fine-tune model PAD.

Tệp này dùng để:
- Mở webcam, phát hiện khuôn mặt (YuNet) và tự động lưu crop khuôn mặt
  (mở rộng 10% như ảnh dataset) vào data/webcam_data/live/ hoặc
  data/webcam_data/spoof/.
- Mode "live": bạn ngồi trước camera, cử động/đổi góc/đủ sáng — ảnh sẽ được
  gán nhãn bona_fide.
- Mode "spoof": đưa ảnh mặt in giấy / mặt trên điện thoại vào trước camera —
  ảnh được gán nhãn spoof.

Cách dùng (chạy từ thư mục gốc dự án):
    python -m scripts.collect_webcam_data --mode live --count 300
    python -m scripts.collect_webcam_data --mode spoof --count 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 720
WINDOW_NAME = "Thu thap du lieu webcam (q de thoat)"

YUNET_MODEL = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"


def parse_args() -> argparse.Namespace:
    """Đọc tham số: mode, số ảnh, tỉ lệ mở rộng khung, tần suất lưu."""
    parser = argparse.ArgumentParser(description="Thu thập ảnh webcam để fine-tune")
    parser.add_argument("--mode", choices=["live", "spoof"], default="live",
                        help="live = mặt thật (bona_fide), spoof = ảnh/điện thoại")
    parser.add_argument("--count", type=int, default=300,
                        help="số ảnh muốn thu (mặc định 300)")
    parser.add_argument("--margin", type=float, default=0.10,
                        help="tỉ lệ mở rộng khung quanh mặt (mặc định 0.10)")
    parser.add_argument("--every", type=int, default=4,
                        help="lưu 1 ảnh sau mỗi N khung có mặt (mặc định 4)")
    parser.add_argument("--out", default="data/webcam_data",
                        help="thư mục lưu ảnh (mặc định data/webcam_data)")
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Mở webcam, phát hiện mặt và lưu crop cho đến khi đủ số lượng."""
    args = parse_args()

    out_dir = Path(args.out) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    if not YUNET_MODEL.is_file():
        raise FileNotFoundError(f"Khong tim thay YuNet model: {YUNET_MODEL}")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Khong mo duoc camera so {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ok, first_frame = capture.read()
    if not ok:
        raise RuntimeError("Khong doc duoc khung hinh tu camera")
    frame_h, frame_w = first_frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (frame_w, frame_h),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print(f"Mode: {args.mode} | luu vao {out_dir} | muc tieu {args.count} anh")
    print("Nhan 'q' hoac ESC de dung.")

    saved = 0
    frame_idx = 0
    frame = first_frame
    while saved < args.count:
        frame = cv2.flip(frame, 1)
        frame_idx += 1

        _, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            x, y, w, h = (int(round(v)) for v in faces[0][:4])
            pad_w, pad_h = int(w * args.margin), int(h * args.margin)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(frame_w, x + w + pad_w), min(frame_h, y + h + pad_h)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0 and frame_idx % args.every == 0:
                name = out_dir / f"img_{saved:05d}.jpg"
                cv2.imwrite(str(name), crop)
                saved += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, f"{saved}/{args.count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3)
        else:
            cv2.putText(frame, "Khong tim thay khuon mat", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

        ok, frame = capture.read()
        if not ok:
            break

    capture.release()
    cv2.destroyAllWindows()
    print(f"Da luu {saved} anh vao {out_dir}")


if __name__ == "__main__":
    main()
