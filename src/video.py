"""src/video.py — xử lý video cho bài toán PAD.

Tệp này dùng để (theo mục 20 của tài liệu kỹ thuật):
Pipeline khuyến nghị:
      video
        -> sample N frames (sample_frames)
        -> crop khuôn mặt (crop_face, tùy chọn)
        -> tiền xử lý (transform từ src/transforms.py)
        -> PAD model (predict_frames)
        -> xác suất từng frame
        -> gộp điểm (aggregate_frame_scores)
        -> xác suất video

Các hàm chính:
- sample_frames(video_path, num_frames): lấy num_frames khung hình CÁCH ĐỀU
  nhau trong video (tất định, không ngẫu nhiên), trả về danh sách ảnh PIL RGB.
- crop_face(frame): phát hiện và cắt khuôn mặt lớn nhất bằng Haar cascade
  của OpenCV; nếu không tìm thấy khuôn mặt thì trả nguyên khung hình.
- predict_frames(model, frames, transform, device, threshold): chạy model
  trên từng frame, trả về danh sách xác suất spoof.
- aggregate_frame_scores(scores, method): gộp điểm các frame; bắt đầu với
  method="mean" (mục 20 tài liệu), hỗ trợ thêm "median".
- predict_video(...): tiện ích chạy toàn bộ pipeline cho một tệp video.

Chú ý: KHÔNG thêm mô hình temporal phức tạp cho đến khi pipeline theo
frame hoạt động đúng (mục 20). Mọi dự đoán dựa trên src/inference.py.

Cách dùng:
    from src.video import predict_video
    result = predict_video(model, "clip.mp4", eval_transform, num_frames=10)
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from PIL import Image
from torch import nn

from src.inference import predict_pil

VALID_AGGREGATION = ("mean", "median")


def sample_frames(
    video_path: str,
    num_frames: int,
) -> list[Image.Image]:
    """Lấy num_frames khung hình cách đều trong video (mục 20 tài liệu).

    Cách lấy tất định: các chỉ số frame được chọn cách đều trên toàn video.
    Nếu video có ít frame hơn num_frames thì trả về TẤT CẢ các frame.

    Args:
        video_path: Đường dẫn tệp video.
        num_frames: Số khung hình muốn lấy (>= 1).

    Returns:
        Danh sách ảnh PIL RGB (thứ tự thời gian).

    Raises:
        ValueError: nếu num_frames < 1 hoặc video không mở được.
        FileNotFoundError: nếu tệp video không tồn tại.
    """
    import os

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError(f"num_frames must be an integer >= 1, got {num_frames!r}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            raise ValueError(f"Video has no readable frames: {video_path}")

        count = min(num_frames, total)
        # Chỉ số cách đều, không trùng lặp (linspace + round rồi unique).
        indices = sorted({int(round(i)) for i in np.linspace(0, total - 1, count)})
        # Phòng trường hợp round làm thiếu mẫu (số frame cực ít).
        if len(indices) < count:
            indices = list(range(total))

        frames: list[Image.Image] = []
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = capture.read()
            if not ok:
                continue
            # OpenCV đọc theo BGR -> đổi sang RGB rồi thành ảnh PIL.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        if not frames:
            raise ValueError(f"Failed to read any frame from: {video_path}")
        return frames
    finally:
        capture.release()


def crop_face(frame: Image.Image) -> Image.Image:
    """Cắt khuôn mặt lớn nhất bằng Haar cascade (bước "face crop" mục 20).

    Nếu không phát hiện được khuôn mặt (ví dụ video không có người), hoặc
    tệp Haar cascade không đi kèm bản OpenCV đang cài, thì trả về ảnh GỐC
    để pipeline vẫn chạy được thay vì dừng lỗi.

    Args:
        frame: Ảnh PIL đầu vào.

    Returns:
        Ảnh PIL (có thể đã cắt, cùng mode RGB).
    """
    import os

    import cv2
    import cv2.data  # bản opencv-python 5.x chỉ có cv2.data khi import tường minh

    frame = frame.convert("RGB")

    # Kiểm tra tệp cascade có tồn tại (một số bản opencv không đóng gói kèm).
    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    if not os.path.isfile(cascade_path):
        return frame

    array = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return frame

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return frame

    # Chọn khuôn mặt có diện tích lớn nhất.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    height, width = array.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)

    cropped = array[y0:y1, x0:x1]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


def predict_frames(
    model: nn.Module,
    frames: list[Image.Image],
    transform: Callable,
    device: str = "cpu",
    threshold: float = 0.5,
) -> list[float]:
    """Chạy model PAD trên từng frame, trả về xác suất spoof của mỗi frame.

    Args:
        model: Model PAD đã huấn luyện (1 logit đầu ra).
        frames: Danh sách ảnh PIL.
        transform: Transform eval tất định (xem src/transforms.py).
        device: Thiết bị chạy.
        threshold: Ngưỡng quyết định (chỉ dùng khi cần nhãn; xác suất không
            phụ thuộc ngưỡng).

    Returns:
        Danh sách float (probability_spoof) cùng độ dài với frames.

    Raises:
        ValueError: nếu danh sách frames rỗng.
    """
    if not frames:
        raise ValueError("frames must not be empty")

    scores: list[float] = []
    for frame in frames:
        result = predict_pil(model, frame, transform, device=device, threshold=threshold)
        scores.append(result["probability_spoof"])
    return scores


def aggregate_frame_scores(scores: list[float], method: str = "mean") -> float:
    """Gộp điểm các frame thành xác suất video (mục 20 tài liệu).

    Bắt đầu với method="mean" (khuyến nghị của tài liệu); hỗ trợ thêm
    "median" để kháng frame ngoại lai.

    Args:
        scores: Danh sách xác suất spoof của các frame.
        method: "mean" hoặc "median".

    Returns:
        float: xác suất spoof của video.

    Raises:
        ValueError: nếu scores rỗng hoặc method không hợp lệ.
    """
    if not scores:
        raise ValueError("scores must not be empty")
    if method not in VALID_AGGREGATION:
        raise ValueError(
            f"Unknown aggregation method: {method!r}. "
            f"Expected one of {list(VALID_AGGREGATION)}"
        )

    values = np.asarray(scores, dtype=np.float64)
    if method == "median":
        return float(np.median(values))
    return float(np.mean(values))


def predict_video(
    model: nn.Module,
    video_path: str,
    transform: Callable,
    num_frames: int = 10,
    device: str = "cpu",
    threshold: float = 0.5,
    aggregate: str = "mean",
    use_face_crop: bool = True,
) -> dict:
    """Chạy toàn bộ pipeline PAD cho một tệp video (mục 20 tài liệu).

    Args:
        model: Model PAD đã huấn luyện.
        video_path: Đường dẫn tệp video.
        transform: Transform eval tất định.
        num_frames: Số frame lấy mẫu cách đều.
        device: Thiết bị chạy.
        threshold: Ngưỡng quyết định nhãn cuối cùng.
        aggregate: Cách gộp điểm frame ("mean" hoặc "median").
        use_face_crop: Nếu True, cắt khuôn mặt trước khi dự đoán.

    Returns:
        {"video_probability": float, "prediction": "spoof"|"bona_fide",
         "frame_scores": [...], "num_frames": int}
    """
    frames = sample_frames(video_path, num_frames)
    if use_face_crop:
        frames = [crop_face(frame) for frame in frames]

    frame_scores = predict_frames(model, frames, transform, device=device, threshold=threshold)
    video_probability = aggregate_frame_scores(frame_scores, method=aggregate)

    prediction = "spoof" if video_probability >= threshold else "bona_fide"
    return {
        "video_probability": video_probability,
        "prediction": prediction,
        "frame_scores": frame_scores,
        "num_frames": len(frame_scores),
    }
