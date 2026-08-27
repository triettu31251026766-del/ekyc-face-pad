"""tests/test_video.py — kiểm thử (unit test) cho module src/video.py.

Tệp này dùng để (theo mục 20 của tài liệu kỹ thuật):
- Tạo video tổng hợp bằng OpenCV (MJPG) để test mà không cần dữ liệu thật.
- Kiểm tra sample_frames: đúng số frame, ảnh PIL RGB, TẤT ĐỊNH (2 lần chạy
  giống nhau), video ngắn hơn num_frames thì trả toàn bộ frame.
- Kiểm tra crop_face: không có khuôn mặt -> trả ảnh gốc (không lỗi).
- Kiểm tra predict_frames và aggregate_frame_scores (mean/median).
- Kiểm tra predict_video end-to-end với model giả logit cố định.
- Kiểm tra lỗi rõ ràng với tham số sai / tệp không tồn tại.

Chạy kiểm thử:
    python -m pytest tests/test_video.py
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor

from src.video import (
    aggregate_frame_scores,
    crop_face,
    predict_frames,
    predict_video,
    sample_frames,
)


class FixedLogitModel(nn.Module):
    """Model giả: luôn trả về logit cố định."""

    def __init__(self, logit):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.logit, dtype=torch.float32)


def _transform(image):
    return ToTensor()(image.resize((32, 32)))


def _make_video(tmp_path, n_frames=20, size=(64, 64), name="clip.avi"):
    """Tạo video MJPG với các frame màu khác nhau (tất định)."""
    path = str(tmp_path / name)
    writer = None
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    for i in range(n_frames):
        # Màu thay đổi theo frame để phân biệt được các frame khác nhau.
        color = (30 + i * 10) % 256
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_sample_frames_returns_requested_count(tmp_path):
    video = _make_video(tmp_path, n_frames=20)
    frames = sample_frames(video, num_frames=10)
    assert len(frames) == 10
    assert all(isinstance(f, Image.Image) for f in frames)
    assert all(f.mode == "RGB" for f in frames)


def test_sample_frames_is_deterministic(tmp_path):
    video = _make_video(tmp_path, n_frames=20)
    first = sample_frames(video, num_frames=5)
    second = sample_frames(video, num_frames=5)
    for a, b in zip(first, second):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_sample_frames_shorter_video_returns_all_frames(tmp_path):
    video = _make_video(tmp_path, n_frames=3)
    frames = sample_frames(video, num_frames=10)
    assert len(frames) == 3


def test_sample_frames_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        sample_frames(str(tmp_path / "khong_ton_tai.avi"), num_frames=5)


def test_sample_frames_invalid_num_frames_raises(tmp_path):
    video = _make_video(tmp_path, n_frames=5)
    with pytest.raises(ValueError, match="num_frames"):
        sample_frames(video, num_frames=0)


def test_crop_face_no_face_returns_original(tmp_path):
    """Video tổng hợp không có khuôn mặt -> crop_face trả ảnh gốc, không lỗi."""
    video = _make_video(tmp_path, n_frames=3)
    frame = sample_frames(video, num_frames=1)[0]
    cropped = crop_face(frame)
    assert cropped.size == frame.size
    assert np.array_equal(np.asarray(cropped), np.asarray(frame))


def test_predict_frames_with_fixed_model(tmp_path):
    video = _make_video(tmp_path, n_frames=6)
    frames = sample_frames(video, num_frames=4)
    model = FixedLogitModel(logit=2.0)
    scores = predict_frames(model, frames, _transform, device="cpu")
    assert len(scores) == 4
    assert all(abs(s - 0.8808) < 1e-3 for s in scores)  # sigmoid(2.0)


def test_predict_frames_empty_raises():
    model = FixedLogitModel(logit=0.0)
    with pytest.raises(ValueError, match="empty"):
        predict_frames(model, [], _transform)


def test_aggregate_mean_and_median():
    scores = [0.1, 0.2, 0.9]
    assert aggregate_frame_scores(scores, method="mean") == pytest.approx(0.4)
    assert aggregate_frame_scores(scores, method="median") == pytest.approx(0.2)


def test_aggregate_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate_frame_scores([0.5], method="sum")


def test_aggregate_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        aggregate_frame_scores([])


def test_predict_video_end_to_end(tmp_path):
    video = _make_video(tmp_path, n_frames=12)
    model = FixedLogitModel(logit=2.0)
    result = predict_video(model, video, _transform, num_frames=6,
                           device="cpu", use_face_crop=False)
    assert set(result.keys()) == {"video_probability", "prediction",
                                  "frame_scores", "num_frames"}
    assert result["num_frames"] == 6
    assert result["video_probability"] == pytest.approx(0.8808, abs=1e-3)
    assert result["prediction"] == "spoof"


def test_predict_video_negative_logit_bona_fide(tmp_path):
    video = _make_video(tmp_path, n_frames=4)
    model = FixedLogitModel(logit=-2.0)
    result = predict_video(model, video, _transform, num_frames=2,
                           device="cpu", use_face_crop=False)
    assert result["prediction"] == "bona_fide"
