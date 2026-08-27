"""tests/test_dataset.py — kiểm thử (unit test) cho module src/dataset.py.

Tệp này dùng để (theo mục 34 tài liệu kỹ thuật):
- Kiểm tra len(dataset) đúng số mẫu.
- Kiểm tra sample["image"] là torch.Tensor với kích thước chuẩn (C, H, W).
- Kiểm tra sample["label"] là số nguyên 0/1.
- Kiểm tra cấu trúc dict trả về giống nhau giữa mọi mẫu (train và eval).
- Kiểm tra transform tùy chỉnh được áp dụng, và lỗi rõ ràng khi ảnh hỏng.

Chạy kiểm thử:
    python -m pytest tests/test_dataset.py
"""

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize

from src.data import Sample
from src.dataset import PADDataset


def _make_image(path, size=(16, 12), color=(200, 100, 50)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _make_samples(tmp_path, count=4):
    """Tạo danh sách Sample giả với ảnh thật trên đĩa."""
    samples = []
    for i in range(count):
        img = tmp_path / "imgs" / f"{1000 + i}_0.jpg"
        _make_image(img)
        samples.append(
            Sample(
                path=str(img),
                label=i % 2,
                subject_id=f"{1000 + i}",
                attack_type="bona_fide" if i % 2 == 0 else "photo",
                metadata={},
            )
        )
    return samples


def test_len_dataset(tmp_path):
    samples = _make_samples(tmp_path, count=5)
    assert len(PADDataset(samples)) == 5


def test_item_returns_tensor_image_and_int_label(tmp_path):
    samples = _make_samples(tmp_path)
    dataset = PADDataset(samples)
    item = dataset[0]
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].ndim == 3 and item["image"].shape[0] == 3
    assert isinstance(item["label"], int)
    assert item["label"] in (0, 1)


def test_item_structure_consistent_across_samples(tmp_path):
    """Cấu trúc dict phải giống nhau cho mọi mẫu (mục 8 tài liệu)."""
    samples = _make_samples(tmp_path)
    dataset = PADDataset(samples)
    keys = {frozenset(dataset[i].keys()) for i in range(len(dataset))}
    assert len(keys) == 1
    assert keys.pop() == {"image", "label", "path", "subject_id", "attack_type"}


def test_item_metadata_fields(tmp_path):
    samples = _make_samples(tmp_path)
    dataset = PADDataset(samples)
    item = dataset[0]
    assert item["path"].endswith(".jpg")
    assert item["subject_id"] == "1000"
    assert item["attack_type"] == "bona_fide"


def test_custom_transform_is_applied(tmp_path):
    samples = _make_samples(tmp_path)
    # Transform tùy chỉnh: resize về 8x8 rồi chuyển Tensor.
    transform = lambda image: ToTensor()(image.resize((8, 8)))
    dataset = PADDataset(samples, transform=transform)
    item = dataset[0]
    assert item["image"].shape == (3, 8, 8)


def test_dataloader_works(tmp_path):
    samples = _make_samples(tmp_path, count=6)
    loader = DataLoader(PADDataset(samples), batch_size=2)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 12, 16)
    assert batch["label"].tolist() == [0, 1]
    # Tổng số batch = 3 cho 6 mẫu với batch_size 2.
    assert len(loader) == 3


def test_missing_image_raises_clear_error(tmp_path):
    samples = _make_samples(tmp_path)
    samples[0].path = str(tmp_path / "imgs" / "khong_ton_tai.jpg")
    dataset = PADDataset(samples)
    with pytest.raises(RuntimeError, match="Failed to load image"):
        dataset[0]
