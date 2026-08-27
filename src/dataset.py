"""src/dataset.py — lớp PyTorch Dataset cho dữ liệu PAD.

Tệp này dùng để (theo mục 8 của tài liệu kỹ thuật):
- Bọc danh sách Sample (từ src/data.py) thành torch.utils.data.Dataset.
- Mỗi mẫu trả về dict với cấu trúc CỐ ĐỊNH:
      {"image", "label", "path", "subject_id", "attack_type"}
  trong đó:
      image       -> torch.Tensor (đã qua transform, dải [0, 1] theo ToTensor)
      label       -> int (0 = bona_fide, 1 = spoof)
      path        -> str (đường dẫn tệp ảnh)
      subject_id  -> str (mã người, có thể rỗng)
      attack_type -> str (kiểu tấn công, ví dụ "photo", "bona_fide")
- Cấu trúc dict giống nhau giữa tập train và tập eval (không trả khác dạng).

Chú ý: module này KHÔNG tạo split, KHÔNG chứa logic model. Transform do
src/transforms.py hoặc caller cung cấp.

Cách dùng:
    from torch.utils.data import DataLoader
    from src.dataset import PADDataset
    dataset = PADDataset(samples, transform=train_transform)
    loader = DataLoader(dataset, batch_size=64)
"""

from __future__ import annotations

from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from src.data import Sample

# Transform mặc định khi không truyền transform: chỉ chuyển PIL -> Tensor [0, 1].
# (Các bước resize/normalize đầy đủ nằm ở src/transforms.py, xem mục 9 tài liệu.)
DEFAULT_TRANSFORM = ToTensor()


class PADDataset(Dataset):
    """Dataset PAD từ danh sách Sample (mục 8 tài liệu)."""

    def __init__(
        self,
        samples: list[Sample],
        transform: Callable | None = None,
    ):
        """Khởi tạo dataset.

        Args:
            samples: Danh sách đối tượng Sample (xem src/data.py).
            transform: Callable nhận ảnh PIL và trả về Tensor (nếu None thì
                dùng DEFAULT_TRANSFORM = ToTensor).
        """
        self.samples = list(samples)
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

    def __len__(self) -> int:
        """Trả về số mẫu trong dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """Trả về mẫu thứ index dưới dạng dict cấu trúc cố định.

        Raises:
            RuntimeError: nếu tệp ảnh không mở được (kèm đường dẫn để dễ dò lỗi).
        """
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as handle:
                # Chuẩn hóa về RGB để mọi ảnh có 3 kênh (ảnh grayscale/P có thể < 3).
                image = handle.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {sample.path}") from exc

        # Áp dụng transform để có Tensor theo đúng chuẩn của model.
        image = self.transform(image)

        return {
            "image": image,
            "label": int(sample.label),
            "path": str(sample.path),
            "subject_id": str(sample.subject_id),
            "attack_type": str(sample.attack_type),
        }
