"""src/data.py — module phát hiện dataset, đọc metadata và tạo splits.

Tệp này dùng để (theo mục 6, 7 của tài liệu kỹ thuật):
- Tìm tệp dataset và tệp annotation (discover_dataset).
- Đọc và kiểm tra metadata (load_metadata).
- Chuẩn hóa mỗi mẫu về dạng Sample(path, label, subject_id, attack_type, metadata)
  với quy ước nhãn: 0 = bona_fide, 1 = spoof.
- Tạo splits train/val/test có thể tái lập (create_splits), ưu tiên
  chiến lược "subject_disjoint" (không để ảnh của cùng 1 người nằm ở cả
  train lẫn test).
- Lưu/đọc thông tin splits (save_splits / load_splits).

Chú ý: module này KHÔNG huấn luyện model, KHÔNG tính metric. Chỉ phụ trách dữ liệu.

Cách dùng:
    from src.data import discover_dataset, load_metadata, build_samples, create_splits
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# --- Hằng số quy ước nhãn toàn dự án (mục 6 tài liệu) ---
LABEL_BONA_FIDE = 0
LABEL_SPOOF = 1

# Tên kiểu tấn công theo nhãn thô của CelebA-Spoof:
# 0 = live (bona_fide), 1 = photo, 2 = poster, 3 = paper, 4 = mask.
ATTACK_TYPE_NAMES = {
    0: "bona_fide",
    1: "photo",
    2: "poster",
    3: "paper",
    4: "mask",
}

# Các tên tệp annotation thường gặp (tìm theo thứ tự ưu tiên).
DEFAULT_ANNOTATION_FILES = (
    "train_list.txt",
    "test_list.txt",
    "train.txt",
    "test.txt",
    "labels.txt",
    "annotations.txt",
)

# Các thư mục chứa ảnh thường gặp (tìm theo thứ tự ưu tiên).
DEFAULT_IMAGE_DIRS = (
    "SpoofingData",
    "spoofingdata",
    "images",
    "frames",
    "data",
)


class DataError(ValueError):
    """Ném ra khi dữ liệu dataset không hợp lệ hoặc không tìm thấy."""


@dataclass
class Sample:
    """Biểu diễn chuẩn hóa của một mẫu ảnh (mục 6 tài liệu)."""

    path: str
    label: int
    subject_id: str
    attack_type: str
    metadata: dict = field(default_factory=dict)


def discover_dataset(
    root: str | Path,
    annotation_file: str | Path | None = None,
    image_dir: str | Path | None = None,
) -> dict:
    """Phát hiện cấu trúc dataset: tìm tệp annotation và thư mục chứa ảnh.

    Args:
        root: Thư mục gốc dataset.
        annotation_file: Tệp annotation chỉ định sẵn (nếu None sẽ tự tìm).
        image_dir: Thư mục ảnh chỉ định sẵn (nếu None sẽ tự tìm).

    Returns:
        dict với các khóa "root", "image_root", "annotation_file" (đường dẫn tuyệt đối).

    Raises:
        DataError: nếu root không tồn tại hoặc không tìm thấy tệp annotation.
    """
    root = Path(root)
    if not root.is_dir():
        raise DataError(f"Dataset root not found: {root}")

    # Bước 1: xác định tệp annotation.
    if annotation_file is not None:
        annotation_path = Path(annotation_file)
        if not annotation_path.is_absolute():
            annotation_path = root / annotation_path
        if not annotation_path.is_file():
            raise DataError(f"Annotation file not found: {annotation_path}")
    else:
        annotation_path = _find_annotation_file(root)
        if annotation_path is None:
            raise DataError(
                f"No annotation file found under '{root}'. "
                f"Tried names: {', '.join(DEFAULT_ANNOTATION_FILES)}. "
                f"Use the 'annotation_file' argument to specify one explicitly."
            )

    # Bước 2: xác định thư mục chứa ảnh.
    if image_dir is not None:
        image_path = Path(image_dir)
        if not image_path.is_absolute():
            image_path = root / image_path
        if not image_path.is_dir():
            raise DataError(f"Image directory not found: {image_path}")
    else:
        image_path = _find_image_dir(root)

    return {
        "root": str(root.resolve()),
        "image_root": str(image_path.resolve()),
        "annotation_file": str(annotation_path.resolve()),
    }


def load_metadata(
    root: str | Path,
    annotation_file: str | Path | None = None,
) -> list[dict]:
    """Đọc tệp annotation và trả về danh sách các dòng thô.

    Định dạng mỗi dòng (phân tách bằng khoảng trắng):
        <đường_dẫn_ảnh_tương_đối> <nhãn_thô> [<cột_phụ> ...]

    Nhãn thô: 0 = live (bona_fide), 1..4 = các kiểu spoof (CelebA-Spoof).
    Các cột phụ (nếu có) được lưu vào metadata dưới tên "illumination",
    "environment" (vị trí thứ 3, 4) và danh sách "extra".

    Args:
        root: Thư mục gốc dataset.
        annotation_file: Tệp annotation (nếu None sẽ tự phát hiện).

    Returns:
        Danh sách dict: {"path", "raw_label", "illumination", "environment", "extra"}.

    Raises:
        DataError: nếu tệp annotation không hợp lệ hoặc dòng sai định dạng.
    """
    info = discover_dataset(root, annotation_file=annotation_file)
    annotation_path = Path(info["annotation_file"])

    rows: list[dict] = []
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            # Bỏ qua dòng trống và dòng chú thích.
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            if len(tokens) < 2:
                raise DataError(
                    f"Invalid annotation line {line_number} in "
                    f"'{annotation_path}': expected '<path> <label> [...]', "
                    f"got {line!r}"
                )

            rel_path = tokens[0]
            try:
                raw_label = int(tokens[1])
            except ValueError as exc:
                raise DataError(
                    f"Invalid label on line {line_number} in '{annotation_path}': "
                    f"{tokens[1]!r} is not an integer"
                ) from exc
            if raw_label < 0:
                raise DataError(
                    f"Invalid label on line {line_number} in '{annotation_path}': "
                    f"label must be >= 0, got {raw_label}"
                )

            extra_tokens = tokens[2:]
            row: dict[str, Any] = {"path": rel_path, "raw_label": raw_label, "extra": extra_tokens}
            # Cột thứ 3, 4 (nếu có) là illumination, environment của CelebA-Spoof.
            if len(extra_tokens) >= 1:
                row["illumination"] = extra_tokens[0]
            if len(extra_tokens) >= 2:
                row["environment"] = extra_tokens[1]
            rows.append(row)

    if not rows:
        raise DataError(f"Annotation file is empty or has no valid rows: {annotation_path}")
    return rows


def build_samples(
    metadata: list[dict],
    image_root: str | Path | None = None,
    check_exists: bool = True,
) -> list[Sample]:
    """Chuẩn hóa metadata thô thành danh sách Sample theo quy ước nhãn dự án.

    Quy tắc chuyển nhãn (mục 6 tài liệu):
        raw_label == 0  ->  label 0 (bona_fide)
        raw_label >= 1  ->  label 1 (spoof), attack_type theo ATTACK_TYPE_NAMES.

    Args:
        metadata: Kết quả của load_metadata.
        image_root: Thư mục gốc để ghép với đường dẫn ảnh tương đối.
        check_exists: Nếu True, kiểm tra tệp ảnh có tồn tại (báo lỗi rõ ràng).

    Returns:
        Danh sách đối tượng Sample.

    Raises:
        DataError: nếu nhãn không hợp lệ hoặc (khi check_exists) tệp ảnh bị thiếu.
    """
    image_root = Path(image_root) if image_root is not None else Path(".")

    samples: list[Sample] = []
    for row in metadata:
        rel_path = Path(str(row["path"]))
        raw_label = int(row["raw_label"])

        # Chuẩn hóa nhãn về 0/1 và xác định kiểu tấn công.
        label = LABEL_BONA_FIDE if raw_label == 0 else LABEL_SPOOF
        attack_type = ATTACK_TYPE_NAMES.get(raw_label, f"attack_{raw_label}")

        full_path = image_root / rel_path
        if check_exists and not full_path.is_file():
            raise DataError(f"Image file not found: {full_path}")

        sample_metadata: dict[str, Any] = {"raw_label": raw_label}
        for key in ("illumination", "environment"):
            if key in row:
                sample_metadata[key] = row[key]

        samples.append(
            Sample(
                path=str(full_path),
                label=label,
                subject_id=extract_subject_id(rel_path.name),
                attack_type=attack_type,
                metadata=sample_metadata,
            )
        )

    return samples


def extract_subject_id(filename: str, separator: str = "_") -> str:
    """Trích mã định danh người (subject) từ tên tệp ảnh.

    Ví dụ CelebA-Spoof: "12345_10.jpg" -> "12345" (phần trước dấu "_" đầu tiên).
    Nếu tên không chứa dấu phân tách, dùng toàn bộ tên (bỏ đuôi mở rộng).
    """
    stem = Path(filename).stem
    return stem.split(separator, 1)[0] if separator in stem else stem


def label_distribution(samples: list[Sample]) -> dict:
    """Tính phân bố nhãn của danh sách mẫu (mục 15 tài liệu).

    Returns:
        {"total", "bona_fide", "spoof", "spoof_ratio"}
    """
    total = len(samples)
    num_spoof = sum(1 for sample in samples if sample.label == LABEL_SPOOF)
    num_bona_fide = total - num_spoof
    return {
        "total": total,
        "bona_fide": num_bona_fide,
        "spoof": num_spoof,
        "spoof_ratio": (num_spoof / total) if total else 0.0,
    }


def create_splits(
    samples: list[Sample],
    seed: int,
    strategy: str = "subject_disjoint",
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> dict:
    """Tạo splits train/val/test có thể tái lập từ danh sách mẫu.

    Chiến lược (mục 7 tài liệu):
        - "subject_disjoint": gom toàn bộ ảnh của cùng một subject vào cùng
          một split; các split không trùng subject. Mẫu không có subject_id
          được xem như subject riêng lẻ (theo đường dẫn) để tránh rò rỉ dữ liệu.
        - "random": trộn ngẫu nhiên ở mức mẫu ảnh.

    Tính tái lập: cùng seed + cùng danh sách mẫu => cùng kết quả split.

    Args:
        samples: Danh sách Sample cần chia.
        seed: Seed cho bộ sinh số ngẫu nhiên.
        strategy: "subject_disjoint" hoặc "random".
        val_ratio: Tỉ lệ mẫu cho tập val (0 <= val_ratio < 1).
        test_ratio: Tỉ lệ mẫu cho tập test (0 <= test_ratio < 1).

    Returns:
        {"train": [...], "val": [...], "test": [...], "meta": {...}}.

    Raises:
        DataError: nếu tham số không hợp lệ hoặc nhãn không thuộc {0, 1}.
    """
    if not samples:
        raise DataError("Cannot create splits from an empty sample list")
    if strategy not in ("subject_disjoint", "random"):
        raise DataError(f"Unknown split strategy: {strategy!r}")
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise DataError(f"val_ratio/test_ratio must be in [0, 1), got {val_ratio}, {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise DataError(
            f"val_ratio + test_ratio must be < 1.0, got {val_ratio} + {test_ratio}"
        )

    # Kiểm tra nhãn hợp lệ trước khi chia.
    for sample in samples:
        if sample.label not in (LABEL_BONA_FIDE, LABEL_SPOOF):
            raise DataError(
                f"Invalid label {sample.label!r} for {sample.path}. "
                f"Labels must be 0 (bona_fide) or 1 (spoof)."
            )

    # Dùng bộ sinh ngẫu nhiên cục bộ: không phụ thuộc trạng thái RNG toàn cục.
    rng = random.Random(seed)

    if strategy == "random":
        order = list(samples)
        rng.shuffle(order)
        train, val, test = _slice_by_ratio(order, val_ratio, test_ratio)
    else:
        # Gộp mẫu theo subject. Mẫu thiếu subject_id được tách thành
        # "giả subject" riêng theo đường dẫn để không gây rò rỉ.
        groups: dict[str, list[Sample]] = {}
        for sample in samples:
            key = sample.subject_id or f"__single__{sample.path}"
            groups.setdefault(key, []).append(sample)

        subject_keys = sorted(groups.keys())
        rng.shuffle(subject_keys)

        # Lần lượt gán subject vào train, val, test theo số lượng mục tiêu.
        n_total = len(samples)
        n_val = max(int(round(n_total * val_ratio)), 0)
        n_test = max(int(round(n_total * test_ratio)), 0)
        n_train = n_total - n_val - n_test

        train, val, test = [], [], []
        for key in subject_keys:
            group = groups[key]
            if len(train) < n_train:
                train.extend(group)
            elif len(val) < n_val:
                val.extend(group)
            else:
                test.extend(group)

    meta = {
        "seed": seed,
        "strategy": strategy,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total": len(samples),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
    }
    return {"train": train, "val": val, "test": test, "meta": meta}


def save_splits(splits: dict, path: str | Path) -> None:
    """Lưu thông tin splits ra tệp JSON (đường dẫn, nhãn, subject, attack_type).

    Tệp lưu bao gồm phần "meta" (seed, strategy, counts) để có thể tái lập
    và kiểm tra chiến lược split khi viết kết quả thí nghiệm.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "meta": dict(splits["meta"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits": {
            name: [asdict(sample) for sample in splits[name]]
            for name in ("train", "val", "test")
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_splits(path: str | Path) -> dict:
    """Đọc lại splits từ tệp JSON đã lưu bởi save_splits.

    Returns:
        Cùng cấu trúc với kết quả create_splits: {"train", "val", "test", "meta"}.

    Raises:
        DataError: nếu tệp không tồn tại hoặc nội dung không hợp lệ.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Splits file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataError(f"Failed to parse splits file '{path}': {exc}") from exc

    required = ("train", "val", "test")
    raw_splits = data.get("splits")
    if not isinstance(raw_splits, dict) or any(name not in raw_splits for name in required):
        raise DataError(f"Splits file '{path}' is missing one of {list(required)}")

    result: dict[str, Any] = {"meta": data.get("meta", {})}
    for name in required:
        result[name] = [_sample_from_dict(item) for item in raw_splits[name]]
    return result


# --- Các hàm nội bộ ---


def _find_annotation_file(root: Path) -> Path | None:
    """Tìm tệp annotation trong root và các thư mục con trực tiếp."""
    for name in DEFAULT_ANNOTATION_FILES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    for child in sorted(root.iterdir()):
        if child.is_dir():
            for name in DEFAULT_ANNOTATION_FILES:
                candidate = child / name
                if candidate.is_file():
                    return candidate
    return None


def _find_image_dir(root: Path) -> Path:
    """Tìm thư mục chứa ảnh; nếu không có thư mục quen thuộc thì dùng chính root."""
    for name in DEFAULT_IMAGE_DIRS:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root


def _slice_by_ratio(
    ordered: list[Sample],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Chia danh sách đã trộn thành 3 phần theo tỉ lệ val/test (phần còn lại là train)."""
    n_total = len(ordered)
    n_val = max(int(round(n_total * val_ratio)), 0)
    n_test = max(int(round(n_total * test_ratio)), 0)
    n_train = n_total - n_val - n_test

    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    return train, val, test


def _sample_from_dict(item: dict) -> Sample:
    """Dựng lại đối tượng Sample từ dict (dùng trong load_splits)."""
    return Sample(
        path=str(item["path"]),
        label=int(item["label"]),
        subject_id=str(item.get("subject_id", "")),
        attack_type=str(item.get("attack_type", "")),
        metadata=dict(item.get("metadata", {})),
    )
