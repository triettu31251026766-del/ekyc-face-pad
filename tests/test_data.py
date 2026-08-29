"""tests/test_data.py — kiểm thử (unit test) cho module src/data.py.

Tệp này dùng để (theo mục 34 tài liệu kỹ thuật):
- Kiểm tra validate đường dẫn dataset (thiếu root / thiếu annotation phải báo lỗi).
- Kiểm tra nhãn sau chuẩn hóa chỉ gồm 0 (bona_fide) và 1 (spoof).
- Kiểm tra split "subject_disjoint" không để subject trùng giữa train/val/test.
- Kiểm tra mọi mẫu đều được gán vào đúng một split.
- Kiểm tra split tái lập được (cùng seed => cùng kết quả) và lưu/đọc splits.

Chạy kiểm thử:
    python -m pytest tests/test_data.py
"""

import pytest
from PIL import Image

from src.data import (
    DataError,
    build_samples,
    create_splits,
    discover_dataset,
    extract_subject_id,
    label_distribution,
    load_metadata,
    load_splits,
    save_splits,
)


def _make_image(path, size=(8, 8)):
    """Tạo ảnh RGB nhỏ để giả lập ảnh dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)


def _make_celeba_like_dataset(tmp_path):
    """Dựng cấu trúc giả lập CelebA-Spoof:

    root/
      SpoofingData/            # thư mục ảnh
        <subject>_<frame>.jpg  # tên ảnh: phần trước "_" là subject id
      train_list.txt           # mỗi dòng: <ảnh> <nhãn thô> <illum> <env>
    """
    root = tmp_path / "celeba_spoof"
    image_root = root / "SpoofingData"
    image_root.mkdir(parents=True)

    # 6 subject, mỗi subject có cả ảnh live (0) và spoof (1, 2, ...).
    lines = []
    images = []
    for subject in range(1001, 1007):
        for frame, raw_label in enumerate([0, 1, 0, 2]):
            name = f"{subject}_{frame}.jpg"
            _make_image(image_root / name)
            images.append((name, raw_label))
            lines.append(f"{name} {raw_label} 0 0")

    (root / "train_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root, image_root, images


# --- Kiểm tra discover_dataset ---


def test_discover_dataset_finds_annotation_and_images(tmp_path):
    root, _, _ = _make_celeba_like_dataset(tmp_path)
    info = discover_dataset(root)
    assert info["annotation_file"].endswith("train_list.txt")
    assert info["image_root"].endswith("SpoofingData")


def test_discover_dataset_missing_root_raises(tmp_path):
    with pytest.raises(DataError, match="not found"):
        discover_dataset(tmp_path / "does_not_exist")


def test_discover_dataset_missing_annotation_raises(tmp_path):
    empty = tmp_path / "empty_root"
    empty.mkdir()
    with pytest.raises(DataError, match="annotation"):
        discover_dataset(empty)


# --- Kiểm tra load_metadata và build_samples ---


def test_load_metadata_parses_rows(tmp_path):
    root, _, _ = _make_celeba_like_dataset(tmp_path)
    rows = load_metadata(root)
    assert len(rows) == 24
    first = rows[0]
    assert first["path"].endswith(".jpg")
    assert isinstance(first["raw_label"], int)


def test_load_metadata_skips_comments_and_blank_lines(tmp_path):
    root, _, _ = _make_celeba_like_dataset(tmp_path)
    ann = root / "train_list.txt"
    ann.write_text("# chú thích\n\n" + ann.read_text(encoding="utf-8"), encoding="utf-8")
    assert len(load_metadata(root)) == 24


def test_build_samples_labels_only_zero_and_one(tmp_path):
    root, image_root, _ = _make_celeba_like_dataset(tmp_path)
    samples = build_samples(load_metadata(root), image_root=image_root)
    assert samples, "danh sách mẫu không được rỗng"
    assert all(s.label in (0, 1) for s in samples)


def test_build_samples_attack_type_mapping(tmp_path):
    root, image_root, _ = _make_celeba_like_dataset(tmp_path)
    samples = build_samples(load_metadata(root), image_root=image_root)
    by_attack = {s.attack_type for s in samples}
    assert "bona_fide" in by_attack
    assert "photo" in by_attack  # nhãn thô 1 -> spoof kiểu photo
    assert "poster" in by_attack  # nhãn thô 2 -> spoof kiểu poster
    assert not any(s.label == 1 and s.attack_type == "bona_fide" for s in samples)


def test_build_samples_missing_image_raises(tmp_path):
    root, image_root, _ = _make_celeba_like_dataset(tmp_path)
    rows = load_metadata(root)
    (image_root / "1001_0.jpg").unlink()
    with pytest.raises(DataError, match="not found"):
        build_samples(rows, image_root=image_root)


def test_extract_subject_id():
    assert extract_subject_id("12345_10.jpg") == "12345"
    assert extract_subject_id("plain.jpg") == "plain"


# --- Kiểm tra create_splits ---


def _build_samples(tmp_path):
    root, image_root, _ = _make_celeba_like_dataset(tmp_path)
    return build_samples(load_metadata(root), image_root=image_root)


def test_subject_disjoint_no_overlap(tmp_path):
    samples = _build_samples(tmp_path)
    splits = create_splits(samples, seed=123, strategy="subject_disjoint",
                           val_ratio=1 / 3, test_ratio=1 / 3)
    train_ids = {s.subject_id for s in splits["train"]}
    val_ids = {s.subject_id for s in splits["val"]}
    test_ids = {s.subject_id for s in splits["test"]}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_every_sample_assigned_once(tmp_path):
    samples = _build_samples(tmp_path)
    splits = create_splits(samples, seed=7, strategy="subject_disjoint",
                           val_ratio=0.25, test_ratio=0.25)
    assigned = splits["train"] + splits["val"] + splits["test"]
    paths = [s.path for s in assigned]
    assert len(paths) == len(samples)
    assert len(set(paths)) == len(samples), "không được trùng hoặc thiếu mẫu"
    assert set(paths) == {s.path for s in samples}


def test_splits_reproducible_with_same_seed(tmp_path):
    samples = _build_samples(tmp_path)
    first = create_splits(samples, seed=123, strategy="subject_disjoint")
    second = create_splits(samples, seed=123, strategy="subject_disjoint")
    for name in ("train", "val", "test"):
        assert [s.path for s in first[name]] == [s.path for s in second[name]]


def test_random_strategy_assigns_every_sample(tmp_path):
    samples = _build_samples(tmp_path)
    splits = create_splits(samples, seed=1, strategy="random",
                           val_ratio=0.2, test_ratio=0.2)
    assigned = splits["train"] + splits["val"] + splits["test"]
    assert len(assigned) == len(samples)


def test_invalid_label_raises(tmp_path):
    samples = _build_samples(tmp_path)
    samples[0].label = 2
    with pytest.raises(DataError, match="label"):
        create_splits(samples, seed=123)


def test_invalid_ratios_raise(tmp_path):
    samples = _build_samples(tmp_path)
    with pytest.raises(DataError, match="ratio"):
        create_splits(samples, seed=123, val_ratio=0.6, test_ratio=0.6)


# --- Kiểm tra save_splits / load_splits ---


def test_save_and_load_splits_roundtrip(tmp_path):
    samples = _build_samples(tmp_path)
    splits = create_splits(samples, seed=123, strategy="subject_disjoint",
                           val_ratio=0.25, test_ratio=0.25)
    out = tmp_path / "splits.json"
    save_splits(splits, out)
    loaded = load_splits(out)
    assert loaded["meta"]["seed"] == 123
    assert loaded["meta"]["strategy"] == "subject_disjoint"
    for name in ("train", "val", "test"):
        assert [s.path for s in loaded[name]] == [s.path for s in splits[name]]
        assert all(s.label in (0, 1) for s in loaded[name])


def test_load_splits_missing_file_raises(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_splits(tmp_path / "nope.json")


# --- Kiểm tra label_distribution ---


def test_label_distribution_counts(tmp_path):
    samples = _build_samples(tmp_path)
    dist = label_distribution(samples)
    assert dist["total"] == 24
    assert dist["bona_fide"] + dist["spoof"] == 24
    assert 0.0 <= dist["spoof_ratio"] <= 1.0
