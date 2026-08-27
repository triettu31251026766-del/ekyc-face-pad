"""tests/test_evaluate.py — kiểm thử (unit test) cho module src/evaluate.py.

Tệp này dùng để (theo mục 18 của tài liệu kỹ thuật):
- Kiểm tra evaluate_model trả về đúng cấu trúc:
  metrics + predictions + probabilities + labels + paths + subject_ids + attack_types.
- Kiểm tra các danh sách trả về cùng độ dài bằng số mẫu dataset.
- Kiểm tra đánh giá là TẤT ĐỊNH: 2 lần chạy cho kết quả giống hệt nhau.
- Kiểm tra ngưỡng threshold thay đổi dự đoán đúng như định nghĩa.
- Kiểm tra save_predictions ghi CSV đúng cột và đúng số dòng.

Chạy kiểm thử:
    python -m pytest tests/test_evaluate.py
"""

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.data import Sample
from src.dataset import PADDataset
from src.evaluate import evaluate_model, save_predictions
from src.model import build_model


def _make_loader(tmp_path, n_samples=12):
    """DataLoader với ảnh đơn giản: class 0 tối, class 1 sáng."""
    samples = []
    for i in range(n_samples):
        img = tmp_path / f"{i}.png"
        label = i % 2
        base = 40 if label == 0 else 215
        Image.new("RGB", (32, 32), color=(base, base, base)).save(img)
        samples.append(
            Sample(path=str(img), label=label, subject_id=f"s{i}",
                   attack_type="bona_fide" if label == 0 else "photo")
        )
    dataset = PADDataset(samples)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_evaluate_returns_expected_structure(tmp_path):
    torch.manual_seed(0)
    model = build_model("custom_cnn", num_classes=1)
    loader = _make_loader(tmp_path, n_samples=12)

    result = evaluate_model(model, loader, device="cpu", threshold=0.5)
    assert set(result.keys()) == {
        "metrics", "predictions", "probabilities", "labels",
        "paths", "subject_ids", "attack_types",
    }
    assert len(result["predictions"]) == 12
    assert len(result["probabilities"]) == 12
    assert len(result["labels"]) == 12
    assert len(result["paths"]) == 12


def test_evaluate_metric_keys(tmp_path):
    torch.manual_seed(0)
    model = build_model("custom_cnn", num_classes=1)
    result = evaluate_model(model, _make_loader(tmp_path, n_samples=8),
                            device="cpu", threshold=0.5)
    for key in ("accuracy", "precision", "recall", "f1",
                "roc_auc", "pr_auc", "apcer", "bpcer", "acer"):
        assert key in result["metrics"]


def test_evaluate_is_deterministic(tmp_path):
    """Hai lần đánh giá cùng model + cùng dữ liệu phải cho kết quả giống hệt nhau."""
    torch.manual_seed(0)
    model = build_model("custom_cnn", num_classes=1)
    loader = _make_loader(tmp_path, n_samples=8)

    first = evaluate_model(model, loader, device="cpu", threshold=0.5)
    second = evaluate_model(model, loader, device="cpu", threshold=0.5)
    assert first["probabilities"] == second["probabilities"]
    assert first["predictions"] == second["predictions"]
    assert first["metrics"] == second["metrics"]


def test_threshold_changes_predictions(tmp_path):
    torch.manual_seed(0)
    model = build_model("custom_cnn", num_classes=1)
    loader = _make_loader(tmp_path, n_samples=8)

    result = evaluate_model(model, loader, device="cpu", threshold=0.0)
    # Ngưỡng 0.0 -> mọi mẫu đều dự đoán spoof (1).
    assert all(pred == 1 for pred in result["predictions"])

    result = evaluate_model(model, loader, device="cpu", threshold=1.0)
    # Ngưỡng 1.0 -> không mẫu nào đạt -> toàn bona_fide (0).
    assert all(pred == 0 for pred in result["predictions"])


def test_save_predictions_writes_correct_csv(tmp_path):
    torch.manual_seed(0)
    model = build_model("custom_cnn", num_classes=1)
    result = evaluate_model(model, _make_loader(tmp_path, n_samples=8),
                            device="cpu", threshold=0.5)

    out = tmp_path / "predictions.csv"
    save_predictions(result, str(out))
    assert out.is_file()

    frame = pd.read_csv(out)
    expected_columns = ["path", "subject_id", "attack_type", "label",
                        "probability_spoof", "prediction", "correct"]
    assert list(frame.columns) == expected_columns
    assert len(frame) == 8
    assert frame["correct"].isin([0, 1]).all()
    assert ((frame["probability_spoof"] >= 0.0) & (frame["probability_spoof"] <= 1.0)).all()


def test_evaluate_empty_loader_raises(tmp_path):
    torch.manual_seed(0)
    model = build_model("custom_cnn", num_classes=1)
    dataset = PADDataset([])
    loader = DataLoader(dataset, batch_size=4)
    with torch.inference_mode():
        try:
            evaluate_model(model, loader, device="cpu")
            assert False, "phải ném lỗi với dataloader rỗng"
        except ValueError as exc:
            assert "empty" in str(exc)
