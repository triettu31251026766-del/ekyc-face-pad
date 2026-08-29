"""tests/test_config.py — kiểm thử (unit test) cho module src/config.py.

Tệp này dùng để:
- Kiểm tra việc nạp và validate cấu hình YAML: cấu hình hợp lệ, file không tồn tại,
  YAML sai cú pháp, thiếu trường bắt buộc, giá trị không hợp lệ, ...
- Đảm bảo mọi cấu hình lỗi đều báo lỗi rõ ràng (ConfigError).

Chạy kiểm thử:
    python -m pytest tests/test_config.py
"""

import yaml
import pytest

from src.config import ConfigError, load_config


def _write(tmp_path, content, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def _base_config():
    return {
        "seed": 123,
        "dataset": {"name": "celeba_spoof", "root": "data/raw/celeba_spoof"},
        "split": {"strategy": "subject_disjoint"},
        "model": {"name": "mobilenet_v2", "image_size": 224},
        "training": {
            "epochs": 20,
            "batch_size": 64,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
        },
        "loss": {"name": "bce_with_logits", "use_pos_weight": False},
        "evaluation": {"threshold": 0.5},
        "device": {"name": "auto"},
    }


def test_load_valid_base_config(tmp_path):
    path = _write(tmp_path, _base_config())
    config = load_config(path)
    assert config["seed"] == 123
    assert config["model"]["name"] == "mobilenet_v2"
    assert config["training"]["epochs"] == 20


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_invalid_yaml_raises(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


def test_missing_required_section_raises(tmp_path):
    config = _base_config()
    del config["dataset"]
    with pytest.raises(ConfigError, match="dataset"):
        load_config(_write(tmp_path, config))


def test_invalid_model_name_raises(tmp_path):
    config = _base_config()
    config["model"]["name"] = "resnet5000"
    with pytest.raises(ConfigError, match="model.name"):
        load_config(_write(tmp_path, config))


def test_invalid_threshold_raises(tmp_path):
    config = _base_config()
    config["evaluation"]["threshold"] = 1.5
    with pytest.raises(ConfigError, match="threshold"):
        load_config(_write(tmp_path, config))


def test_zero_epochs_raises(tmp_path):
    config = _base_config()
    config["training"]["epochs"] = 0
    with pytest.raises(ConfigError, match="epochs"):
        load_config(_write(tmp_path, config))


def test_unknown_device_raises(tmp_path):
    config = _base_config()
    config["device"]["name"] = "tpu"
    with pytest.raises(ConfigError, match="device.name"):
        load_config(_write(tmp_path, config))


def test_load_degradation_config(tmp_path):
    config = {"seed": 123, "degradation": {"name": "jpeg", "quality": 50}}
    loaded = load_config(_write(tmp_path, config))
    assert loaded["degradation"] == {"name": "jpeg", "quality": 50}


def test_invalid_degradation_name_raises(tmp_path):
    config = {"seed": 123, "degradation": {"name": "solarize", "quality": 50}}
    with pytest.raises(ConfigError, match="degradation.name"):
        load_config(_write(tmp_path, config))


def test_invalid_jpeg_quality_raises(tmp_path):
    config = {"seed": 123, "degradation": {"name": "jpeg", "quality": 101}}
    with pytest.raises(ConfigError, match="quality"):
        load_config(_write(tmp_path, config))


def test_even_blur_kernel_raises(tmp_path):
    config = {
        "seed": 123,
        "degradation": {"name": "blur", "kernel_size": 8, "sigma": 2.0},
    }
    with pytest.raises(ConfigError, match="kernel_size"):
        load_config(_write(tmp_path, config))


def test_missing_degradation_param_raises(tmp_path):
    config = {"seed": 123, "degradation": {"name": "noise"}}
    with pytest.raises(ConfigError, match="std"):
        load_config(_write(tmp_path, config))


def test_load_robustness_config(tmp_path):
    config = {
        "seed": 123,
        "robustness": {
            "enabled": True,
            "augmentations": {
                "jpeg": {"enabled": True, "quality_range": [50, 90]},
                "resize": {"enabled": True, "scale_range": [0.5, 1.0]},
                "blur": {"enabled": True, "sigma_range": [0.5, 2.0]},
                "noise": {"enabled": True, "std_range": [0.005, 0.03]},
                "brightness": {"enabled": True, "factor_range": [0.7, 1.3]},
            },
        },
    }
    loaded = load_config(_write(tmp_path, config))
    assert loaded["robustness"]["enabled"] is True


def test_disabled_augmentation_needs_no_range(tmp_path):
    config = {
        "seed": 123,
        "robustness": {
            "enabled": True,
            "augmentations": {"blur": {"enabled": False}},
        },
    }
    load_config(_write(tmp_path, config))


def test_invalid_augmentation_range_raises(tmp_path):
    config = {
        "seed": 123,
        "robustness": {
            "enabled": True,
            "augmentations": {
                "jpeg": {"enabled": True, "quality_range": [90, 50]},
            },
        },
    }
    with pytest.raises(ConfigError, match="quality_range"):
        load_config(_write(tmp_path, config))
