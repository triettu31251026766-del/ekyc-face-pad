"""src/config.py — module nạp và kiểm tra (validate) cấu hình YAML.

Tệp này dùng để:
- Nạp file cấu hình YAML của các thí nghiệm (baseline, degradation, robustness).
- Kiểm tra các trường bắt buộc và giá trị hợp lệ theo tài liệu kỹ thuật (mục 25-27).
- Báo lỗi rõ ràng (ConfigError) khi cấu hình thiếu trường hoặc giá trị sai.

Các loại cấu hình hỗ trợ:
    - cấu hình huấn luyện đầy đủ (seed, dataset, split, model, training,
      loss, evaluation, device) theo mục 25
    - cấu hình đánh giá suy giảm chất lượng (seed, degradation) theo mục 26
    - cấu hình huấn luyện robustness (seed, robustness) theo mục 27

Cách dùng:
    from src.config import load_config
    config = load_config("configs/base.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


TRAINING_SECTIONS = (
    "seed",
    "dataset",
    "split",
    "model",
    "training",
    "loss",
    "evaluation",
    "device",
)

DEGRADATION_SECTIONS = ("seed", "degradation")

ROBUSTNESS_SECTIONS = ("seed", "robustness")

VALID_DATASETS = ("celeba_spoof",)
VALID_MODELS = ("mobilenet_v2", "mobilenet_v3", "custom_cnn")
VALID_SPLIT_STRATEGIES = ("subject_disjoint", "random")
VALID_DEVICES = ("auto", "cpu", "cuda")
VALID_LOSSES = ("bce_with_logits",)
VALID_DEGRADATIONS = ("jpeg", "resize", "blur", "noise", "brightness")

_RANGE_KEYS = {
    "jpeg": ("quality_range",),
    "resize": ("scale_range",),
    "blur": ("sigma_range",),
    "noise": ("std_range",),
    "brightness": ("factor_range",),
}


def load_config(path: str | Path) -> dict:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML in '{path}': {exc}") from exc

    if config is None:
        raise ConfigError(f"Config file is empty: {path}")
    if not isinstance(config, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(config).__name__}")

    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Validate a parsed config dict. Raises ConfigError on invalid values."""
    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a mapping, got {type(config).__name__}")

    if "degradation" in config:
        _validate_required_sections(config, DEGRADATION_SECTIONS)
        _validate_degradation(config["degradation"])
        return

    if "robustness" in config:
        _validate_required_sections(config, ROBUSTNESS_SECTIONS)
        _validate_robustness(config["robustness"])
        return

    _validate_required_sections(config, TRAINING_SECTIONS)
    _validate_seed(config["seed"])
    _validate_dataset(config["dataset"])
    _validate_split(config["split"])
    _validate_model(config["model"])
    _validate_training(config["training"])
    _validate_loss(config["loss"])
    _validate_evaluation(config["evaluation"])
    _validate_device(config["device"])


def _validate_required_sections(config: dict, sections: tuple) -> None:
    missing = [name for name in sections if name not in config]
    if missing:
        raise ConfigError(
            f"Missing required top-level section(s): {', '.join(missing)}. "
            f"Expected: {', '.join(sections)}"
        )


def _require_mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a mapping, got {type(value).__name__}")
    return value


def _require_number(
    value: Any,
    name: str,
    *,
    is_int: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number, got {value!r}")
    if is_int and not isinstance(value, int):
        raise ConfigError(f"'{name}' must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"'{name}' must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"'{name}' must be <= {maximum}, got {value!r}")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise ConfigError(f"'{name}' must be > {exclusive_minimum}, got {value!r}")


def _require_one_of(value: Any, name: str, options: tuple) -> None:
    if value not in options:
        raise ConfigError(f"'{name}' must be one of {list(options)}, got {value!r}")


def _validate_seed(seed: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"'seed' must be a non-negative integer, got {seed!r}")


def _validate_dataset(dataset: Any) -> None:
    dataset = _require_mapping(dataset, "dataset")
    _require_one_of(dataset.get("name"), "dataset.name", VALID_DATASETS)
    root = dataset.get("root")
    if not isinstance(root, str) or not root.strip():
        raise ConfigError("'dataset.root' must be a non-empty string")


def _validate_split(split: Any) -> None:
    split = _require_mapping(split, "split")
    _require_one_of(split.get("strategy"), "split.strategy", VALID_SPLIT_STRATEGIES)


def _validate_model(model: Any) -> None:
    model = _require_mapping(model, "model")
    _require_one_of(model.get("name"), "model.name", VALID_MODELS)
    _require_number(
        model.get("image_size"),
        "model.image_size",
        is_int=True,
        minimum=1,
    )


def _validate_training(training: Any) -> None:
    training = _require_mapping(training, "training")
    _require_number(
        training.get("epochs"),
        "training.epochs",
        is_int=True,
        minimum=1,
    )
    _require_number(
        training.get("batch_size"),
        "training.batch_size",
        is_int=True,
        minimum=1,
    )
    _require_number(
        training.get("learning_rate"),
        "training.learning_rate",
        exclusive_minimum=0.0,
    )
    _require_number(
        training.get("weight_decay"),
        "training.weight_decay",
        minimum=0.0,
    )


def _validate_loss(loss: Any) -> None:
    loss = _require_mapping(loss, "loss")
    _require_one_of(loss.get("name"), "loss.name", VALID_LOSSES)
    if not isinstance(loss.get("use_pos_weight"), bool):
        raise ConfigError("'loss.use_pos_weight' must be a boolean")


def _validate_evaluation(evaluation: Any) -> None:
    evaluation = _require_mapping(evaluation, "evaluation")
    _require_number(
        evaluation.get("threshold"),
        "evaluation.threshold",
        minimum=0.0,
        maximum=1.0,
    )


def _validate_device(device: Any) -> None:
    device = _require_mapping(device, "device")
    _require_one_of(device.get("name"), "device.name", VALID_DEVICES)


def _validate_degradation(degradation: Any) -> None:
    degradation = _require_mapping(degradation, "degradation")
    name = degradation.get("name")
    _require_one_of(name, "degradation.name", VALID_DEGRADATIONS)

    if name == "jpeg":
        _require_number(
            degradation.get("quality"),
            "degradation.quality",
            is_int=True,
            minimum=1,
            maximum=100,
        )
    elif name == "resize":
        _require_number(
            degradation.get("scale"),
            "degradation.scale",
            exclusive_minimum=0.0,
            maximum=1.0,
        )
    elif name == "blur":
        kernel_size = degradation.get("kernel_size")
        _require_number(
            kernel_size,
            "degradation.kernel_size",
            is_int=True,
            minimum=1,
        )
        if kernel_size % 2 == 0:
            raise ConfigError(
                f"'degradation.kernel_size' must be odd, got {kernel_size!r}"
            )
        _require_number(
            degradation.get("sigma"),
            "degradation.sigma",
            exclusive_minimum=0.0,
        )
    elif name == "noise":
        _require_number(
            degradation.get("std"),
            "degradation.std",
            minimum=0.0,
        )
    elif name == "brightness":
        _require_number(
            degradation.get("factor"),
            "degradation.factor",
            exclusive_minimum=0.0,
        )


def _validate_robustness(robustness: Any) -> None:
    robustness = _require_mapping(robustness, "robustness")
    if not isinstance(robustness.get("enabled"), bool):
        raise ConfigError("'robustness.enabled' must be a boolean")

    augmentations = _require_mapping(robustness.get("augmentations"), "robustness.augmentations")
    for name, spec in augmentations.items():
        _require_one_of(name, f"robustness.augmentations.{name}", VALID_DEGRADATIONS)
        spec = _require_mapping(spec, f"robustness.augmentations.{name}")
        if not isinstance(spec.get("enabled"), bool):
            raise ConfigError(f"'robustness.augmentations.{name}.enabled' must be a boolean")
        if spec["enabled"]:
            _validate_augmentation_range(name, spec)


def _validate_augmentation_range(name: str, spec: dict) -> None:
    for key in _RANGE_KEYS[name]:
        value = spec.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ConfigError(
                f"'robustness.augmentations.{name}.{key}' must be a list of two numbers"
            )
        low, high = value
        if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise ConfigError(
                f"'robustness.augmentations.{name}.{key}' must be a list of two numbers"
            )
        if low < 0 or high < 0 or low > high:
            raise ConfigError(
                f"'robustness.augmentations.{name}.{key}' must be [low, high] with 0 <= low <= high"
            )
