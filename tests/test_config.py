"""配置层单元测试。"""

from pathlib import Path

import pytest

from core.config import PROJECT_ROOT, load_config, resolve_device, validate_config


def test_load_default_config():
    settings = load_config()
    assert settings.source_path == PROJECT_ROOT / "config.yaml"
    assert settings.video.target_fps > 0
    assert settings.smoothing.filter in {"one_euro", "moving_average", "savgol", "none"}


def test_paths_are_absolute():
    settings = load_config()
    assert settings.paths.resolve("sam3d_checkpoint").is_absolute()
    assert settings.paths.resolve("mhr_model").is_absolute()
    assert settings.logging.resolve_dir().is_absolute()
    assert settings.output.resolve_root().is_absolute()


def test_invalid_device_rejected(tmp_path: Path):
    custom = tmp_path / "config.yaml"
    custom.write_text("runtime:\n  device: tpu\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime.device"):
        load_config(custom)


def test_invalid_time_range_rejected(tmp_path: Path):
    custom = tmp_path / "config.yaml"
    custom.write_text(
        "video:\n  start_time: 10\n  end_time: 5\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="start_time"):
        load_config(custom)


def test_validate_config_accepts_defaults():
    validate_config(load_config())


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}
