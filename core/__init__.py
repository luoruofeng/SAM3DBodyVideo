"""核心层：配置、日志、推理封装、时间平滑等纯逻辑代码（不依赖 GUI）。"""

from .config import (
    PROJECT_ROOT,
    Settings,
    load_config,
    resolve_device,
    validate_config,
)
from .logging_setup import setup_logging, setup_logging_from_config

__all__ = [
    "PROJECT_ROOT",
    "Settings",
    "load_config",
    "resolve_device",
    "validate_config",
    "setup_logging",
    "setup_logging_from_config",
]
