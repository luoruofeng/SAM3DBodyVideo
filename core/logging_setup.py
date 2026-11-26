"""日志配置：控制台 + 文件双输出，基于 loguru。"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(
    log_dir: str | Path = "logs",
    level: str = "INFO",
    file_name: str = "sam3dvideo.log",
    rotation: str = "10 MB",
    retention: str = "14 days",
) -> Path:
    """初始化全局日志，返回日志文件路径。重复调用不会重复挂载 handler。"""
    global _CONFIGURED
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / file_name

    if _CONFIGURED:
        return log_file

    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, colorize=True)
    logger.add(
        log_file,
        level=level,
        format=fmt,
        encoding="utf-8",
        rotation=rotation,
        retention=retention,
        enqueue=True,
    )
    _CONFIGURED = True
    return log_file


def setup_logging_from_config(settings) -> Path:
    """按 config.yaml 中的 logging 段落初始化日志。"""
    cfg = settings.logging
    return setup_logging(
        log_dir=cfg.resolve_dir(),
        level=cfg.level,
        file_name=cfg.file_name,
        rotation=cfg.rotation,
        retention=cfg.retention,
    )
