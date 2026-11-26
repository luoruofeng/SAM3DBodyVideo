"""项目配置加载：把 config.yaml 解析为强类型 dataclass。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_FILE = PROJECT_ROOT / ".env"

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def load_dotenv(path: Path = ENV_FILE) -> None:
    """把 .env 中的 KEY=VALUE 注入环境变量（不覆盖已有的系统变量）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))



@dataclass
class RuntimeSettings:
    device: str = "auto"
    precision: str = "fp32"
    batch_size: int = 1
    vram_headroom: float = 0.85


@dataclass
class PathSettings:
    third_party: str = "third_party/sam-3d-body-main"
    checkpoints_dir: str = "checkpoints"
    sam3d_checkpoint: str = "checkpoints/sam-3d-body-dinov3/model.ckpt"
    sam3d_config: str = "checkpoints/sam-3d-body-dinov3/model_config.yaml"
    mhr_model: str = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    detector_vitdet_dir: str = "checkpoints/vitdet"
    fov_moge2_dir: str = "checkpoints/moge2"
    checksum_manifest: str = "checkpoints/manifest.json"

    def resolve(self, name: str) -> Path:
        """把配置里的相对路径解析为绝对路径。"""
        value = getattr(self, name)
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class VideoSettings:
    target_fps: int = 30
    start_time: float | None = None
    end_time: float | None = None
    resize_long_side: int | None = None


@dataclass
class InferenceSettings:
    bbox_threshold: float = 0.8
    detector: str = "vitdet"
    segmentor: str = "sam2"
    fov: str = "moge2"


@dataclass
class HFSettings:
    token_env: str = "HF_TOKEN"
    repo_id: str = "facebook/sam-3d-body-dinov3"
    endpoint: str = ""
    local_dir: str = "checkpoints/sam-3d-body-dinov3"

    def resolve_token(self) -> str | None:
        return os.environ.get(self.token_env) or None

    def resolve_dir(self) -> Path:
        path = Path(self.local_dir)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class OutputSettings:
    output_root: str = "outputs"
    keep_intermediate: bool = True

    def resolve_root(self) -> Path:
        path = Path(self.output_root)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class SmoothingSettings:
    filter: str = "one_euro"
    one_euro: dict[str, float] = field(
        default_factory=lambda: {
            "min_cutoff": 1.0,
            "beta": 0.05,
            "root_min_cutoff": 0.5,
            "root_beta": 0.02,
        }
    )


@dataclass
class LoggingSettings:
    level: str = "INFO"
    dir: str = "logs"
    file_name: str = "sam3dvideo.log"
    rotation: str = "10 MB"
    retention: str = "14 days"

    def resolve_dir(self) -> Path:
        path = Path(self.dir)
        return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class Settings:
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    hf: HFSettings = field(default_factory=HFSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    smoothing: SmoothingSettings = field(default_factory=SmoothingSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    source_path: Path | None = None


def _build(cls: type, data: dict[str, Any] | None):
    """用 dict 覆盖 dataclass 默认值，忽略未知字段。"""
    data = data or {}
    allowed = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in allowed})


def load_config(path: str | Path | None = None) -> Settings:
    """读取 YAML 配置并做基础校验。"""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    load_dotenv()

    settings = Settings(
        runtime=_build(RuntimeSettings, raw.get("runtime")),
        paths=_build(PathSettings, raw.get("paths")),
        video=_build(VideoSettings, raw.get("video")),
        inference=_build(InferenceSettings, raw.get("inference")),
        hf=_build(HFSettings, raw.get("hf")),
        output=_build(OutputSettings, raw.get("output")),
        smoothing=_build(SmoothingSettings, raw.get("smoothing")),
        logging=_build(LoggingSettings, raw.get("logging")),
        source_path=config_path,
    )
    validate_config(settings)
    return settings


def validate_config(settings: Settings) -> None:
    """校验关键字段取值，尽早暴露配置错误。"""
    if settings.runtime.device not in VALID_DEVICES:
        raise ValueError(
            f"runtime.device 必须是 {VALID_DEVICES} 之一，当前为 {settings.runtime.device!r}"
        )
    if settings.video.target_fps <= 0:
        raise ValueError("video.target_fps 必须为正数")
    start, end = settings.video.start_time, settings.video.end_time
    if start is not None and end is not None and start >= end:
        raise ValueError("video.start_time 必须小于 video.end_time")


def resolve_device(requested: str) -> str:
    """把 auto 解析为当前机器上真实可用的设备。"""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
