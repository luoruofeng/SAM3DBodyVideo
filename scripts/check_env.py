"""环境自检：Python / torch / CUDA / ffmpeg / 依赖 / 权重完整性。

用法：
    python scripts/check_env.py                # 只检查
    python scripts/check_env.py --write-manifest  # 把现有权重的 sha256 写入清单
退出码 0 = 全部关键项通过，1 = 有关键项缺失。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config  # noqa: E402

REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "yaml",
    "loguru",
    "pytorch_lightning",
    "scipy",
    "trimesh",
]
OPTIONAL_MODULES = ["pyrender", "timm", "einops", "pyrootutils", "hydra", "sam3"]


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def sha256sum(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def check_python() -> bool:
    version = platform.python_version()
    supported = sys.version_info[:2] in {(3, 11), (3, 12), (3, 13)}
    (ok if supported else warn)(
        f"Python {version} ({sys.executable})"
        + ("" if supported else " —— 推荐 3.11 / 3.12 / 3.13")
    )
    return True


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        fail("torch 未安装")
        return False

    ok(f"torch {torch.__version__} (cuda build: {torch.version.cuda})")
    if not torch.cuda.is_available():
        warn("CUDA 不可用，推理将回退到 CPU（极慢）")
        return True

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    ok(f"GPU: {props.name} | {vram_gb:.1f} GB | sm_{props.major}{props.minor} | {torch.cuda.device_count()} 卡")
    if vram_gb < 12:
        warn("显存 < 12 GB，长视频需降低批大小/分辨率")
    return True


def check_ffmpeg() -> bool:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        fail(f"未找到 {', '.join(missing)}，视频探测与拆帧会失败")
        return False
    exe = shutil.which("ffmpeg")
    try:
        out = subprocess.run(
            [exe, "-version"], capture_output=True, text=True, timeout=20
        ).stdout.splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        warn(f"ffmpeg 存在但无法执行: {exc}")
        return False
    ok(out)
    return True


def check_modules() -> bool:
    all_ok = True
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            fail(f"缺少依赖 {name}: {exc}")
            all_ok = False
        else:
            version = getattr(module, "__version__", "?")
            ok(f"{name} {version}")
    for name in OPTIONAL_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            warn(f"可选依赖 {name} 未安装")
        else:
            ok(f"{name} {getattr(module, '__version__', '?')} (可选)")
    return all_ok


def check_third_party(settings) -> bool:
    repo = settings.paths.resolve("third_party")
    entry = repo / "sam_3d_body" / "__init__.py"
    if entry.exists():
        ok(f"官方代码: {repo}")
        return True
    fail(f"未找到官方 SAM 3D Body 代码：{repo}（运行 scripts/fetch_third_party.ps1）")
    return False


def check_weights(settings, write_manifest: bool) -> bool:
    manifest_path = settings.paths.resolve("checksum_manifest")
    manifest: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})

    targets = {
        "sam3d_checkpoint": settings.paths.resolve("sam3d_checkpoint"),
        "sam3d_config": settings.paths.resolve("sam3d_config"),
        "mhr_model": settings.paths.resolve("mhr_model"),
    }
    new_manifest: dict[str, str] = {}
    all_ok = True
    for key, path in targets.items():
        if not path.exists():
            fail(f"{key} 缺失: {path}")
            all_ok = False
            continue
        size_mb = path.stat().st_size / 1024**2
        digest = sha256sum(path)
        new_manifest[key] = digest
        expected = manifest.get(key)
        if expected is None:
            warn(f"{key} 存在但清单里没有记录哈希: {path} ({size_mb:.1f} MB)")
        elif expected != digest:
            fail(f"{key} 哈希不匹配: {path}")
            all_ok = False
        else:
            ok(f"{key} 校验通过 ({size_mb:.1f} MB) sha256={digest[:16]}...")

    if not all_ok:
        warn("权重不完整：阶段 2/3 的推理无法运行")
    if write_manifest and new_manifest:
        payload = {"files": new_manifest}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ok(f"已写入哈希清单: {manifest_path}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM3DBodyVideo 环境自检")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="把已存在权重的 sha256 写入 checkpoints/manifest.json",
    )
    args = parser.parse_args()

    settings = load_config()
    print("=" * 72)
    print("SAM3DBodyVideo 环境自检")
    print(f"项目根目录: {PROJECT_ROOT}")
    print("=" * 72)

    results = {
        "python": check_python(),
        "torch": check_torch(),
        "ffmpeg": check_ffmpeg(),
        "deps": check_modules(),
        "third_party": check_third_party(settings),
        "weights": check_weights(settings, args.write_manifest),
    }

    print("=" * 72)
    failed = [name for name, passed in results.items() if not passed]
    if failed:
        print(f"结论: 未通过 -> {', '.join(failed)}")
        return 1
    print("结论: 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
