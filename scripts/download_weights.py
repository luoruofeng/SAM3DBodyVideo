"""下载 SAM 3D Body 门控权重（HuggingFace）。

用法：python scripts/download_weights.py [--dry-run]

Token 从 .env / 环境变量（默认 HF_TOKEN）读取；国内网络可用 config.yaml 中的 hf.endpoint 镜像。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config  # noqa: E402

# 推理只需这些文件，避免整仓快照
ALLOW_PATTERNS = ["model.ckpt", "model_config.yaml", "assets/*", "LICENSE*", "*.md"]


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 SAM 3D Body 权重")
    parser.add_argument("--dry-run", action="store_true", help="只检查连通性与权限")
    args = parser.parse_args()

    settings = load_config()
    hf = settings.hf
    token = hf.resolve_token()
    local_dir = hf.resolve_dir()

    if not token:
        print(f"未找到 Token：请在 .env 中设置 {hf.token_env}")
        return 2

    if hf.endpoint:
        os.environ["HF_ENDPOINT"] = hf.endpoint
    os.environ["HF_TOKEN"] = token

    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    print(f"仓库: {hf.repo_id}")
    print(f"端点: {hf.endpoint or 'https://huggingface.co'}")
    print(f"目标: {local_dir}")

    try:
        cfg = hf_hub_download(
            repo_id=hf.repo_id,
            filename="model_config.yaml",
            token=token,
            local_dir=str(local_dir),
        )
        print(f"连通性/权限检查通过: {cfg}")
    except GatedRepoError:
        print("门控拒绝：该账号尚未获得仓库访问权限，请先在 HuggingFace 页面申请。")
        return 3
    except HfHubHTTPError as exc:
        print(f"请求失败（可能是网络或权限）: {exc}")
        return 4

    if args.dry_run:
        return 0

    snapshot_download(
        repo_id=hf.repo_id,
        token=token,
        local_dir=str(local_dir),
        allow_patterns=ALLOW_PATTERNS,
        max_workers=4,
    )
    print("下载完成")
    print("下一步: python scripts/check_env.py --write-manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
