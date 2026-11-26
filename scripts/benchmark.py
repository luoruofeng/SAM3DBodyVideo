"""性能基准：冷/热启动单帧耗时、批量吞吐、显存峰值。

用法：
    python scripts/benchmark.py                    # 默认 300 帧，写 docs/benchmark.md
    python scripts/benchmark.py --frames 50 --no-detector

需要先准备权重（checkpoints/ 下），否则脚本会给出提示并退出。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config, resolve_device  # noqa: E402


def gpu_summary() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "未检测到 NVIDIA GPU"


def setup_estimator(settings, use_detector: bool):
    repo = settings.paths.resolve("third_party")
    if not (repo / "sam_3d_body").exists():
        sys.exit(f"未找到官方代码目录: {repo}，请先运行 scripts/fetch_third_party.ps1")
    sys.path.insert(0, str(repo))

    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    ckpt = settings.paths.resolve("sam3d_checkpoint")
    mhr = settings.paths.resolve("mhr_model")
    missing = [str(p) for p in (ckpt, mhr) if not p.exists()]
    if missing:
        sys.exit(
            "缺少权重文件:\n  " + "\n  ".join(missing) + "\n"
            "请按 third_party/sam-3d-body-main/INSTALL.md 申请并下载 HuggingFace 门控权重。"
        )

    device = resolve_device(settings.runtime.device)
    t0 = time.perf_counter()
    model, model_cfg = load_sam_3d_body(str(ckpt), device=device, mhr_path=str(mhr))
    load_time = time.perf_counter() - t0

    detector = None
    if use_detector:
        from tools.build_detector import HumanDetector

        detector = HumanDetector(
            name=settings.inference.detector,
            device=device,
            path=str(settings.paths.resolve("detector_vitdet_dir")),
        )

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model, model_cfg=model_cfg, human_detector=detector
    )
    return estimator, device, load_time


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM 3D Body 单帧性能基准")
    parser.add_argument("--frames", type=int, default=300, help="吞吐测试的帧数")
    parser.add_argument("--warmup", type=int, default=3, help="预热帧数")
    parser.add_argument("--no-detector", action="store_true", help="跳过人体检测，直接整图推理")
    parser.add_argument("--image", default="", help="测试图片路径（默认用官方示例图）")
    parser.add_argument("--out", default="docs/benchmark.md", help="报告输出路径")
    args = parser.parse_args()

    settings = load_config()
    import cv2
    import torch

    image_path = Path(args.image) if args.image else (
        settings.paths.resolve("third_party") / "notebook" / "images" / "dancing.jpg"
    )
    if not image_path.exists():
        sys.exit(f"测试图片不存在: {image_path}")
    img = cv2.imread(str(image_path))

    estimator, device, load_time = setup_estimator(settings, not args.no_detector)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 冷启动：第一帧包含 cudnn/内核编译开销
    t0 = time.perf_counter()
    outputs = estimator.process_one_image(img, bbox_thr=settings.inference.bbox_threshold)
    cold = time.perf_counter() - t0
    if not outputs:
        sys.exit("未检测到人体：请换一张全身图或去掉 --no-detector")

    for _ in range(args.warmup):
        estimator.process_one_image(img, bbox_thr=settings.inference.bbox_threshold)

    samples: list[float] = []
    t_batch = time.perf_counter()
    for _ in range(args.frames):
        t0 = time.perf_counter()
        estimator.process_one_image(img, bbox_thr=settings.inference.bbox_threshold)
        samples.append(time.perf_counter() - t0)
    batch_time = time.perf_counter() - t_batch

    vram_peak = vram_reserved = 0.0
    if device == "cuda":
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        vram_reserved = torch.cuda.max_memory_reserved() / 1024**3

    # 保存一份 mesh，证明单图链路可产出结果
    out_dir = PROJECT_ROOT / "outputs" / "_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    person = outputs[0]
    np.savez_compressed(
        out_dir / "single_frame_mesh.npz",
        vertices=person["pred_vertices"],
        faces=estimator.faces,
        keypoints_3d=person["pred_keypoints_3d"],
        keypoints_2d=person["pred_keypoints_2d"],
        bbox=person["bbox"],
    )

    stats = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "machine": {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "gpu": gpu_summary(),
        },
        "model_load_s": round(load_time, 3),
        "cold_frame_s": round(cold, 3),
        "warm_mean_s": round(statistics.mean(samples), 3),
        "warm_median_s": round(statistics.median(samples), 3),
        "warm_p95_s": round(statistics.quantiles(samples, n=20)[-1], 3) if len(samples) >= 20 else None,
        "warm_min_s": round(min(samples), 3),
        "frames": args.frames,
        "batch_total_s": round(batch_time, 2),
        "fps": round(args.frames / batch_time, 2),
        "projected_300_frames_s": round(statistics.mean(samples) * 300, 1),
        "vram_peak_allocated_gb": round(vram_peak, 2),
        "vram_peak_reserved_gb": round(vram_reserved, 2),
        "vertices": int(person["pred_vertices"].shape[0]),
        "faces": int(estimator.faces.shape[0]),
    }
    (out_dir / "benchmark.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# 性能基准（阶段 0.3）",
        "",
        f"- 采集时间：{stats['timestamp']}",
        f"- 机器：{stats['machine']['os']} / {stats['machine']['cpu']}",
        f"- Python {stats['machine']['python']} · torch {stats['machine']['torch']} · device={stats['machine']['device']}",
        f"- GPU：{stats['machine']['gpu']}",
        f"- 测试图片：`{image_path.relative_to(PROJECT_ROOT)}`（{img.shape[1]}x{img.shape[0]}）",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 模型加载耗时（含权重读取） | {stats['model_load_s']} s |",
        f"| 冷启动单帧 | {stats['cold_frame_s']} s |",
        f"| 热启动单帧（均值 / 中位 / P95） | {stats['warm_mean_s']} / {stats['warm_median_s']} / {stats['warm_p95_s']} s |",
        f"| 连续 {stats['frames']} 帧总耗时 | {stats['batch_total_s']} s |",
        f"| 吞吐 | {stats['fps']} FPS |",
        f"| 300 帧（10 s @30fps）预计耗时 | {stats['projected_300_frames_s']} s |",
        f"| 显存峰值（allocated / reserved） | {stats['vram_peak_allocated_gb']} / {stats['vram_peak_reserved_gb']} GB |",
        f"| 单帧 mesh | {stats['vertices']} 顶点 / {stats['faces']} 面 |",
        "",
        "## 结论与批处理建议",
        "",
        f"- 单帧约 {stats['warm_mean_s']} s，10 秒 30fps 视频需 {stats['projected_300_frames_s'] / 60:.1f} 分钟。",
        f"- 显存峰值 {stats['vram_peak_reserved_gb']} GB，可并行帧批大小建议先取 1，逐步上调直至触及 "
        f"{int(settings.runtime.vram_headroom * 100)}% 上限。",
        "- 长视频必须启用断点续跑与抽帧区间，避免整段无谓推理。",
    ]
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
