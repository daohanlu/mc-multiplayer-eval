#!/usr/bin/env python3
"""
Compute visual similarity metrics (LPIPS, CLIP, DINOv2, DINOv3) between
alpha and bravo generated frames for turnToLookEval / turnToLookOppositeEval.

Reuses the same handler keyframe logic and frame extraction as run_eval.py
to ensure identical frames are compared.

Usage:
  python run_consistency_metrics.py [options]

  # Quick test with 2 episodes on one model
  python run_consistency_metrics.py --limit 2 --models flagship

  # Full run on all models with 2 GPUs
  python run_consistency_metrics.py --num-gpus 2
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import extract_query_frames, find_mc_video_pairs, identify_handler
from vlm_utils import find_generated_video_subdir

EVAL_TYPES = ["turnToLookEval", "turnToLookOppositeEval"]
DATASET_BASE_DEFAULT = Path("mc_multiplayer_v2_eval_new_sneak_combined")
GENERATIONS_DEFAULT = Path("mc_multiplayer_v2_generations")
RESULTS_DIR_DEFAULT = Path("results_consistency_metrics")

DEFAULT_VIDEO_PAIR_LIMIT = 32


def png_bytes_to_rgb(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes to RGB numpy array (H, W, 3), uint8."""
    import cv2

    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_metric_models(device: str) -> dict:
    """Load all 4 metric models on the given device."""
    import torch
    import lpips as lpips_lib
    import open_clip
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    models: dict = {}

    print(f"  Loading LPIPS (alex) ...", flush=True)
    models["lpips"] = lpips_lib.LPIPS(net="alex").to(device).eval()

    print(f"  Loading CLIP ViT-B-32 ...", flush=True)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    models["clip_model"] = clip_model.to(device).eval()
    models["clip_preprocess"] = clip_preprocess

    print(f"  Loading DINOv2 (vitb14) ...", flush=True)
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    models["dinov2"] = dinov2.to(device).eval()

    print(f"  Loading DINOv3 (vitb16 via timm) ...", flush=True)
    dinov3 = timm.create_model("vit_base_patch16_dinov3", pretrained=True, num_classes=0)
    models["dinov3"] = dinov3.to(device).eval()
    dinov3_cfg = resolve_data_config(dinov3.pretrained_cfg)
    models["dinov3_transform"] = create_transform(**dinov3_cfg)

    print(f"  All models loaded on {device}", flush=True)
    return models


def compute_metrics(
    alpha_rgb: np.ndarray, bravo_rgb: np.ndarray, models: dict, device: str
) -> Dict[str, float]:
    """Compute LPIPS, CLIP, DINOv2, DINOv3 between two RGB images."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image

    alpha_pil = Image.fromarray(alpha_rgb)
    bravo_pil = Image.fromarray(bravo_rgb)

    results: Dict[str, float] = {}

    with torch.no_grad():
        # --- LPIPS (lower = more similar) ---
        lpips_tf = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        a_lp = lpips_tf(alpha_pil).unsqueeze(0).to(device)
        b_lp = lpips_tf(bravo_pil).unsqueeze(0).to(device)
        results["lpips"] = models["lpips"](a_lp, b_lp).item()

        # --- CLIP cosine similarity (higher = more similar) ---
        clip_pp = models["clip_preprocess"]
        a_cl = clip_pp(alpha_pil).unsqueeze(0).to(device)
        b_cl = clip_pp(bravo_pil).unsqueeze(0).to(device)
        a_feat = F.normalize(models["clip_model"].encode_image(a_cl), dim=-1)
        b_feat = F.normalize(models["clip_model"].encode_image(b_cl), dim=-1)
        results["clip_cosine_sim"] = (a_feat @ b_feat.T).item()

        # --- DINOv2 cosine similarity (higher = more similar) ---
        dino_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        a_d2 = dino_tf(alpha_pil).unsqueeze(0).to(device)
        b_d2 = dino_tf(bravo_pil).unsqueeze(0).to(device)
        a_feat = F.normalize(models["dinov2"](a_d2), dim=-1)
        b_feat = F.normalize(models["dinov2"](b_d2), dim=-1)
        results["dinov2_cosine_sim"] = (a_feat @ b_feat.T).item()

        # --- DINOv3 cosine similarity (higher = more similar) ---
        d3_tf = models["dinov3_transform"]
        a_d3 = d3_tf(alpha_pil).unsqueeze(0).to(device)
        b_d3 = d3_tf(bravo_pil).unsqueeze(0).to(device)
        a_feat = F.normalize(models["dinov3"](a_d3), dim=-1)
        b_feat = F.normalize(models["dinov3"](b_d3), dim=-1)
        results["dinov3_cosine_sim"] = (a_feat @ b_feat.T).item()

    return results


@dataclass
class WorkItem:
    """A single (model_name, eval_type) evaluation unit."""

    model_name: str
    eval_type: str


def extract_frame_pairs(
    eval_type: str,
    model_name: str,
    dataset_base: Path,
    generations_dir: Path,
    limit: Optional[int],
) -> List[dict]:
    """Extract alpha/bravo frame pairs for one (model, eval_type).

    Returns list of dicts with keys: episode, instance, alpha_rgb, bravo_rgb.
    """
    folder = dataset_base / eval_type / "test"
    video_pairs = find_mc_video_pairs(folder)

    if limit:
        video_pairs = video_pairs[:limit]
    elif len(video_pairs) > DEFAULT_VIDEO_PAIR_LIMIT:
        video_pairs = video_pairs[:DEFAULT_VIDEO_PAIR_LIMIT]

    handler = identify_handler(eval_type)

    gen_path = generations_dir / model_name
    gen_subdir = find_generated_video_subdir(gen_path, eval_type)

    gen_video_count = len(list(gen_subdir.glob("video_*_side_by_side.mp4")))
    if gen_video_count < len(video_pairs):
        video_pairs = video_pairs[:gen_video_count]

    pairs_data: List[dict] = []
    current_video_id = -1

    for pair in video_pairs:
        current_video_id += 1
        try:
            queries = handler.extract_keyframes(pair)
        except (ValueError, Exception) as e:
            print(f"    Skipping episode {pair.episode_num}: {e}")
            continue
        if not queries:
            continue

        query = queries[0]
        frame1_idx = query.metadata["frame1"]

        try:
            frames = extract_query_frames(
                query=query,
                generated_subdir=gen_subdir,
                current_video_id=current_video_id,
                frame1_idx=frame1_idx,
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"    Skipping episode {pair.episode_num}: {e}")
            continue

        pairs_data.append(
            {
                "episode": pair.episode_num,
                "instance": pair.instance_num,
                "alpha_rgb": png_bytes_to_rgb(frames["alpha_frame"]),
                "bravo_rgb": png_bytes_to_rgb(frames["bravo_frame"]),
            }
        )

    return pairs_data


METRIC_NAMES = ["lpips", "clip_cosine_sim", "dinov2_cosine_sim", "dinov3_cosine_sim"]


def process_work_items(
    gpu_id: int,
    work_items: List[WorkItem],
    dataset_base: Path,
    generations_dir: Path,
    results_dir: Path,
    limit: Optional[int],
) -> List[dict]:
    """Worker: load models on gpu_id, process assigned (model, eval_type) items."""
    import torch

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f"\n[GPU {gpu_id}] Loading metric models ...", flush=True)
    models = load_metric_models(device)

    summaries: List[dict] = []

    for wi in work_items:
        t0 = time.time()
        print(
            f"\n[GPU {gpu_id}] Processing {wi.model_name} / {wi.eval_type} ...",
            flush=True,
        )

        pairs_data = extract_frame_pairs(
            wi.eval_type, wi.model_name, dataset_base, generations_dir, limit
        )
        print(f"  Extracted {len(pairs_data)} frame pairs", flush=True)

        per_episode: List[dict] = []
        for i, pd in enumerate(pairs_data):
            metrics = compute_metrics(pd["alpha_rgb"], pd["bravo_rgb"], models, device)
            per_episode.append(
                {
                    "episode": pd["episode"],
                    "instance": pd["instance"],
                    **metrics,
                }
            )
            if (i + 1) % 8 == 0 or i == len(pairs_data) - 1:
                print(
                    f"  [{i+1}/{len(pairs_data)}] lpips={metrics['lpips']:.4f} "
                    f"clip={metrics['clip_cosine_sim']:.4f} "
                    f"dinov2={metrics['dinov2_cosine_sim']:.4f} "
                    f"dinov3={metrics['dinov3_cosine_sim']:.4f}",
                    flush=True,
                )

        agg_stats: Dict[str, dict] = {}
        for m in METRIC_NAMES:
            values = [e[m] for e in per_episode]
            agg_stats[m] = {
                "mean": statistics.mean(values) if values else None,
                "std": statistics.pstdev(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }

        output = {
            "model_name": wi.model_name,
            "eval_type": wi.eval_type,
            "num_episodes": len(per_episode),
            "stats": agg_stats,
            "per_episode": per_episode,
        }

        out_dir = results_dir / f"{wi.model_name}_{wi.eval_type}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        elapsed = time.time() - t0
        print(
            f"  Done in {elapsed:.1f}s. Saved to {out_path}",
            flush=True,
        )

        summaries.append(
            {
                "model_name": wi.model_name,
                "eval_type": wi.eval_type,
                "num_episodes": len(per_episode),
                **{f"{m}_mean": agg_stats[m]["mean"] for m in METRIC_NAMES},
                **{f"{m}_std": agg_stats[m]["std"] for m in METRIC_NAMES},
            }
        )

    return summaries


def _gpu_worker(args):
    """Top-level function for multiprocessing (must be picklable)."""
    gpu_id, work_items_dicts, dataset_base_str, generations_dir_str, results_dir_str, limit = args
    work_items = [WorkItem(**d) for d in work_items_dicts]
    return process_work_items(
        gpu_id,
        work_items,
        Path(dataset_base_str),
        Path(generations_dir_str),
        Path(results_dir_str),
        limit,
    )


def write_summary_markdown(summaries: List[dict], results_dir: Path) -> Path:
    """Write a summary markdown table from all collected summaries."""
    md_path = results_dir / "consistency_metrics_summary.md"

    lines = ["# Consistency Metrics Summary", ""]
    lines.append(
        "Visual similarity between alpha and bravo generated frames "
        "(turnToLookEval expects similar, turnToLookOppositeEval expects different)."
    )
    lines.append("")

    for eval_type in EVAL_TYPES:
        et_rows = [s for s in summaries if s["eval_type"] == eval_type]
        if not et_rows:
            continue
        et_rows.sort(key=lambda r: r["model_name"])

        lines.append(f"## {eval_type}")
        lines.append("")
        lines.append(
            "| Model | LPIPS (mean +/- std) | CLIP cos sim | DINOv2 cos sim | DINOv3 cos sim |"
        )
        lines.append("|---|---|---|---|---|")

        for r in et_rows:
            def fmt(m):
                mean = r.get(f"{m}_mean")
                std = r.get(f"{m}_std")
                if mean is None:
                    return "N/A"
                return f"{mean:.4f} +/- {std:.4f}"

            lines.append(
                f"| {r['model_name']} "
                f"| {fmt('lpips')} "
                f"| {fmt('clip_cosine_sim')} "
                f"| {fmt('dinov2_cosine_sim')} "
                f"| {fmt('dinov3_cosine_sim')} |"
            )
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return md_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute visual similarity metrics for consistency evaluation",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_BASE_DEFAULT,
        help=f"Path to GT dataset (default: {DATASET_BASE_DEFAULT})",
    )
    parser.add_argument(
        "--generations",
        type=Path,
        default=GENERATIONS_DEFAULT,
        help=f"Path to generations directory (default: {GENERATIONS_DEFAULT})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR_DEFAULT,
        help=f"Output directory for results (default: {RESULTS_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to evaluate (default: all subdirs in --generations)",
    )
    parser.add_argument(
        "--eval-types",
        nargs="+",
        default=EVAL_TYPES,
        choices=EVAL_TYPES,
        help="Eval types to run (default: both)",
    )
    parser.add_argument("--limit", type=int, help="Limit episodes per eval")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect)",
    )

    args = parser.parse_args()

    import torch

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs detected. Set --num-gpus or check CUDA installation.")
        return 1

    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        return 1
    if not args.generations.exists():
        print(f"Error: Generations directory not found: {args.generations}")
        return 1

    if args.models:
        model_names = args.models
    else:
        model_names = sorted(
            d.name
            for d in args.generations.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    print(f"Models: {model_names}")
    print(f"Eval types: {args.eval_types}")
    print(f"GPUs: {num_gpus}")
    print(f"Results dir: {args.results_dir}")
    if args.limit:
        print(f"Episode limit: {args.limit}")

    work_items: List[WorkItem] = []
    for model_name in model_names:
        for eval_type in args.eval_types:
            work_items.append(WorkItem(model_name=model_name, eval_type=eval_type))

    print(f"\nTotal work items: {len(work_items)}")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    if num_gpus == 1:
        all_summaries = process_work_items(
            gpu_id=0,
            work_items=work_items,
            dataset_base=args.dataset,
            generations_dir=args.generations,
            results_dir=args.results_dir,
            limit=args.limit,
        )
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")

        chunks: List[List[dict]] = [[] for _ in range(num_gpus)]
        for i, wi in enumerate(work_items):
            gpu = i % num_gpus
            chunks[gpu].append({"model_name": wi.model_name, "eval_type": wi.eval_type})

        pool_args = [
            (
                gpu_id,
                chunks[gpu_id],
                str(args.dataset),
                str(args.generations),
                str(args.results_dir),
                args.limit,
            )
            for gpu_id in range(num_gpus)
        ]

        with ctx.Pool(processes=num_gpus) as pool:
            results_per_gpu = pool.map(_gpu_worker, pool_args)

        all_summaries = []
        for gpu_summaries in results_per_gpu:
            all_summaries.extend(gpu_summaries)

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"ALL DONE in {elapsed:.1f}s")
    print(f"{'='*80}")

    md_path = write_summary_markdown(all_summaries, args.results_dir)
    print(f"\nSummary written to: {md_path}")

    print("\n--- Quick Summary ---")
    for s in sorted(all_summaries, key=lambda x: (x["eval_type"], x["model_name"])):
        print(
            f"  {s['eval_type']:30s} {s['model_name']:25s} "
            f"LPIPS={s['lpips_mean']:.4f}  CLIP={s['clip_cosine_sim_mean']:.4f}  "
            f"DINOv2={s['dinov2_cosine_sim_mean']:.4f}  DINOv3={s['dinov3_cosine_sim_mean']:.4f}"
        )

    return 0


if __name__ == "__main__":
    exit(main())
