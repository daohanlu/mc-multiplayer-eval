#!/usr/bin/env python3
"""Build the self-contained human-eval bundle.

Produces two annotation tasks:

**Consistency** (task 1) — the *exact* screenshot pairs that the VLM saw for
``turnToLookEval`` and ``turnToLookOppositeEval`` under the STRICT late-episode
toggle, for ``flagship`` and ``causvid_regression``. Frames are produced by
calling ``run_eval.extract_query_frames`` — the same "single source of truth"
helper the paper run used — so the PNGs are pixel-identical to the VLM inputs.
32 episodes x 2 timestamps (original + late-horizon) x 2 evals x 2 models
= 256 comparisons.

**Artifacts** (task 2) — 63 supplementary clips under
``Model Generations on Eval/``: 7 models x {Movement, Grounding, Building} x the
first 3 of the 5 clips per cell, copied verbatim. These are already
generated-only (640x704, alpha over bravo), full length, H.264. The two
Consistency folders are excluded.

Items are written with opaque ids in a build-time shuffled order, so neither the
filename nor the ordering leaks the model. The model/episode key lives in
``data/*_key.json``, which the annotation pages never fetch.

Usage
-----
::

    python human-eval/build_human_eval.py
    python human-eval/build_human_eval.py --skip-frames   # re-shuffle only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

# STRICT toggle must be set before the handlers are imported/used so that
# camera_utils._late_episode_strict_enabled() sees it.
os.environ["LATE_EPISODE_QUERY_STRICT"] = "1"

from run_eval import (  # noqa: E402
    extract_query_frames,
    find_generated_video_subdir,
    find_mc_video_pairs,
    identify_handler,
    _maybe_set_late_episode_gen_len,
)

# --- configuration ---------------------------------------------------------

DATASET_BASE = REPO_ROOT / "mc_multiplayer_v2_eval_new_sneak_combined"
GENERATIONS_DIR = REPO_ROOT / "mc_multiplayer_v2_generations"
SUPPLEMENTARY = HERE / "Model Generations on Eval"

# Task 1: the two Table 3 rows the paper reports at 56.8 +/- 2.9 and 34.9 +/- 1.5.
CONSISTENCY_MODELS = ["flagship", "causvid_regression"]
CONSISTENCY_EVALS = ["turnToLookEval", "turnToLookOppositeEval"]

# Task 2: everything in the supplementary folder except the two Consistency
# folders. Including those would give 7 x 5 x N instead of 7 x 3 x N.
ARTIFACT_CATEGORIES = ["Movement", "Grounding", "Building"]

# The supplementary folder ships 5 clips per (category, model); we annotate the
# first N by video index. Changing this reassigns every artifacts item id and so
# invalidates existing artifacts responses — check responses/ before touching it.
ARTIFACT_VIDEOS_PER_CELL = 3

# Supplementary display name -> results/generations model directory.
MODEL_DIR_BY_DISPLAY = {
    "Solaris Default": "flagship",
    "Solaris without Pretrain (Training Ablation)": "from_scratch",
    "Frame Concat (Training Ablation)": "concat_c",
    "Independent (Training Ablation)": "no_player_attn_sf",
    "ODE Reg (Self-Forcing Ablation)": "causvid_regression",
    "Pre-DMD (Self-Forcing Ablation)": "causvid_dmd",
    "No KV Back Prop (Self-Forcing Ablation)": "no_kv_cache_backprop",
}

SHUFFLE_SEED = 20260725

# run_eval.py appends /test to the dataset folder and caps at 32 pairs when
# --limit is not given. Both are replicated below so the query list matches the
# paper run exactly.
DEFAULT_VIDEO_PAIR_LIMIT = 32


def load_video_pairs(dataset_name: str):
    """``run_eval``'s pair discovery: dataset/test, sorted, capped at 32."""
    folder = DATASET_BASE / dataset_name / "test"
    if not folder.is_dir():
        raise SystemExit(f"dataset folder not found: {folder}")
    pairs = find_mc_video_pairs(folder)
    if len(pairs) > DEFAULT_VIDEO_PAIR_LIMIT:
        pairs = pairs[:DEFAULT_VIDEO_PAIR_LIMIT]
    return pairs

OUT_DATA = HERE / "data"
OUT_FRAMES = HERE / "frames"
OUT_VIDEOS = HERE / "videos"


# --- task 1: consistency screenshot pairs ----------------------------------


def collect_consistency_records(model: str, dataset_name: str) -> list[dict]:
    """Replicate ``run_eval.run_evaluation``'s query loop for one (model, eval)
    pair and return one record per query, with the alpha/bravo PNG bytes.

    The loop mirrors run_eval exactly — same pair ordering, same
    ``current_video_id`` bookkeeping, same ``extract_query_frames`` call — so
    the frames match the ones sent to Gemini during the paper run.
    """
    generated_path = GENERATIONS_DIR / model

    generated_subdir = find_generated_video_subdir(generated_path, dataset_name)
    if generated_subdir is None:
        raise SystemExit(f"no generated subdir for {model}/{dataset_name}")

    # Exports LATE_EPISODE_GEN_LEN, which the strict late-horizon frame index
    # depends on. Without it the handler would silently fall back to len(GT)-20.
    _maybe_set_late_episode_gen_len(
        generated_subdir_override=None,
        generated_path=generated_path,
        dataset_name=dataset_name,
    )

    handler = identify_handler(dataset_name)
    video_pairs = load_video_pairs(dataset_name)

    num_generated = len(list(generated_subdir.glob("video_*_side_by_side.mp4")))
    if num_generated < len(video_pairs):
        video_pairs = video_pairs[:num_generated]

    all_queries = []
    for pair in video_pairs:
        all_queries.extend(handler.extract_keyframes(pair))

    records: list[dict] = []
    current_video_id = -1
    last_episode_instance = None

    for query in all_queries:
        meta = query.metadata
        episode_instance = (meta["episode"], meta["instance"])
        if episode_instance != last_episode_instance:
            current_video_id += 1
            last_episode_instance = episode_instance

        frames = extract_query_frames(
            query=query,
            generated_subdir=generated_subdir,
            current_video_id=current_video_id,
            frame1_idx=meta["frame1"],
        )
        records.append({
            "model": model,
            "eval": dataset_name,
            "query_type": meta.get("query_type", "default"),
            "episode": meta["episode"],
            "instance": meta["instance"],
            "expected": query.expected_answer,
            "frame1": meta["frame1"],
            "alpha_frame": meta["alpha_frame"],
            "bravo_frame": meta["bravo_frame"],
            "video_id": current_video_id,
            "generated_subdir": generated_subdir.name,
            "_alpha_png": frames["alpha_frame"],
            "_bravo_png": frames["bravo_frame"],
        })

    print(f"  {model:20s} {dataset_name:24s} {len(records)} queries "
          f"({generated_subdir.name})")
    return records


def collect_calibration_pair(dataset_name: str, pair_index: int = 0) -> dict:
    """Extract one **ground-truth** screenshot pair for ``dataset_name``.

    Shown to the annotator up front so they calibrate on what counts as
    "same" vs "different" scenery. GT frames come from the real episode videos
    (``generated_subdir=None``), so these are unambiguous by construction:
    turnToLookEval is same-side (expected "yes") and turnToLookOppositeEval is
    opposite-sides (expected "no"). Because they are GT rather than model
    output, they cannot leak anything about the 256 scored items.
    """
    handler = identify_handler(dataset_name)
    pair = load_video_pairs(dataset_name)[pair_index]

    # The first query is the original turn-end timestamp (the late-horizon
    # duplicate is appended after it), which is the cleaner teaching example.
    query = handler.extract_keyframes(pair)[0]
    frames = extract_query_frames(
        query=query,
        generated_subdir=None,
        current_video_id=0,
        frame1_idx=query.metadata["frame1"],
    )
    return {
        "eval": dataset_name,
        "expected": query.expected_answer,
        "episode": query.metadata["episode"],
        "instance": query.metadata["instance"],
        "_alpha_png": frames["alpha_frame"],
        "_bravo_png": frames["bravo_frame"],
    }


def build_calibration(skip_frames: bool) -> None:
    """Write the two GT teaching examples shown before the task starts."""
    # Episode indices are hand-picked for legibility: these two pairs make the
    # same/different distinction as unambiguous as the dataset allows.
    same = collect_calibration_pair("turnToLookEval", pair_index=0)
    diff = collect_calibration_pair("turnToLookOppositeEval", pair_index=1)
    assert same["expected"] == "yes" and diff["expected"] == "no", (
        "calibration examples no longer match the handlers' expected answers"
    )

    examples = []
    for slug, rec, verdict, blurb in [
        ("same", same, "Same scenery",
         "Both players are looking at the same part of the world. The camera "
         "position and angle differ — notice the trees and the ridge line "
         "appear in both, just from slightly different spots. A difference in "
         "viewpoint alone never makes a pair &ldquo;different&rdquo;."),
        ("different", diff, "Different scenery",
         "The players are facing opposite directions, so they are looking at "
         "unrelated parts of the world: a large rock formation on one side, "
         "open grassland on the other. Note that the sky, the biome and the "
         "on-screen HUD look alike in both — that is <em>not</em> enough to "
         "call them the same. Judge by the landmarks and terrain."),
    ]:
        if not skip_frames:
            (OUT_FRAMES / f"cal_{slug}_a.png").write_bytes(rec["_alpha_png"])
            (OUT_FRAMES / f"cal_{slug}_b.png").write_bytes(rec["_bravo_png"])
        examples.append({
            "id": f"cal_{slug}",
            "image_a": f"frames/cal_{slug}_a.png",
            "image_b": f"frames/cal_{slug}_b.png",
            "verdict": verdict,
            "answer": "same" if rec["expected"] == "yes" else "different",
            "explanation": blurb,
        })

    _write_json(OUT_DATA / "consistency_calibration.json", {
        "note": "Ground-truth (real, non-generated) example pairs. "
                "Shown before the task; not scored.",
        "examples": examples,
    })
    print(f"  -> {len(examples)} GT calibration examples")


def build_consistency(skip_frames: bool) -> None:
    print("\n[task 1] consistency — extracting the VLM's exact screenshot pairs")
    records: list[dict] = []
    for model in CONSISTENCY_MODELS:
        for dataset_name in CONSISTENCY_EVALS:
            records.extend(collect_consistency_records(model, dataset_name))

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(records)

    if not skip_frames:
        if OUT_FRAMES.exists():
            shutil.rmtree(OUT_FRAMES)
        OUT_FRAMES.mkdir(parents=True)

    items, key = [], []
    for i, rec in enumerate(records, 1):
        item_id = f"c{i:04d}"
        if not skip_frames:
            (OUT_FRAMES / f"{item_id}_a.png").write_bytes(rec["_alpha_png"])
            (OUT_FRAMES / f"{item_id}_b.png").write_bytes(rec["_bravo_png"])
        items.append({
            "id": item_id,
            "image_a": f"frames/{item_id}_a.png",
            "image_b": f"frames/{item_id}_b.png",
        })
        key.append({"id": item_id, **{k: v for k, v in rec.items()
                                      if not k.startswith("_")}})

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    build_calibration(skip_frames=skip_frames)
    _write_json(OUT_DATA / "consistency_items.json", {
        "task": "consistency",
        "question": "Do these two screenshots show the same scenery?",
        "options": [
            {"value": "same", "label": "Same scenery", "key": "S"},
            {"value": "different", "label": "Different scenery", "key": "D"},
        ],
        "items": items,
    })
    _write_json(OUT_DATA / "consistency_key.json", {
        "task": "consistency",
        "strict_late_episode": True,
        "shuffle_seed": SHUFFLE_SEED,
        "models": CONSISTENCY_MODELS,
        "evals": CONSISTENCY_EVALS,
        "items": key,
    })
    print(f"  -> {len(items)} comparisons, {len(items) * 2} PNGs")


# --- task 2: artifact videos -----------------------------------------------


def build_artifacts() -> None:
    print("\n[task 2] artifacts — copying supplementary clips")
    if not SUPPLEMENTARY.is_dir():
        raise SystemExit(f"missing supplementary folder: {SUPPLEMENTARY}")

    records: list[dict] = []
    for category in ARTIFACT_CATEGORIES:
        cat_dir = SUPPLEMENTARY / category
        if not cat_dir.is_dir():
            raise SystemExit(f"missing category folder: {cat_dir}")
        for display, model_dir in MODEL_DIR_BY_DISPLAY.items():
            model_path = cat_dir / display
            if not model_path.is_dir():
                raise SystemExit(f"missing model folder: {model_path}")
            available = sorted(model_path.glob("video_*_gen.mp4"),
                               key=lambda p: int(p.stem.split("_")[1]))
            if len(available) < ARTIFACT_VIDEOS_PER_CELL:
                raise SystemExit(
                    f"{model_path} has {len(available)} clips, "
                    f"need {ARTIFACT_VIDEOS_PER_CELL}"
                )
            for video in available[:ARTIFACT_VIDEOS_PER_CELL]:
                records.append({
                    "category": category,
                    "model": model_dir,
                    "model_display": display,
                    "video_index": int(video.stem.split("_")[1]),
                    "source": str(video.relative_to(HERE)),
                    "_path": video,
                })
        print(f"  {category:12s} "
              f"{sum(1 for r in records if r['category'] == category)} videos")

    rng = random.Random(SHUFFLE_SEED + 1)
    rng.shuffle(records)

    if OUT_VIDEOS.exists():
        shutil.rmtree(OUT_VIDEOS)
    OUT_VIDEOS.mkdir(parents=True)

    items, key = [], []
    for i, rec in enumerate(records, 1):
        item_id = f"a{i:04d}"
        shutil.copyfile(rec["_path"], OUT_VIDEOS / f"{item_id}.mp4")
        items.append({"id": item_id, "video": f"videos/{item_id}.mp4"})
        key.append({"id": item_id, **{k: v for k, v in rec.items()
                                      if not k.startswith("_")}})

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    _write_json(OUT_DATA / "artifacts_items.json", {
        "task": "artifacts",
        "question": "Does this video contain visual artifacts?",
        "options": [
            {"value": "none", "label": "No artifacts", "key": "1"},
            {"value": "character", "label": "Character artifacts", "key": "2"},
            {"value": "terrain", "label": "Terrain artifacts", "key": "3"},
            {"value": "other", "label": "Other artifacts", "key": "4",
             "free_text": True},
        ],
        "items": items,
    })
    _write_json(OUT_DATA / "artifacts_key.json", {
        "task": "artifacts",
        "shuffle_seed": SHUFFLE_SEED + 1,
        "categories": ARTIFACT_CATEGORIES,
        "items": key,
    })
    print(f"  -> {len(items)} videos")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--skip-frames",
        action="store_true",
        help="Reuse the PNGs already in frames/ (re-shuffles ids only). "
             "Only safe if the shuffle seed is unchanged.",
    )
    parser.add_argument("--only", choices=["consistency", "artifacts"],
                        help="Build just one task.")
    args = parser.parse_args()

    if args.only != "artifacts":
        build_consistency(skip_frames=args.skip_frames)
    if args.only != "consistency":
        build_artifacts()

    (HERE / "responses").mkdir(exist_ok=True)
    print("\ndone. serve with:  python human-eval/serve.py")


if __name__ == "__main__":
    main()
