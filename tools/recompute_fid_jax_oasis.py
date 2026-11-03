"""Recompute jax_oasis's in-training eval FID from saved side-by-side eval videos.

The training eval (jax_oasis src/trainers/mg_mp_trainer_util.py) computes FID with
FIDCalculator(num_sources=2) via calculate_metrics_from_batch: Inception features of every
frame except the first (prompt) frame, one FID per player, averaged over the two players.
This script feeds the same function from the saved mp4s instead of the in-memory frames:
each 1280x704 video holds GT on the left and the generation on the right, player 1 on top
and player 2 below. The mp4 is H.264-compressed, so results can differ slightly from the
logged values, which were computed before compression.

Usage (from the eval repo root):
  python tools/recompute_fid_jax_oasis.py mc_multiplayer_v2_generations/<model>/<eval_dir> [...]
"""

import argparse
import sys
from pathlib import Path

import av
import jax.numpy as jnp
import numpy as np

JAX_OASIS = Path.home() / "GitHub/jax_oasis"
sys.path.insert(0, str(JAX_OASIS))
from src.metrics.compute_metrics import FIDCalculator, calculate_metrics_from_batch  # noqa: E402

N_PROMPT_FRAMES = 1  # as in mg_mp_trainer_util.py


def load_side_by_side(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred, gt), each [T, P=2, 352, 640, 3] uint8, prompt frame dropped."""
    with av.open(str(path)) as c:
        frames = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    T, H, W, _ = frames.shape
    h, w = H // 2, W // 2
    gt = np.stack([frames[:, :h, :w], frames[:, h:, :w]], axis=1)
    pred = np.stack([frames[:, :h, w:], frames[:, h:, w:]], axis=1)
    return pred[N_PROMPT_FRAMES:], gt[N_PROMPT_FRAMES:]


def fid_for_dir(d: Path) -> tuple[float, int, int]:
    videos = sorted(d.glob("video_*_side_by_side.mp4"),
                    key=lambda p: int(p.name.split("_")[1]))
    assert videos, f"no side-by-side videos in {d}"
    calc = FIDCalculator(num_sources=2)
    n_frames = 0
    for v in videos:
        pred, gt = load_side_by_side(v)
        T = pred.shape[0]
        n_frames += T
        calculate_metrics_from_batch(
            preds=jnp.asarray(pred[None]), targets=jnp.asarray(gt[None]),
            metrics_to_cal=["fid"], real_lengths=jnp.array([T]),
            fid_calculator=calc,
        )
    return float(calc.get_fid_curve_jax()), len(videos), n_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    for d in ap.parse_args().dirs:
        fid, nv, nf = fid_for_dir(d)
        print(f"RESULT fid={fid:.4f} videos={nv} frames_per_player={nf} dir={d}", flush=True)


if __name__ == "__main__":
    main()
