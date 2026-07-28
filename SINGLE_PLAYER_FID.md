# Single-player FID: Solaris vs Matrix-Game 2.0

Backs **Q2** of the R2 nAFu rebuttal. The numbers live in two other repos; this
file records what was reported, where it came from, and one **unresolved
discrepancy** (see the bottom section).

## Reported numbers

Dataset `wander_maxlook`, 64 held-out episodes x 257 frames = 16,448 frames per
side. Clean-FID against the aligned ground-truth frames of the same episodes.

| Model | clean-FID | Above floor |
|---|---|---|
| Ground truth (floor) | 20.646 | -- |
| **Solaris Default** | **27.638** | **+7.0** |
| Matrix-Game 2.0, camera clipped to +/-0.1 | 76.787 | +56.1 |

Solaris is the 2-player flagship (`solaris.pt`) with **player attention on**, not
a single-player variant: player 0 gets the real episode and its recorded actions,
player 1 is an inert partner conditioned on a frame from a different episode and
issuing no actions. Only player 0 is scored. So this measures "flagship with an
irrelevant partner", not a true single-player model.

Matrix-Game 2.0's camera input is clipped to +/-0.1, the only value its shipped
demos use. Solaris needs no clipping: it was trained with a +/-20 deg/frame limit
and the largest single-frame yaw in the 64 windows is 17.19 deg, so its
conditioning is exact by construction.

**We deliberately do not report MG2's unclipped arm** (clean-FID 95.696).
Sustained high mouse values degenerate in that model, so unclipped is not an
intended inference mode. The clipped arm is also MG2's *best* arm, so quoting it
is the conservative choice. The asymmetry is still disclosed in the rebuttal by
the sentence stating that we clip MG2 and do not clip Solaris.

Not reported either, by decision: temporal coherence. For the record, Solaris is
+0.116 above GT coherence (0.758 vs 0.642) and MG2 clip-0.1 is +0.166 above its
GT (0.810 vs 0.644) on this same dataset, so Solaris is the closer of the two.

**These FID values do not compare against the paper's FID columns.** Different
dataset, different reference set, and the paper's cells come from another pipeline
(hard-coded in `build_vlm_tables.py`). The rebuttal says so explicitly.

## Provenance

| What | Where |
|---|---|
| MG2 rollouts, scorer, floor | `solaris-wm/matrix-game-2-eval`, `main` — `results/wander_maxlook/fid_results_clip0.1.json` |
| Solaris rollouts | `ojmichel/persistent-mp-wm`, branch `egor/sp-fid` — `results/wander_mg2_fid/wander_maxlook/fid_results.json` |
| Scorer | `matrix-game-2-eval/compute_fid.py`, used **unmodified** by both |

Both repos are private; read them with `gh api`, e.g.

```bash
gh api "repos/solaris-wm/matrix-game-2-eval/contents/results/wander_maxlook/fid_results_clip0.1.json" \
  --jq '.content' | base64 -d
```

## OPEN: the two repos resample differently before scoring

**Status: known, not resolved. Deliberately left alone — the fix was judged too
risky to attempt during the rebuttal window.**

There are two resizes in the chain, and only the second is governed by any FID
convention:

| Step | Who | MG2 repo | Solaris repo |
|---|---|---|---|
| 1280x720 -> 640x352 (**export**, writes the mp4s) | each repo's own runner | PIL bicubic | `jax.image.resize(method=LINEAR)` |
| 640x352 -> 299x299 (**scorer**, into Inception) | shared `compute_fid.py` | PIL bicubic + antialias | same |

Step 2 is already canonical and identical for both: `compute_fid.py:38` calls
`make_resizer("PIL", False, "bicubic", (299, 299))`, which is clean-fid's
reference resizer and the prescription of the clean-FID paper (Parmar, Zhang &
Zhu, CVPR 2022). Nothing wrong there.

Step 1 is where they diverge, and **no FID convention governs it** — it is just
how each repo chose to render and store frames. What matters is only that both
models get identical treatment, because different resampling leaves different
high-frequency content, which shifts Inception features.

**Why this is minor rather than broken:** `jax.image.resize` defaults to
`antialias=True` (verified against the JAX docs), so the Solaris export is
antialiased with a *linear* kernel rather than aliased. The empirical evidence
agrees — the GT-vs-GT floor computed from each repo's own exported GT mp4s comes
out **20.646** (MG2, bicubic) vs **20.791** (Solaris, jax linear), 0.7% apart.
The floor depends only on the GT frames, so that 0.145 gap is the whole
measurable effect of the resampling difference.

**What the rebuttal reports:** the bicubic floor, 20.646, as the single floor
value, with both models' "above floor" derived from it. Solaris's own-export
floor of 20.791 would put it at +6.8 instead of +7.0. The difference is inside
rounding and does not touch any conclusion.

### If someone resolves this later

Standardize the **export** on PIL bicubic, i.e. change the Solaris side:

```python
PIL.Image.fromarray(arr).resize((640, 352), PIL.Image.BICUBIC)   # keep the crop as-is
```

Reasons for that direction: it matches the bicubic filter already used downstream
by the scorer, it is the convention so it needs no defending, and it is the
cheaper side to redo (one Solaris arm, versus MG2's three).

Three things to get right:

1. **The generated export and the `_gt.mp4` export must change together.** The
   floor is computed from the exported GT, so changing only one makes the floor
   incomparable to the score.
2. **Recompute the FID and the floor in the same run.** Both will move slightly.
   Expect order-of-a-point, not a transformation.
3. It needs a TPU rollout, which is the reason this was not done:
   `TPU_NAME=<tpu> bash scripts/tpu/experiments/24_wander_mg2_fid_diff_partner.sh`,
   then `compute_fid.py` from `matrix-game-2-eval`. A code change alone changes no
   number, and shipping an unvalidated resampling change to a shared branch is
   worse than the 0.145 discrepancy it fixes.
