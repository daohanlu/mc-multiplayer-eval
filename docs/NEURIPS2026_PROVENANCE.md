# NeurIPS 2026 submission: where the reported numbers come from

The submission was sent on 2026-05-06. This file records how to regenerate each table
of evaluation numbers in it, and which results trees, code states, and data produced them.
Branch `neurips2026-camera-ready` holds the code for the camera-ready. The training side,
including the experiment script and checkpoints behind each paper row and the FID logs,
is on the same-named branch of the training repo
([ojmichel/jax_oasis, `neurips2026-camera-ready`](https://github.com/ojmichel/jax_oasis/tree/neurips2026-camera-ready),
see its `NEURIPS2026.md`).

| Paper table | Regenerate with | Details |
|---|---|---|
| Tables 2 and 3 (VLM accuracy and FID) | `python vlm_eval/build_vlm_tables.py --late-tree results_json_late_episode_strict` | [below](#tables-2-and-3-vlm-accuracy-and-fid) |
| Table 8 (test-split FVD) | `bash fvd/fvd_table8.sh` | [`fvd/README.md`](../fvd/README.md) |
| Co-movement eval | `bash vlm_eval/comovement_rel_models.sh` | [`vlm_eval/COMOVEMENT_EVAL.md`](../vlm_eval/COMOVEMENT_EVAL.md) |
| Fig. 9 (N-player memory and time scaling) | `python figures/n_player_scaling/plot_n_scaling.py` | [below](#fig-9-n-player-scaling) |

## Model names

| Paper row | VLM results dir (`build_vlm_tables.py`) | Co-movement key | FVD short name |
|---|---|---|---|
| Independent | `no_player_attn_sf` | `no_player_attn_sf` | `no_player_attn_sf` |
| Frame concat | `concat_c` | `concat_c` | `v2_blk` |
| Solaris w/o pretrain | `from_scratch` | `from_scratch` | `v2_scratch` |
| Solaris (and the shaded Self Forcing row) | `flagship` | `solaris` | `v2_flag` |
| ODE Reg, Pre-DMD ×, KV-BP ✓ | `causvid_regression` | `causvid_regression` | `v2_reg` |
| Causal FT, Pre-DMD ✓, KV-BP ✓ | `causvid_dmd` | `causvid_dmd` | `dmd_cont` |
| Causal FT, Pre-DMD ×, KV-BP × | `no_kv_cache_backprop` | `no_kv_cache_backprop` | `v2_nokv` |

All models are checkpoint `step_0001200`.

## Tables 2 and 3: VLM accuracy and FID

`build_vlm_tables.py --late-tree results_json_late_episode_strict` reproduces all 32 cells
of the submitted tables, accuracy ± std and FID (checked against the PDF on 2026-07-25 and
again on 2026-11-02). The FID cells are written into the script's template, not read from
the results JSON. Their source is traced [below](#fid-cells).

**`--late-tree` must be given.** The default (`results_json_late_episode_mixed_thinking`)
does not reproduce the paper. It matches only 7 of the 24 Grounding, Building, and
Consistency cells (checked 2026-11-02); Solaris Building, for example, comes out 10.4 ± 1.5
instead of 9.4 ± 0.0.

| Column | Eval | Generation subdir | Results tree | Gemini thinking |
|---|---|---|---|---|
| Movement | translationEval | `..._eval_translation` | `results_json/` | off |
| Grounding | oneLooksAwayEval_long | `..._eval_one_looks_away_max_speed_long` | `results_json_late_episode_strict/` | off |
| Building | structureEval | `..._eval_structure_max_speed` | `results_json_late_episode_strict/` | on |
| Consistency | turnToLookEval + turnToLookOppositeEval | `..._eval_turn_to_look{,_opposite}_max_speed` | `results_json_late_episode_strict/` | on |

Ground truth is `mc_multiplayer_v2_eval_new_sneak_combined/`, generations are
`mc_multiplayer_v2_generations/<model>/`, 32 episodes per eval, 3 trials. The strict tree
has 3 VLM API errors, all for `no_kv_cache_backprop`, counted as incorrect.

### Code state

Grounding, Building, and Consistency ran from 2026-05-05 23:07 to 2026-05-06 08:47 on
`icml-ablations` at `3360d56` with uncommitted changes. Those changes were committed on
2026-07-25 as `4eb426c`, the first commit that reproduces the numbers. The logs are
`logs/eval_strict_generated{,_retry,_structure,_structure_thinking}.log`.

Movement was not part of that run. Its trial files come from older code:

| Models | Date | HEAD |
|---|---|---|
| flagship, from_scratch, causvid_dmd, causvid_regression, no_kv_cache_backprop | 2026-01-26 | `0c79a4a` |
| concat_c | 2026-01-28 | `27d6e77` |
| no_player_attn_sf | 2026-05-02 | `3360d56` with uncommitted changes |

`handlers/mc_multiplayer_handler_translation.py` and `handlers/camera_utils.py` are
identical between `0c79a4a` and `3360d56`, and the `run_eval.py` changes between them (the
33-frame ablation flag, `_long` dataset matching) do not affect translationEval. The column
was still never run under a single code state. One `--eval-types translation` run on
`4eb426c` or later would settle that for the camera-ready.

### FID cells

The 28 FID cells come from the in-training eval of jax_oasis at step 1200 (logged as
step 1201), not from this repo. They were copied by hand into the template of
`build_vlm_tables.py`. Traced on 2026-11-03:

| Column | Source |
|---|---|
| Movement | `test_fid__…eval_translation` |
| Grounding | `one_looks_away_max_speed_long_new` in jax_oasis `long_eval_logs/<run>.log` (branch `neurips2026-camera-ready`; read with `scripts/extract_fid.sh`); Independent from wandb `test_fid__eval_one_looks_away` |
| Building | FID on `eval_structure_max_speed`. The training runs only logged `eval_structure`, which gives different values (ODE Reg 88.45, Frame concat 100.19), so these came from separate eval runs that were not found |
| Consistency | mean of the turn-to-look and turn-to-look-opposite (`_max_speed`) FIDs |

The team spreadsheet export `Solaris Experiments - Multiplayer v2 Script Check.csv`
(2026-01-28, repo root) records Movement ("FID Translation"), Building, and Consistency for
the six non-Independent models. It truncates to two decimals (logged 129.4963 appears as
129.49), and the paper rounds those values.

| Paper row | Movement | Grounding | Building | Consistency |
|---|---|---|---|---|
| Independent | 58.9 ← wandb 58.91 | 67.1 ← wandb 67.08 | **77.8: not found** | 104.8 ← wandb (104.99 + 104.54)/2 |
| Frame concat | 69.5 ← wandb 69.5026 | 66.6 ← log 66.57 | 103.2 ← sheet 103.24 | 129.4 ← sheet 129.38 (wandb has turn-to-look 129.4963 only) |
| Solaris w/o pretrain | 37.11 ← sheet | 49.9 ← log 49.85 | 86.6 ← sheet 86.60 | 121.4 ← sheet 121.39 |
| Solaris | 36.84 ← sheet | 38.0 ← log 38.03 | 83.6 ← sheet 83.58 | 99.4 ← sheet 99.40 |
| ODE Reg | 77.42 ← wandb 77.4241 | 56.6 ← log 56.59 | 95.7 ← sheet 95.69 | 142.3 ← wandb (130.93 + 153.72)/2 |
| Causal FT, Pre-DMD ✓ | 54.36 ← sheet | 40.4 ← log 40.38 | 90.5 ← sheet 90.51 | 160.1 ← sheet 160.08 |
| Causal FT, no KV-BP | 53.76 ← sheet | 55.2 ← log 55.23 | 87.4 ← sheet 87.43 | 105.1 ← sheet 105.07 |

"wandb" is the run history of `nyu-visionx/oasis` at step 1201: run
`v2_concat_c_fix_bidirectional_fix_blocksize_sf`,
`v2_flagship_fix_bidirectional_sf_regression_lr_1e-4_continue`, or `no_player_attn_sf`.
"log" is `long_eval_logs`. "sheet" is the spreadsheet export.

In summary, 12 cells match a raw logged value, 15 match only the spreadsheet, and 1 was
not found. The linked training runs of the other models have no eval FIDs in their wandb
history, so their raw values were logged elsewhere, probably in eval-only runs or TPU logs.
For Independent Building, `no_player_attn_sf` has an `eval_structure` key with no values.
**GCS search (2026-11-03): no logged Building FID was found.** Recursive listings of
`gs://solaris-central1/{output,outputs,solaris/output,persist}` and
`gs://solaris-east5/{outputs,solaris,persist}` (about 370,000 objects) have no
`structure_max_speed` folder and no metric or eval log for it. The only `.log` files are the
July co-movement `run.log`s. The January generations were downloaded (zsh history,
2026-01-28) from `gs://solaris-central1/solaris/output/<run>/step_0001200_multiplayer_v2_eval_*`,
and that prefix now holds 3 objects, so the original eval folders, with any files saved next to
them, no longer exist. Independent's local `step_0001200_multiplayer_v2_eval_structure_max_speed`
is a copy of `gs://solaris-central1/output/no_player_attn_sf/step_0001200_eval_structure/`
(all 32 files md5-identical, created 2026-05-02). Its ground-truth halves show the same
episodes as Solaris's `structure_max_speed` set, so it is the same Building eval under the
newer set name. The run that made it, wandb `no_player_attn_sf`, has a
`test_fid__eval_structure` key with no values. So 77.8 has no recorded source. The paper
keeps it as submitted (Fred's decision).

**Recompute attempt (2026-11-03): does not reproduce.** `tools/recompute_fid_jax_oasis.py`
feeds the saved side-by-side mp4s to jax_oasis's own `calculate_metrics_from_batch` with
`FIDCalculator(num_sources=2)`, as the training eval does: prompt frame dropped, one FID per
player, averaged. On Solaris's 32 Building videos
(`mc_multiplayer_v2_generations/flagship/step_0001200_multiplayer_v2_eval_structure_max_speed`,
8,192 frames per player) it gives **81.38**, not 83.58. Independent was therefore not
recomputed, since the recipe is not validated. Known differences from the logged run: the
mp4s are H.264-compressed (the training eval scored in-memory frames), and it is not
established that these videos come from the same eval run as the logged value. The
Inception model ran on CPU (`JAX_PLATFORMS=cpu`): on the RTX 4090, XLA rewrites the
pool-count convolution in `inception.py` `avg_pool` into int8, which cuDNN rejects, even with
`--xla_gpu_strict_conv_algorithm_picker=false`.

## Table 8: test-split FVD

See [`fvd/README.md`](../fvd/README.md). In short: Table 8 is the random-window 16-frame FVD
(seed 42) with both players pooled, computed on 2026-05-06 on the 2,048 test-split segments.
The shaded Self Forcing row is Solaris, so it should read **81.3**. The 88.7 printed there
is Independent's value. The features, I3D weights, and script snapshots are archived on
`/mnt/data-hdd/dl3957/neurips2026_eval/fvd/` and at
`gs://solaris-central1/solaris/neurips2026_eval/fvd/`.

## Fig. 9: N-player scaling

`figures/n_player_scaling/plot_n_scaling.py` draws the figure from the four CSVs in
`figures/n_player_scaling/data/`: peak HBM and time per training iteration on TPU v5p, for
Splash and standard attention at full and 25% width, with random inputs (which cost the same
memory and time as real gameplay data). Each CSV row names its profiling run; all 51 runs are
under `gs://solaris-central1/outputs/<run_name>/` (checked 2026-11-06). The script writes
`plots/plot_scaling.pdf`, the paper figure: both panels at the paper's text width, vector,
with TrueType fonts. Copy it to `figs/assets/scaling.pdf` in the paper. It also writes the two
original standalone slide-size plots.

## Co-movement eval

Each model's generations are in its own bucket under `gs://solaris-east5/outputs/`, in a
`co_movement_rel/` subfolder of 32 side-by-side videos. The `co_movement_divider_rel/`
subfolder is not used. Ground truth is
`gs://solaris-central1/solaris/data/neurips_eval_coMovement/coMovementAlwaysRelativeMotionEval`.
`vlm_eval/comovement_rel_models.sh` lists every bucket with its restored checkpoint, and
downloads and runs the eval. The flagship's key there is `solaris`, not `flagship`, because
its results were first collected under that name.

## Camera-ready text corrections found so far

| Where | Submitted | Should be |
|---|---|---|
| Table 8, shaded Self Forcing row | 88.7 | 81.3 |
| Sec. 6, FVD set | "2,048 held-out training videos" | 2,048 random 16-frame segments from the held-out test split |
| App. D, FVD | "averaged across two players" | pooled over both players |
| Table 8 caption | "the full evaluation set" | the 2,048 test-split segments (not split by task type) |
