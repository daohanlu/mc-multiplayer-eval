# NeurIPS 2026 submission: where the reported numbers come from

The submission was sent on 2026-05-06. This file records how to regenerate each table
of evaluation numbers in it, and which results trees, code states, and data produced them.
Branch `neurips2026-camera-ready` holds the code for the camera-ready.

| Paper table | Regenerate with | Details |
|---|---|---|
| Tables 2 and 3 (VLM accuracy and FID) | `python vlm_eval/build_vlm_tables.py --late-tree results_json_late_episode_strict` | [below](#tables-2-and-3-vlm-accuracy-and-fid) |
| Table 8 (test-split FVD) | `bash fvd/fvd_table8.sh` | [`fvd/README.md`](../fvd/README.md) |
| Co-movement eval | `bash vlm_eval/comovement_rel_models.sh` | [`vlm_eval/COMOVEMENT_EVAL.md`](../vlm_eval/COMOVEMENT_EVAL.md) |

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
the results JSON.

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

## Table 8: test-split FVD

See [`fvd/README.md`](../fvd/README.md). In short: Table 8 is the random-window 16-frame FVD
(seed 42) with both players pooled, computed on 2026-05-06 on the 2,048 test-split segments.
The shaded Self Forcing row is Solaris, so it should read **81.3**. The 88.7 printed there
is Independent's value. The features, I3D weights, and script snapshots are archived on
`/mnt/data-hdd/dl3957/neurips2026_eval/fvd/` and at
`gs://solaris-central1/solaris/neurips2026_eval/fvd/`.

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
