# Dataset Source Mapping

This document maps each eval subdirectory in `mc_multiplayer_v2_eval_new_sneak_combined/` to its original source data on disk and its corresponding eval_ids file in `~/GitHub/jax_oasis/src/data/eval_ids/`.

## Source data

| Combined subdir | Source on `/mnt/data/dl3957/` |
|---|---|
| `bothLookAwayEval` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes/bothLookAwayEval` |
| `bothLookAwayEval_long` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_new_sneak_max_speed_long/bothLookAwayEval` |
| `oneLooksAwayEval` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes/oneLooksAwayEval` |
| `oneLooksAwayEval_long` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_new_sneak_max_speed_long/oneLooksAwayEval` |
| `rotationEval` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes/rotationEval` |
| `structureEval` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes/structureEval` |
| `structureNoPlaceEval` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes/structureNoPlaceEval` |
| `translationEval` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_32_episodes/translationEval` |
| `turnToLookEval` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes/turnToLookEval` |
| `turnToLookOppositeEval` | `/mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes/turnToLookOppositeEval` |

`translationEval` comes from the non-max-speed packaged data; everything else (except `_long`) comes from the max-speed variant.

## Corresponding eval_ids

All eval_ids have been verified to match the combined directory exactly (32 episodes each, zero mismatches).

| Combined subdir | eval_ids file |
|---|---|
| `bothLookAwayEval` | `eval_ids_multiplayer_v2_eval_both_look_away_max_speed.json` |
| `bothLookAwayEval_long` | `eval_ids_multiplayer_v2_eval_both_look_away_max_speed_long.json` |
| `oneLooksAwayEval` | `eval_ids_multiplayer_v2_eval_one_looks_away_max_speed.json` |
| `oneLooksAwayEval_long` | `eval_ids_multiplayer_v2_eval_one_looks_away_max_speed_long.json` |
| `rotationEval` | `eval_ids_multiplayer_v2_eval_rotation_max_speed.json` |
| `structureEval` | `eval_ids_multiplayer_v2_eval_structure_max_speed.json` |
| `structureNoPlaceEval` | `eval_ids_multiplayer_v2_eval_structure_no_place_max_speed.json` |
| `translationEval` | `eval_ids_multiplayer_v2_eval_translation.json` |
| `turnToLookEval` | `eval_ids_multiplayer_v2_eval_turn_to_look_max_speed.json` |
| `turnToLookOppositeEval` | `eval_ids_multiplayer_v2_eval_turn_to_look_opposite_max_speed.json` |

The structure eval_ids were regenerated on 2026-03-27 from the local 32-episode data to fix a prior mismatch (the old eval_ids had been generated from a different 64-episode collection).
