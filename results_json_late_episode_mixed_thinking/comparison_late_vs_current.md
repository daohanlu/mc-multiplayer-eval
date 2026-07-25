# Late-episode vs current-pattern comparison

- OLD (current pattern): `results_json/generated`
- NEW (late-episode toggle): `results_json_late_episode_mixed_thinking/generated`
- Models (7): `causvid_dmd`, `causvid_regression`, `concat_c`, `flagship`, `from_scratch`, `no_kv_cache_backprop`, `no_player_attn_sf`
- Evals: `translationEval`, `rotationEval`, `structureEval`, `turnToLookEval`, `turnToLookOppositeEval`, `oneLooksAwayEval_long`, `bothLookAwayEval_long`

Notes:
- `translation*` is pinned to the OLD value in the NEW table (and Δ=0). The dataset reverses bot actions in the second half of the episode, so a late-horizon comparison frame is ill-posed.
- `turnToLook avg` averages `turnToLookEval` and `turnToLookOppositeEval` per (model, pattern).

## Per-model accuracy across evals

### Mean episode accuracy (%) — OLD pattern (current)

| model | translation | rotation | structure | turnToLook | turnToLookOpp | turnToLook avg | oneLooksAway_long | bothLookAway_long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `causvid_dmd` | 35.42 | 7.29 | 8.33 | 90.62 | 19.79 | 55.21 | 3.12 | 0.00 |
| `causvid_regression` | 19.79 | 27.08 | 3.12 | 62.50 | 35.42 | 48.96 | 3.12 | 0.00 |
| `concat_c` | 84.38 | 69.79 | 0.00 | 12.50 | 86.46 | 49.48 | 53.12 | 37.50 |
| `flagship` | 67.71 | 68.75 | 20.83 | 71.88 | 70.83 | 71.35 | 62.50 | 37.50 |
| `from_scratch` | 69.79 | 68.75 | 0.00 | 61.46 | 37.50 | 49.48 | 29.17 | 18.75 |
| `no_kv_cache_backprop` | 88.54 | 68.75 | 15.62 | 78.12 | 63.54 | 70.83 | 72.92 | 48.96 |
| `no_player_attn_sf` | 12.50 | 84.38 | 0.00 | 28.12 | 83.33 | 55.73 | 83.33 | 77.08 |

### Mean episode accuracy (%) — NEW pattern (late-episode toggle)

`translation*` reuses the OLD value; the late-episode pattern is not meaningful when the dataset reverses actions in the latter half of the episode.

| model | translation | rotation | structure | turnToLook | turnToLookOpp | turnToLook avg | oneLooksAway_long | bothLookAway_long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `causvid_dmd` | 35.42 | 9.38 | 2.08 | 90.62 | 17.71 | 54.17 | 3.12 | 0.00 |
| `causvid_regression` | 19.79 | 34.38 | 3.12 | 50.00 | 44.79 | 47.40 | 4.17 | 0.00 |
| `concat_c` | 84.38 | 20.83 | 0.00 | 22.92 | 64.58 | 43.75 | 78.12 | 47.92 |
| `flagship` | 67.71 | 71.88 | 10.42 | 76.04 | 64.58 | 70.31 | 52.08 | 43.75 |
| `from_scratch` | 69.79 | 63.54 | 0.00 | 69.79 | 47.92 | 58.85 | 27.08 | 19.79 |
| `no_kv_cache_backprop` | 88.54 | 68.75 | 8.33 | 77.08 | 52.08 | 64.58 | 78.12 | 56.25 |
| `no_player_attn_sf` | 12.50 | 19.79 | 0.00 | 29.17 | 71.88 | 50.52 | 29.17 | 6.25 |

### Δ (NEW − OLD) — late-episode toggle effect

| model | translation | rotation | structure | turnToLook | turnToLookOpp | turnToLook avg | oneLooksAway_long | bothLookAway_long |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `causvid_dmd` | +0.00 | +2.08 | -6.25 | +0.00 | -2.08 | -1.04 | +0.00 | +0.00 |
| `causvid_regression` | +0.00 | +7.29 | +0.00 | -12.50 | +9.38 | -1.56 | +1.04 | +0.00 |
| `concat_c` | +0.00 | -48.96 | +0.00 | +10.42 | -21.88 | -5.73 | +25.00 | +10.42 |
| `flagship` | +0.00 | +3.12 | -10.42 | +4.17 | -6.25 | -1.04 | -10.42 | +6.25 |
| `from_scratch` | +0.00 | -5.21 | +0.00 | +8.33 | +10.42 | +9.38 | -2.08 | +1.04 |
| `no_kv_cache_backprop` | +0.00 | +0.00 | -7.29 | -1.04 | -11.46 | -6.25 | +5.21 | +7.29 |
| `no_player_attn_sf` | +0.00 | -64.58 | +0.00 | +1.04 | -11.46 | -5.21 | -54.17 | -70.83 |

## Discriminativeness across models

Spread = max(model mean) − min(model mean). Std = population stddev across models. Larger spread/std → the test more cleanly separates models. `translation*` rows reuse the OLD value for both patterns (see top of report), so their Δ is 0 by construction.

| eval | OLD spread | NEW spread | Δ spread | OLD std | NEW std | Δ std |
|---|---:|---:|---:|---:|---:|---:|
| `translation*` | 76.04 | 76.04 | +0.00 | 28.76 | 28.76 | +0.00 |
| `rotation` | 77.08 | 62.50 | -14.58 | 25.89 | 24.29 | -1.59 |
| `structure` | 20.83 | 10.42 | -10.42 | 7.83 | 3.97 | -3.87 |
| `turnToLook` | 78.12 | 67.71 | -10.42 | 25.79 | 23.92 | -1.87 |
| `turnToLookOpp` | 66.67 | 54.17 | -12.50 | 23.98 | 16.72 | -7.26 |
| `turnToLook avg` | 22.40 | 26.56 | +4.17 | 9.10 | 8.81 | -0.29 |
| `oneLooksAway_long` | 80.21 | 75.00 | -5.21 | 30.17 | 29.21 | -0.96 |
| `bothLookAway_long` | 77.08 | 56.25 | -20.83 | 25.64 | 22.30 | -3.34 |

## Biggest accuracy swings (|Δ|), top 15

Excludes `translation*` (pinned) and the synthetic `turnToLook avg` column.

| eval | model | OLD | NEW | Δ |
|---|---|---:|---:|---:|
| `bothLookAway_long` | `no_player_attn_sf` | 77.08 | 6.25 | -70.83 |
| `rotation` | `no_player_attn_sf` | 84.38 | 19.79 | -64.58 |
| `oneLooksAway_long` | `no_player_attn_sf` | 83.33 | 29.17 | -54.17 |
| `rotation` | `concat_c` | 69.79 | 20.83 | -48.96 |
| `oneLooksAway_long` | `concat_c` | 53.12 | 78.12 | +25.00 |
| `turnToLookOpp` | `concat_c` | 86.46 | 64.58 | -21.88 |
| `turnToLook` | `causvid_regression` | 62.50 | 50.00 | -12.50 |
| `turnToLookOpp` | `no_kv_cache_backprop` | 63.54 | 52.08 | -11.46 |
| `turnToLookOpp` | `no_player_attn_sf` | 83.33 | 71.88 | -11.46 |
| `turnToLook` | `concat_c` | 12.50 | 22.92 | +10.42 |
| `structure` | `flagship` | 20.83 | 10.42 | -10.42 |
| `turnToLookOpp` | `from_scratch` | 37.50 | 47.92 | +10.42 |
| `oneLooksAway_long` | `flagship` | 62.50 | 52.08 | -10.42 |
| `bothLookAway_long` | `concat_c` | 37.50 | 47.92 | +10.42 |
| `turnToLookOpp` | `causvid_regression` | 35.42 | 44.79 | +9.38 |

## GT validation: is the late-episode question well-posed?

OLD GT: `results_json/real/<eval>/stats.json` — current-pattern accuracy on real videos.
NEW GT: `results_json_late_episode_mixed_thinking/real/<eval>/stats.json` — late-episode toggle on real videos (generated via `validate_late_episode_gt.sh`).

Verdict: `OK` if NEW GT ≥ 90%, `WARN` in [75%, 90%), `FAIL` below 75%. A `WARN`/`FAIL` row points back to the handler's late-frame choice for that eval (see `frame_extraction/<eval>/_real/`). `translationEval` is `n/a (pinned)` because the dataset reverses actions in the latter half of the episode.

| eval | OLD GT (current) | NEW GT (late) | Δ | verdict |
|---|---:|---:|---:|:---:|
| `translation` | 100.00 | n/a (pinned) | n/a | n/a |
| `rotation` | 96.88 | 88.54 | -8.33 | WARN |
| `structure` | 98.96 | 93.75 | -5.21 | OK |
| `turnToLook` | 98.96 | 97.92 | -1.04 | OK |
| `turnToLookOpp` | 93.75 | 96.88 | +3.12 | OK |
| `oneLooksAway_long` | 96.88 | 100.00 | +3.12 | OK |
| `bothLookAway_long` | 92.71 | 98.96 | +6.25 | OK |
