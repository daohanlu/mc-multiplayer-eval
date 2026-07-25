# Mean episode accuracy (%) — STRICT late-episode toggle

Source: `results_json_late_episode_strict/`. The strict toggle
(`LATE_EPISODE_QUERY_STRICT=1`) makes each handler emit BOTH the original
"last" query (right after the last meaningful action) AND a duplicate at the
late-horizon frame (`frame1 + GEN_LEN − 20`). Episode-level accuracy is
computed with AND-semantics across all queries in an episode, so a model only
counts an episode as correct when it answers correctly at BOTH timestamps.
This is designed to penalize models that hallucinate actions / lose state by
the late horizon.

Each cell shows `mean ± std` of `episode_level_accuracy.episode_accuracy`
across 3 trials (population stddev, matching `stats.json`). The
`turnToLook avg` column is the mean of `turnToLookEval` and
`turnToLookOppositeEval`; its std is `sqrt(s1² + s2²) / 2`, treating the two
evals as independent. The final row `real (GT)` is the same VLM run on
ground-truth videos (no `--generated`).

`translation*` is taken from the original (non-strict) eval at
`results_json/`, since the late-episode toggle isn't meaningful for
`translationEval` (the actions are reversed in the latter half of the
episode).

VLM thinking is enabled for `turnToLookEval`, `turnToLookOppositeEval`, and
`structureEval`; it is off for `translationEval` and `oneLooksAwayEval_long`.

| model | translation* | turnToLook | turnToLookOpp | turnToLook avg | oneLooksAway_long | structure |
|---|---:|---:|---:|---:|---:|---:|
| `causvid_dmd` | 35.42 ± 3.90 | 82.29 ± 5.31 | 9.38 ± 0.00 | 45.83 ± 2.66 | 3.12 ± 0.00 | 2.08 ± 1.47 |
| `causvid_regression` | 19.79 ± 1.47 | 45.83 ± 1.47 | 23.96 ± 3.90 | 34.90 ± 2.08 | 4.17 ± 1.47 | 3.12 ± 0.00 |
| `concat_c` | 84.38 ± 4.42 | 6.25 ± 2.55 | 44.79 ± 5.31 | 25.52 ± 2.95 | 50.00 ± 2.55 | 0.00 ± 0.00 |
| `flagship` | 67.71 ± 5.31 | 60.42 ± 6.42 | 53.12 ± 2.55 | 56.77 ± 3.45 | 53.12 ± 0.00 | 9.38 ± 0.00 |
| `from_scratch` | 69.79 ± 1.47 | 45.83 ± 1.47 | 19.79 ± 2.95 | 32.81 ± 1.65 | 25.00 ± 0.00 | 0.00 ± 0.00 |
| `no_kv_cache_backprop` | 88.54 ± 1.47 | 66.67 ± 1.47 | 38.54 ± 2.95 | 52.60 ± 1.65 | 77.08 ± 1.47 | 5.21 ± 2.95 |
| `no_player_attn_sf` | 12.50 ± 0.00 | 11.46 ± 1.47 | 64.58 ± 3.90 | 38.02 ± 2.08 | 26.04 ± 1.47 | 0.00 ± 0.00 |
| *real (GT)* | 100.00 ± 0.00 | 95.83 ± 1.47 | 93.75 ± 5.10 | 94.79 ± 2.66 | 98.96 ± 1.47 | 82.29 ± 5.31 |
