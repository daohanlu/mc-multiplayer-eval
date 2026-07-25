# Mean episode accuracy (%) — NEW pattern (late-episode toggle)

Source: `results_json_late_episode_mixed_thinking/`. Each cell shows
`mean ± std` of `episode_level_accuracy.episode_accuracy` across 3 trials
(population stddev, matching `stats.json`). The `turnToLook avg` column is the
mean of `turnToLookEval` and `turnToLookOppositeEval`; its std is
`sqrt(s1² + s2²) / 2`, treating the two evals as independent. The final row
`real (GT)` is the same VLM run on ground-truth videos (no `--generated`).

> Note: `translation*` is **pinned to the OLD-pattern values** from
> `results_json/generated/{model}_translationEval/` (and `results_json/real/`
> for GT). The late-episode toggle is not meaningful for `translationEval`
> because the dataset reverses bot actions in the latter half of the episode,
> so a late-horizon query answers a different question than the prompt
> intends. All `translation*` cells were collected with `thinking_enabled=False`
> (handler default); every other column uses the per-eval thinking config
> documented in `README.md`.

| model | translation* | rotation | structure | turnToLook | turnToLookOpp | oneLooksAway_long | bothLookAway_long | turnToLook avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `causvid_dmd` | 35.42 ± 3.90 | 9.38 ± 2.55 | 2.08 ± 1.47 | 90.62 ± 4.42 | 17.71 ± 3.90 | 3.12 ± 0.00 | 0.00 ± 0.00 | 54.17 ± 2.95 |
| `causvid_regression` | 19.79 ± 1.47 | 34.38 ± 2.55 | 3.12 ± 0.00 | 50.00 ± 5.10 | 44.79 ± 1.47 | 4.17 ± 1.47 | 0.00 ± 0.00 | 47.40 ± 2.66 |
| `concat_c` | 84.38 ± 4.42 | 20.83 ± 1.47 | 0.00 ± 0.00 | 22.92 ± 5.89 | 64.58 ± 3.90 | 78.12 ± 5.10 | 47.92 ± 3.90 | 43.75 ± 3.53 |
| `flagship` | 67.71 ± 5.31 | 71.88 ± 0.00 | 10.42 ± 1.47 | 76.04 ± 1.47 | 64.58 ± 6.42 | 52.08 ± 1.47 | 43.75 ± 0.00 | 70.31 ± 3.29 |
| `from_scratch` | 69.79 ± 1.47 | 63.54 ± 1.47 | 0.00 ± 0.00 | 69.79 ± 1.47 | 47.92 ± 8.96 | 27.08 ± 2.95 | 19.79 ± 1.47 | 58.85 ± 4.54 |
| `no_kv_cache_backprop` | 88.54 ± 1.47 | 68.75 ± 0.00 | 8.33 ± 1.47 | 77.08 ± 3.90 | 52.08 ± 3.90 | 78.12 ± 0.00 | 56.25 ± 2.55 | 64.58 ± 2.76 |
| `no_player_attn_sf` | 12.50 ± 0.00 | 19.79 ± 2.95 | 0.00 ± 0.00 | 29.17 ± 1.47 | 71.88 ± 4.42 | 29.17 ± 3.90 | 6.25 ± 2.55 | 50.52 ± 2.33 |
| *real (GT)* | 100.00 ± 0.00 | 88.54 ± 1.47 | 93.75 ± 2.55 | 97.92 ± 1.47 | 96.88 ± 0.00 | 100.00 ± 0.00 | 98.96 ± 1.47 | 97.40 ± 0.74 |
