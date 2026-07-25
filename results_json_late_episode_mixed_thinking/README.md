# Late-episode evaluation, mixed-thinking configuration

This results tree applies the late-episode toggle (LATE_EPISODE_QUERY=1) on
top of the OLD evaluation pattern, with VLM "thinking" enabled per-eval
based on the GT-validation ablation in
`results_json_late_episode/comparison_late_vs_current.md`.

## Per-eval thinking configuration

| eval                       | thinking | source                                   |
|----------------------------|---------:|------------------------------------------|
| `translationEval`          | n/a      | pinned to OLD (`results_json/generated`) |
| `rotationEval`             | OFF      | `results_json_late_episode/...`          |
| `structureEval`            | **ON**   | freshly collected here (FORCE_VLM_THINKING=1) |
| `turnToLookEval`           | ON       | `results_json_late_episode/...` (handler default) |
| `turnToLookOppositeEval`   | ON       | `results_json_late_episode/...` (handler default) |
| `oneLooksAwayEval_long`    | OFF      | `results_json_late_episode/...`          |
| `bothLookAwayEval_long`    | OFF      | `results_json_late_episode/...`          |

`thinking_enabled` is recorded per-trial in every `trial_*.json` so the
config is verifiable post-hoc.

## Why this configuration

Empirical GT-late-episode ablation (3 trials × 32 episodes per eval, both
thinking modes):

| eval                       | NEW thinkOFF | NEW thinkON | chosen |
|----------------------------|-------------:|------------:|-------|
| `rotationEval`             | 88.54        | 83.33       | OFF   |
| `structureEval`            | 83.33        | **93.75**   | ON    |
| `turnToLookEval`           | 97.92        | 96.88       | ON*   |
| `turnToLookOppositeEval`   | 96.88        | **98.96**   | ON    |
| `oneLooksAwayEval_long`    | **100.00**   | 98.96       | OFF   |
| `bothLookAwayEval_long`    | **98.96**    | 92.71       | OFF   |

* turnToLookEval picks ON anyway since the per-handler default is ON and
  the OFF/ON deltas are within trial noise.

## Layout

- `generated/{model}_{eval}/`  - per (model, eval) trial outputs.
- `real/{eval}/`               - GT (ground truth video) trial outputs.
- Each subdir contains `trial_{1,2,3}.json` and `stats.json`.
