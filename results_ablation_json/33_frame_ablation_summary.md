# 33-Frame Ablation Eval Results

Model: `33_frame_ablation_generations` (step 120000, 33 frames, 50 denoising steps)

GT dataset: `mc_multiplayer_v2_eval_new_sneak_combined`

VLM judge: `gemini-3-flash-preview`

Results averaged over 3 trials.

## Consistency (turnToLookEval + turnToLookOppositeEval)

| Task | Queries | Mean Accuracy | Std |
|---|---|---|---|
| turnToLookEval (expected: yes) | 32 | 34.38% | 2.55 |
| turnToLookOppositeEval (expected: no) | 32 | 77.08% | 2.95 |
| **Consistency (combined)** | **64** | **55.73%** | **1.95** |

## Translation (by perspective)

| Perspective | Queries | Mean Accuracy | Std |
|---|---|---|---|
| Moving bot | 32 | 84.38% | 0.00 |
| Non-moving bot | 32 | 25.00% | 0.00 |
| **Overall** | **64** | **54.69%** | **0.00** |
