#!/usr/bin/env bash
# Run oneLooksAwayEval_long under the late-episode toggle with thinking ON
# for each of the 7 model generation trees, writing into
# results_json_late_episode_thinking/generated/{model}_oneLooksAwayEval_long/.
#
# This complements the existing late-episode thinking-OFF runs at
# results_json_late_episode/generated/{model}_oneLooksAwayEval_long/ so we can
# directly compare model accuracy at thinking ON vs OFF for this eval.
#
# Sequential because we share a single API key. Wall time ~60-80 min total
# (oneLooksAway has ~64 queries x 3 trials = 192 thinking calls per model;
# 7 models x ~9 min each).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

RESULTS_DIR="results_json_late_episode_thinking"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MODELS=(
  causvid_dmd
  causvid_regression
  concat_c
  flagship
  from_scratch
  no_kv_cache_backprop
  no_player_attn_sf
)

EVAL="oneLooksAwayEval_long"

for m in "${MODELS[@]}"; do
  gen_dir="mc_multiplayer_v2_generations/$m"
  if [[ ! -d "$gen_dir" ]]; then
    echo "[$(date '+%F %T')] SKIP $m: missing $gen_dir" >&2
    continue
  fi

  log="$LOG_DIR/late_episode_thinking_${m}_${EVAL}.log"
  echo "[$(date '+%F %T')] $m / $EVAL [thinking=ON] -> $log"

  LATE_EPISODE_QUERY=1 FORCE_VLM_THINKING=1 \
    python -u run_eval.py "mc_multiplayer_v2_eval_new_sneak_combined/$EVAL" \
      --num-trials 3 \
      --generated "$gen_dir" \
      --results-dir "$RESULTS_DIR" \
      2>&1 | tee "$log"
done

echo "[$(date '+%F %T')] Done. Results under $RESULTS_DIR/generated/"
