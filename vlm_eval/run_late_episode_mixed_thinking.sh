#!/usr/bin/env bash
# Run structureEval with FORCE_VLM_THINKING=1 against each of the 7 model
# generation trees, writing into results_json_late_episode_mixed_thinking/.
# These are the 7 missing cells identified by the inventory; everything
# else in that folder is already in place via symlinks (see
# build_mixed_thinking_dir.sh).
#
# Sequential because we share a single API key. Wall-time ~35-50 min total
# (thinking-mode calls are ~1.5x slower than non-thinking; 7 x ~6 min).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

RESULTS_DIR="results_json_late_episode_mixed_thinking"
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

EVAL="structureEval"

for m in "${MODELS[@]}"; do
  gen_dir="mc_multiplayer_v2_generations/$m"
  if [[ ! -d "$gen_dir" ]]; then
    echo "[$(date '+%F %T')] SKIP $m: missing $gen_dir" >&2
    continue
  fi

  log="$LOG_DIR/late_episode_mixed_thinking_${m}_${EVAL}.log"
  echo "[$(date '+%F %T')] $m / $EVAL [thinking=ON] -> $log"

  LATE_EPISODE_QUERY=1 FORCE_VLM_THINKING=1 \
    python -u "$SCRIPT_DIR/run_eval.py" "mc_multiplayer_v2_eval_new_sneak_combined/$EVAL" \
      --num-trials 3 \
      --generated "$gen_dir" \
      --results-dir "$RESULTS_DIR" \
      2>&1 | tee "$log"
done

echo "[$(date '+%F %T')] Done. Mixed-thinking structureEval results under $RESULTS_DIR/generated/"
