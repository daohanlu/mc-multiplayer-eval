#!/usr/bin/env bash
# Validate that the late-episode toggle produces a well-posed question on
# real (GT) videos. For each eval we:
#   1. Probe the matching flagship-generated `video_0_side_by_side.mp4` for
#      its frame count (auto-discovered so the script keeps working if evals
#      are regenerated at different horizons).
#   2. Run `run_eval.py` against the GT dataset with LATE_EPISODE_QUERY=1 and
#      that probed LATE_EPISODE_GEN_LEN exported, writing into
#      results_json_late_episode/real/<eval>/.
#
# Sequential by design: a single API key is shared across these runs, so
# parallelizing tends to trip provider throttling. Wall time ~15-25 min.
#
# `translationEval` is intentionally excluded (the dataset reverses bot
# actions in the latter half of the episode, so the late-horizon question
# is ill-posed there — it stays pinned to its OLD baseline in the report).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

GEN_BASE="$ROOT/mc_multiplayer_v2_generations/flagship"
RESULTS_DIR="results_json_late_episode"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# eval_name : flagship-generated subdir under $GEN_BASE
EVALS=(
  "rotationEval:step_0001200_multiplayer_v2_eval_rotation_max_speed"
  "structureEval:step_0001200_multiplayer_v2_eval_structure_max_speed"
  "turnToLookEval:step_0001200_multiplayer_v2_eval_turn_to_look_max_speed"
  "turnToLookOppositeEval:step_0001200_multiplayer_v2_eval_turn_to_look_opposite_max_speed"
  "oneLooksAwayEval_long:step_0001200_multiplayer_v2_eval_one_looks_away_max_speed_long"
  "bothLookAwayEval_long:step_0001200_multiplayer_v2_eval_both_look_away_max_speed_long"
)

probe_frame_count() {
  # Print the frame count of the given mp4 via cv2.
  python - "$1" <<'PY'
import sys
import cv2
cap = cv2.VideoCapture(sys.argv[1])
if not cap.isOpened():
    sys.exit(f"cv2 could not open {sys.argv[1]}")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
if n <= 0:
    sys.exit(f"cv2 returned non-positive frame count for {sys.argv[1]}: {n}")
print(n)
PY
}

for entry in "${EVALS[@]}"; do
  ds="${entry%%:*}"
  subdir="${entry#*:}"
  video="$GEN_BASE/$subdir/video_0_side_by_side.mp4"

  if [[ ! -f "$video" ]]; then
    echo "[$(date '+%F %T')] SKIP $ds: missing probe video $video" >&2
    continue
  fi

  gen_len="$(probe_frame_count "$video")"
  log="$LOG_DIR/gt_late_episode_${ds}.log"

  echo "[$(date '+%F %T')] $ds: LATE_EPISODE_GEN_LEN=$gen_len (probed from $subdir/video_0_side_by_side.mp4) -> $log"

  LATE_EPISODE_QUERY=1 LATE_EPISODE_GEN_LEN="$gen_len" \
    python -u run_eval.py "mc_multiplayer_v2_eval_new_sneak_combined/$ds" \
      --num-trials 3 \
      --results-dir "$RESULTS_DIR" \
      2>&1 | tee "$log"
done

echo "[$(date '+%F %T')] Done. GT late-episode results under $RESULTS_DIR/real/"
echo "Next: python compare_late_vs_current.py  # appends GT validation table"
