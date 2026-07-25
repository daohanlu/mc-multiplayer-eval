#!/usr/bin/env bash
# Same as validate_late_episode_gt.sh, but forces VLM thinking ON for every
# eval (via FORCE_VLM_THINKING=1) and writes to a sibling results dir so the
# original (no-thinking) GT artifacts under results_json_late_episode/real/
# are preserved for direct comparison.
#
# Hypothesis: the late-episode GT drops on rotation (88.54%) and structure
# (83.33%) include some VLM-stability noise on borderline frames. Enabling
# thinking might recover accuracy on the close-call cases (the same way
# turn-to-look already runs with thinking on).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

GEN_BASE="$ROOT/mc_multiplayer_v2_generations/flagship"
RESULTS_DIR="results_json_late_episode_thinking"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

EVALS=(
  "rotationEval:step_0001200_multiplayer_v2_eval_rotation_max_speed"
  "structureEval:step_0001200_multiplayer_v2_eval_structure_max_speed"
  "turnToLookEval:step_0001200_multiplayer_v2_eval_turn_to_look_max_speed"
  "turnToLookOppositeEval:step_0001200_multiplayer_v2_eval_turn_to_look_opposite_max_speed"
  "oneLooksAwayEval_long:step_0001200_multiplayer_v2_eval_one_looks_away_max_speed_long"
  "bothLookAwayEval_long:step_0001200_multiplayer_v2_eval_both_look_away_max_speed_long"
)

probe_frame_count() {
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
  log="$LOG_DIR/gt_late_episode_thinking_${ds}.log"

  echo "[$(date '+%F %T')] $ds [thinking=ON]: LATE_EPISODE_GEN_LEN=$gen_len -> $log"

  LATE_EPISODE_QUERY=1 LATE_EPISODE_GEN_LEN="$gen_len" FORCE_VLM_THINKING=1 \
    python -u run_eval.py "mc_multiplayer_v2_eval_new_sneak_combined/$ds" \
      --num-trials 3 \
      --results-dir "$RESULTS_DIR" \
      2>&1 | tee "$log"
done

echo "[$(date '+%F %T')] Done. GT late-episode (thinking ON) under $RESULTS_DIR/real/"
