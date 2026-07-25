#!/usr/bin/env bash
# Materialize results_json_late_episode_mixed_thinking/ by symlinking the
# already-correct cells from existing trees. The 7 (model, structureEval)
# cells are NOT created here -- they are produced by
# run_late_episode_mixed_thinking.sh and written directly into this folder.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DEST="results_json_late_episode_mixed_thinking"
SRC_OFF="results_json_late_episode"           # thinking OFF (per-handler defaults)
SRC_ON_GT="results_json_late_episode_thinking" # thinking ON GT-only run

MODELS=(causvid_dmd causvid_regression concat_c flagship from_scratch no_kv_cache_backprop no_player_attn_sf)

# eval -> recommended thinking
declare -A REC
REC[rotationEval]=OFF
REC[structureEval]=ON
REC[turnToLookEval]=ON
REC[turnToLookOppositeEval]=ON
REC[oneLooksAwayEval_long]=OFF
REC[bothLookAwayEval_long]=OFF
# translationEval is pinned to OLD; the comparison script reads it from
# results_json/generated directly, so we don't include it here.

mkdir -p "$DEST/generated" "$DEST/real"

# Helper: symlink src -> dest if src exists, refresh if dest is already a symlink.
link_dir() {
  local src="$1"
  local dst="$2"
  if [[ ! -d "$src" ]]; then
    echo "  SKIP (missing): $src"
    return
  fi
  rm -rf "$dst"
  ln -s "$ROOT/$src" "$dst"
  echo "  link: $dst -> $src"
}

echo "=== Generated (model x eval) ==="
for ev in "${!REC[@]}"; do
  rec="${REC[$ev]}"
  for m in "${MODELS[@]}"; do
    cell="${m}_${ev}"
    if [[ "$rec" == "OFF" ]]; then
      # Existing late_episode/generated already has thinking=OFF for these.
      link_dir "$SRC_OFF/generated/$cell" "$DEST/generated/$cell"
    elif [[ "$rec" == "ON" ]]; then
      # turnToLook* evals are thinking=ON via handler default -> the existing
      # late_episode tree already has them. structureEval has no
      # thinking-ON model run yet; we'll produce it via run_late_episode_mixed_thinking.sh.
      if [[ "$ev" == "structureEval" ]]; then
        echo "  skip (will be generated): $DEST/generated/$cell"
        continue
      fi
      link_dir "$SRC_OFF/generated/$cell" "$DEST/generated/$cell"
    fi
  done
done

echo
echo "=== Real (GT) ==="
for ev in "${!REC[@]}"; do
  rec="${REC[$ev]}"
  if [[ "$rec" == "OFF" || "$ev" == "turnToLookEval" || "$ev" == "turnToLookOppositeEval" ]]; then
    # turnToLook* GT was already thinking=ON in late_episode/real (handler default),
    # so we can pull it from there.
    link_dir "$SRC_OFF/real/$ev" "$DEST/real/$ev"
  elif [[ "$ev" == "structureEval" ]]; then
    # Use the dedicated thinking-ON GT run we just collected.
    link_dir "$SRC_ON_GT/real/$ev" "$DEST/real/$ev"
  fi
done

# README documenting the per-eval thinking config so the artifact is self-describing.
cat > "$DEST/README.md" <<'EOF'
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
EOF

echo
echo "=== Wrote $DEST/README.md ==="
ls -la "$DEST"
