#!/usr/bin/env bash
# Fetch OpenAI's VPT inverse dynamics model, the one Matrix-Game's
# action-controllability numbers come from.
#
# Matrix-Game reports its IDM as trained on 1,962 hours of Minecraft with 90.6%
# keypress accuracy and R^2 = 0.97 on mouse movement. Those are VPT's published
# figures (Baker et al., 2022, arXiv:2206.11795), and OpenAI released the model,
# so we can run the same one rather than approximate it.
#
#   bash action_following/setup_vpt_idm.sh [target_dir]
#   export VPT_DIR=<target_dir>
#   python3 action_following/vpt_idm.py --validate
#
# About 2 GB of weights and a few minutes on a warm connection.

set -euo pipefail

TARGET="${1:-${VPT_DIR:-$PWD/.vpt}}"
WEIGHTS_URL=https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights
MODEL_URL=https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model
WEIGHTS_BYTES=1929358457

mkdir -p "$(dirname "$TARGET")"
if [ ! -d "$TARGET/.git" ]; then
  git clone --depth 1 https://github.com/openai/Video-Pre-Training.git "$TARGET"
fi
cd "$TARGET"

# The blob store does not honour range requests, so a truncated download cannot
# be resumed and has to be redone. Check the size rather than trusting the file.
if [ ! -f 4x_idm.weights ] || [ "$(stat -c%s 4x_idm.weights)" -ne "$WEIGHTS_BYTES" ]; then
  echo "downloading IDM weights (~1.9 GB)"
  curl -sSL --retry 5 --retry-all-errors -o 4x_idm.weights "$WEIGHTS_URL"
fi
[ -f 4x_idm.model ] || curl -sSL --retry 5 -o 4x_idm.model "$MODEL_URL"

got=$(stat -c%s 4x_idm.weights)
if [ "$got" -ne "$WEIGHTS_BYTES" ]; then
  echo "weights are $got bytes, expected $WEIGHTS_BYTES -- rerun" >&2
  exit 1
fi

# lib/actions.py imports minerl for one lookup table, MINERL_ITEM_MAP, used only
# by item_embed_id_to_name. Predicting buttons and camera never calls it, and
# installing real minerl means a Java and gradle build for a table we do not
# read. Stub it.
mkdir -p minerl/herobraine/hero
: > minerl/__init__.py
: > minerl/herobraine/__init__.py
: > minerl/herobraine/hero/__init__.py
cat > minerl/herobraine/hero/mc.py <<'PY'
"""Minimal stub. See setup_vpt_idm.sh for why this is safe."""

MINERL_ITEM_MAP = []
PY

echo "installing python deps"
pip install -q gym3 attrs "gym==0.26.2"

echo
echo "VPT IDM ready in $TARGET"
echo "  export VPT_DIR=$TARGET"
echo "  python3 action_following/vpt_idm.py --validate"
