#!/bin/bash
# Random-window 16-frame FVD extraction across all 7 models, 4 quadrants each.
# This is the run behind Table 8 of the NeurIPS 2026 submission (2026-05-06).
# Two chains (one per GPU), 14 jobs per chain, ~63 min total wall:
#   GPU=0 QUADS='0 1' bash fvd/run_rw16.sh
#   GPU=1 QUADS='2 3' bash fvd/run_rw16.sh
# Then: bash fvd/fvd_table8.sh
#
# Paths default to the HDD archive (see fvd/README.md) and can be overridden.
# The arguments to compute_features.py are exactly those of the 2026-05-06 run.
set -euo pipefail
cd "$(dirname "$0")"  # fvd/

ARCHIVE=${ARCHIVE:-/mnt/data-hdd/dl3957/neurips2026_eval/fvd}
PY=${PY:-/home/dl3957/miniconda3/envs/minecraft/bin/python}
BASE=${BASE:-$ARCHIVE/test_set_generations}
I3D=${I3D:-$ARCHIVE/models/i3d_torchscript.pt}
OUT=${OUT:-outputs_rw16}          # use a new dir; the archived features are in $ARCHIVE/features_rw16
LOGDIR=${LOGDIR:-$OUT/logs}

# short_name : data_subpath
MODELS=(
    "dmd_cont:flagship_fix_bidirectional_sf_dmd_continue"
    "v2_blk:v2_concat_c_fix_bidirectional_fix_blocksize_sf"
    "v2_flag:v2_flagship_fix_bidirectional_sf"
    "v2_nokv:v2_flagship_fix_bidirectional_sf_no_kv_backprop"
    "v2_reg:v2_flagship_fix_bidirectional_sf_regression_lr_1e-4_continue"
    "v2_scratch:v2_flagship_from_scratch_fix_bidirectional_sf"
    "no_player_attn_sf:no_player_attn_sf/step_0001200_duet_2048"
)

GPU="${GPU:?must set GPU=0 or GPU=1}"
QUADS="${QUADS:?must set QUADS, e.g. '0 1' or '2 3'}"

mkdir -p "$LOGDIR"
for entry in "${MODELS[@]}"; do
    short="${entry%%:*}"
    data="${entry##*:}"
    mkdir -p "$OUT/$short"
    for q in $QUADS; do
        log="$LOGDIR/rw16_${short}_q${q}.log"
        out="$OUT/$short/q${q}.npz"
        if [[ -s "$out" ]]; then
            echo "[skip] $out exists"
            continue
        fi
        echo "[$(date +%H:%M:%S)] GPU $GPU  $short q$q -> $out"
        CUDA_VISIBLE_DEVICES=$GPU $PY compute_features.py \
            --data_dir "$BASE/$data" \
            --quadrant $q --output "$out" --i3d_weights "$I3D" \
            --sample_mode random_window --num_frames 16 --seed 42 \
            --batch_size 32 --num_workers 8 > "$log" 2>&1
    done
done
echo "[$(date +%H:%M:%S)] GPU $GPU chain done"
