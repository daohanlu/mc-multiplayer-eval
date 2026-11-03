#!/bin/bash
# Print NeurIPS 2026 Table 8 (test-split FVD) from saved I3D features.
# Table 8 reports the Mixed FVD: both bots pooled, q0+q2 (GT) vs q1+q3 (Gen).
#   bash fvd/fvd_table8.sh                      # archived 2026-05-06 features
#   FEATS=fvd/outputs_rw16 bash fvd/fvd_table8.sh   # features from a fresh run_rw16.sh
set -euo pipefail
cd "$(dirname "$0")"  # fvd/

FEATS=${FEATS:-/mnt/data-hdd/dl3957/neurips2026_eval/fvd/features_rw16}
PY=${PY:-/home/dl3957/miniconda3/envs/minecraft/bin/python}

# short_name : Table 8 row : value printed in the paper
ROWS=(
    "no_player_attn_sf:Independent:88.7"
    "v2_blk:Frame concat:113.2"
    "v2_scratch:Solaris w/o pretrain:102.0"
    "v2_flag:Solaris:81.3"
    "v2_reg:ODE Reg, Pre-DMD x, KV-BP v:151.9"
    "dmd_cont:Causal FT, Pre-DMD v, KV-BP v:131.8"
    "v2_nokv:Causal FT, Pre-DMD x, KV-BP x:76.9"
    "v2_flag:Causal FT, Pre-DMD x, KV-BP v (shaded = Solaris):88.7 (should be 81.3)"
)

printf "%-50s %10s  %s\n" "Table 8 row" "Mixed FVD" "paper"
for r in "${ROWS[@]}"; do
    IFS=: read -r short name paper <<< "$r"
    f=$FEATS/$short
    fvd=$($PY compute_fvd.py --features1 "$f/q0.npz" "$f/q2.npz" --features2 "$f/q1.npz" "$f/q3.npz" \
          | awk '/FVD =/ {printf "%.1f", $3}')
    printf "%-50s %10s  %s\n" "$name" "$fvd" "$paper"
done
