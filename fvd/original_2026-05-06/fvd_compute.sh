#!/bin/bash
# FVD computation — CPU only, run locally after feature extraction is done.
# Produces 3 FVD values per subdir: bot1, bot2, mixed (both bots).
set -euo pipefail
cd /scratch/$USER/solaris_fvd

PY=.venv/bin/python

for short in dmd_cont v2_blk v2_flag v2_nokv v2_reg v2_scratch no_player_attn_sf; do
    echo "========================================"
    echo "$short"
    echo "========================================"

    echo "--- Bot1 FVD (q0 vs q1) ---"
    $PY compute_fvd.py --features1 outputs/$short/q0.npz --features2 outputs/$short/q1.npz

    echo "--- Bot2 FVD (q2 vs q3) ---"
    $PY compute_fvd.py --features1 outputs/$short/q2.npz --features2 outputs/$short/q3.npz

    echo "--- Mixed FVD (q0+q2 vs q1+q3) ---"
    $PY compute_fvd.py --features1 outputs/$short/q0.npz outputs/$short/q2.npz --features2 outputs/$short/q1.npz outputs/$short/q3.npz

    echo ""
done
