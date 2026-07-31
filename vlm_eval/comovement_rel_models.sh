#!/usr/bin/env bash
# Source of truth for the coMovementAlwaysRelativeMotionEval generations.
#
# Every model in the NeurIPS ablation set was re-run on this eval, each into its
# own bucket with the same `co_movement_rel/` subfolder. This file records which
# bucket belongs to which model, downloads them, and runs the VLM eval.
#
#   ./comovement_rel_models.sh list       # print the model -> bucket mapping
#   ./comovement_rel_models.sh download   # fetch anything not already local
#   ./comovement_rel_models.sh eval       # 3 trials per model, run concurrently
#   ./comovement_rel_models.sh eval concat_c from_scratch   # subset
#
# `eval` writes results_json_comovement/generated/{model}_coMovementAlwaysRelativeMotionEval/.
# Score everything, ground truth included, with `python3 score_comovement.py`.
#
# Model keys match build_vlm_tables.py, except that the flagship is `solaris`
# here because its results were collected under that name. The buckets also hold
# a `co_movement_divider_rel/` folder; it is not downloaded — the divider variant
# has an occlusion confound and is not reported (see COMOVEMENT_EVAL.md).

set -euo pipefail
cd "$(dirname "$0")"

DATASET=mc_multiplayer_v2_eval_new_sneak_combined_simplified_naming/coMovementAlwaysRelativeMotionEval
RESULTS_DIR=results_json_comovement
GEN_ROOT=generations_comovement
TRIALS=3

# model_key : gcs output dir : checkpoint restored by that run (from its run.log)
MODELS=(
    "solaris:gs://solaris-east5/outputs/neurips_eval_coMovement_rel:solaris.pt"
    "no_player_attn_sf:gs://solaris-east5/outputs/neurips_ablation_no_player_attn_sf_rel:no_player_attn_sf_1200.pt"
    "concat_c:gs://solaris-east5/outputs/neurips_ablation_concat_c_rel:concat_c_sf_final"
    "from_scratch:gs://solaris-east5/outputs/neurips_ablation_from_scratch_rel:from_scratch_sf_final"
    "causvid_regression:gs://solaris-east5/outputs/neurips_ablation_causvid_regression_rel:v2_flagship_fix_bidirectional_causal_regression_lr_1e-4_4000.pt"
    "causvid_dmd:gs://solaris-east5/outputs/neurips_ablation_causvid_dmd_rel:v2_flagship_fix_bidirectional_causal_dmd_5000.pt"
    "no_kv_cache_backprop:gs://solaris-east5/outputs/neurips_ablation_no_kv_cache_backprop_rel:flagship_no_bp_sf_final"
)

# The ground truth dataset itself.
GT_BUCKET=gs://solaris-central1/solaris/data/neurips_eval_coMovement/coMovementAlwaysRelativeMotionEval

selected_models() {
    if [ "$#" -eq 0 ]; then
        printf '%s\n' "${MODELS[@]}"
        return
    fi
    for want in "$@"; do
        local found=
        for entry in "${MODELS[@]}"; do
            [ "${entry%%:*}" = "$want" ] && { echo "$entry"; found=1; }
        done
        [ -n "$found" ] || { echo "unknown model: $want" >&2; exit 1; }
    done
}

cmd_list() {
    printf '%-22s %-58s %s\n' MODEL BUCKET CHECKPOINT
    for entry in "${MODELS[@]}"; do
        local key="${entry%%:*}" rest="${entry#*:}"
        printf '%-22s %-58s %s\n' "$key" "${rest%:*}" "${rest##*:}"
    done
    echo
    echo "ground truth: $GT_BUCKET"
}

cmd_download() {
    if [ ! -d "$DATASET/test" ]; then
        gsutil -m cp -r "$GT_BUCKET" "$(dirname "$DATASET")/"
    fi
    while IFS= read -r entry; do
        local key="${entry%%:*}" rest="${entry#*:}" bucket
        bucket="${rest%:*}"
        if [ -d "$GEN_ROOT/$key/co_movement_rel" ]; then
            echo "have $key ($(ls "$GEN_ROOT/$key/co_movement_rel" | wc -l) videos)"
            continue
        fi
        mkdir -p "$GEN_ROOT/$key"
        gsutil -m cp -r "$bucket/co_movement_rel" "$GEN_ROOT/$key/"
        echo "got $key ($(ls "$GEN_ROOT/$key/co_movement_rel" | wc -l) videos)"
    done < <(selected_models "$@")
}

cmd_eval() {
    mkdir -p logs
    local pids=() keys=()
    while IFS= read -r entry; do
        local key="${entry%%:*}"
        local subdir="$GEN_ROOT/$key/co_movement_rel"
        [ -d "$subdir" ] || { echo "missing generations for $key — run download first" >&2; exit 1; }
        echo "starting $key -> logs/comovement_rel_$key.log"
        python3 run_eval.py "$DATASET" \
            --generated-subdir "$subdir" --model-name "$key" \
            --num-trials "$TRIALS" --results-dir "$RESULTS_DIR" \
            > "logs/comovement_rel_$key.log" 2>&1 &
        pids+=($!)
        keys+=("$key")
    done < <(selected_models "$@")

    local status=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "done ${keys[$i]}"
        else
            echo "FAILED ${keys[$i]} — see logs/comovement_rel_${keys[$i]}.log" >&2
            status=1
        fi
    done
    return "$status"
}

case "${1:-list}" in
    list)     shift || true; cmd_list ;;
    download) shift; cmd_download "$@" ;;
    eval)     shift; cmd_eval "$@" ;;
    *) echo "usage: $0 {list|download|eval} [model ...]" >&2; exit 1 ;;
esac
