#!/bin/bash
set -e

SRC_BASE="gs://solaris-central1/output/no_player_attn_sf"
DST_BASE="mc_multiplayer_v2_generations/no_player_attn_sf"

declare -A MAPPING=(
    ["step_0001200_eval_both_look_away"]="step_0001200_multiplayer_v2_eval_both_look_away_max_speed_long"
    ["step_0001200_eval_one_looks_away"]="step_0001200_multiplayer_v2_eval_one_looks_away_max_speed_long"
    ["step_0001200_eval_rotation"]="step_0001200_multiplayer_v2_eval_rotation_max_speed"
    ["step_0001200_eval_structure"]="step_0001200_multiplayer_v2_eval_structure_max_speed"
    ["step_0001200_eval_translation"]="step_0001200_multiplayer_v2_eval_translation"
    ["step_0001200_eval_turn_to_look"]="step_0001200_multiplayer_v2_eval_turn_to_look_max_speed"
    ["step_0001200_eval_turn_to_look_opposite"]="step_0001200_multiplayer_v2_eval_turn_to_look_opposite_max_speed"
)

for src in "${!MAPPING[@]}"; do
    dst="${MAPPING[$src]}"
    echo "==> Downloading $src -> $dst"
    mkdir -p "$DST_BASE/$dst"
    gcloud storage cp -r "$SRC_BASE/$src/*" "$DST_BASE/$dst/"
done

echo "Done."
