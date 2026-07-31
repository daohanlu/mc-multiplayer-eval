SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/run_eval.py" mc_multiplayer_v2_eval_new_sneak_combined/translationEval --num-trials 3
python "$SCRIPT_DIR/run_eval.py" mc_multiplayer_v2_eval_new_sneak_combined/rotationEval --num-trials 3
python "$SCRIPT_DIR/run_eval.py" mc_multiplayer_v2_eval_new_sneak_combined/structureEval --num-trials 3
python "$SCRIPT_DIR/run_eval.py" mc_multiplayer_v2_eval_new_sneak_combined/turnToLookEval --num-trials 3
python "$SCRIPT_DIR/run_eval.py" mc_multiplayer_v2_eval_new_sneak_combined/turnToLookOppositeEval --num-trials 3
python "$SCRIPT_DIR/run_eval.py" mc_multiplayer_v2_eval_new_sneak_combined/oneLooksAwayEval --num-trials 3
python "$SCRIPT_DIR/run_eval.py" mc_multiplayer_v2_eval_new_sneak_combined/bothLookAwayEval --num-trials 3
