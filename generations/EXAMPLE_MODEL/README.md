# Example Model Directory Structure

This is an example showing the expected folder structure for generated videos.

## Directory Layout

```
generations/
└── YOUR_MODEL_NAME/
    └── step_{N}_multiplayer_v2_eval_{type}/
        ├── video_0_side_by_side.mp4
        ├── video_1_side_by_side.mp4
        ├── video_2_side_by_side.mp4
        └── ...
```

## Expected Eval Types

Each subdirectory should match one of these patterns:
- `*_eval_translation`
- `*_eval_rotation`
- `*_eval_structure`
- `*_eval_turn_to_look`
- `*_eval_turn_to_look_opposite`
- `*_eval_one_looks_away`
- `*_eval_both_look_away`

## Video Format

Each `video_{N}_side_by_side.mp4` should be a 1280x720 composite with 4 quadrants:
- Top-left: Alpha ground-truth
- Top-right: Alpha generated
- Bottom-left: Bravo ground-truth
- Bottom-right: Bravo generated

The video index N corresponds to the sorted video pair index from the ground-truth dataset.

## Usage

1. Copy your model's generated videos into a new folder under `generations/`
2. Run evaluations:
   ```bash
   # Single model/dataset
   python run_eval.py mc_multiplayer_v2_eval/translationEval --generated generations/YOUR_MODEL_NAME
   
   # All models and datasets
   python run_all_evals.py
   ```
