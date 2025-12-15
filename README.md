# Minecraft Multiplayer VLM Evaluation Framework

A framework for evaluating video generation models on Minecraft multiplayer scenarios using Vision Language Models (VLMs).

## Overview

This framework evaluates whether video generation models correctly simulate multiplayer interactions by:
1. Extracting keyframes from ground-truth or generated videos
2. Querying a VLM (Gemini) to analyze the frames
3. Comparing VLM responses against expected answers

## Quick Start

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key"

# Run evaluation on ground-truth videos
python run_eval.py mc_multiplayer_v2_eval/turnToLookEval

# Run evaluation on generated videos
python run_eval.py mc_multiplayer_v2_eval/turnToLookEval --generated generations/your_model_name
```

## Usage

### Basic Commands

```bash
# View keyframe detection info (no VLM queries, no cost)
python run_eval.py <dataset_path> --dry-run [--limit N]

# Extract frames for visual inspection
python run_eval.py <dataset_path> --extract-frames [--limit N]

# Run full VLM evaluation
python run_eval.py <dataset_path> [--limit N]

# Evaluate generated videos
python run_eval.py <dataset_path> --generated generations/<model_name>
```

**Note:** The `/test` subdirectory is automatically appended to the dataset path.

### Available Datasets

| Dataset | Handler | Description |
|---------|---------|-------------|
| `turnToLookEval` | MinecraftTurnToLookHandler | Both players turn to look at each other (expected: yes) |
| `turnToLookOppositeEval` | MinecraftTurnToLookOppositeHandler | Players do NOT look at each other (expected: no) |
| `structureNoPlaceEval` | MinecraftStructureNoPlaceHandler | Structure NOT built - no structure visible (expected: no) |
| `mc_multiplayer_eval_translation` | MinecraftTranslationHandler | Detects player movement (forward/backward/left/right) |
| `mc_multiplayer_eval_rotation` | MinecraftRotationHandler | Detects camera rotation direction |
| `mc_multiplayer_eval_structure` | MinecraftStructureBuildingHandler | Structure building evaluation (expected: yes) |
| `mc_multiplayer_eval_looks_away` | MinecraftLooksAwayHandler | One player looks away and back |
| `mc_multiplayer_eval_both_look_away` | MinecraftBothLookAwayHandler | Both players look away and back |

**Note:** By default, evaluation is limited to the first 32 video pairs. Use `--limit N` to override.

### Output Organization

Results are automatically organized:
- **Ground-truth evaluations**: `results_json/real/{dataset_name}.json`
- **Generated video evaluations**: `results_json/generated/{model_name}_{dataset_name}.json`

## Batch Evaluation

To run evaluations across all models:

```bash
python run_all_evals.py
```

This script:
1. Scans the `generations/` directory for model folders
2. Identifies which eval types each model has generated videos for
3. Runs `run_eval.py` for each model/dataset combination

Configure which datasets to evaluate by editing `eval_type_mapping` in `run_all_evals.py`.

## Project Structure

```
├── run_eval.py                  # Main evaluation script
├── run_all_evals.py             # Batch evaluation across all models
├── vlm_utils.py                 # Core utilities (VLM queries, frame extraction)
│
├── handlers/                    # Episode type handlers
│   ├── __init__.py
│   ├── mc_multiplayer_handler_translation.py
│   ├── mc_multiplayer_handler_rotation.py
│   ├── mc_multiplayer_handler_looks_away.py
│   ├── mc_multiplayer_handler_both_look_away.py
│   ├── mc_multiplayer_handler_structure.py
│   ├── mc_multiplayer_handler_structure_no_place.py
│   ├── mc_multiplayer_handler_turn_to_look.py
│   └── mc_multiplayer_handler_turn_to_look_opposite.py
│
├── mc_multiplayer_eval_*/test/  # Ground-truth video datasets
├── generations/                 # Generated videos from models
├── results_json/                # Evaluation results
│   ├── real/                    # Results for ground-truth videos
│   └── generated/               # Results for generated videos
└── frame_extraction/            # Extracted frames (from --extract-frames)
```

## Video Format

### Ground-Truth Videos
```
{episode}_{Alpha|Bravo}_instance_{instance}_camera.mp4
{episode}_{Alpha|Bravo}_instance_{instance}.json
```

### Generated Videos
```
generations/{model_name}/{eval_type}_*/video_{N}_side_by_side.mp4
```

Generated videos are 1280x720 with 4 quadrants:
- Top-left: Alpha ground-truth
- Top-right: Alpha generated
- Bottom-left: Bravo ground-truth
- Bottom-right: Bravo generated

## Handler Architecture

Each handler extends `EpisodeTypeHandler` and implements:
- `DATASET_NAMES`: List of exact dataset folder names this handler supports
- `get_prompt()`: VLM prompt for the evaluation
- `extract_keyframes()`: Logic to identify which frames to evaluate
- `validate_response()`: Compare VLM response to expected answer

## Environment Setup

```bash
# Required
export GEMINI_API_KEY="your-api-key"

# Optional: Pass API key directly
python run_eval.py mc_multiplayer_v2_eval/turnToLookEval --api-key "your-api-key"
```

## Command Reference

```
python run_eval.py <folder> [options]

Arguments:
  folder                    Path to dataset folder (e.g., mc_multiplayer_v2_eval/turnToLookEval)
                            The /test subdirectory is automatically appended.

Options:
  --dry-run                 View keyframe info without VLM queries
  --extract-frames          Extract frames to frame_extraction/ folder
  --generated PATH          Path to generated videos directory
  --limit N                 Limit number of episodes/queries
  --api-key KEY             Gemini API key (or use GEMINI_API_KEY env var)
  --summary-json PATH       Path to structure_building_summary.json (structure dataset only)
  -o, --output PATH         Custom output path (default: auto-organized)
```

## Result Format

```json
{
  "total_queries": 100,
  "unclear_count": 5,
  "unclear_percentage": 5.0,
  "evaluable_queries": 95,
  "correct": 80,
  "accuracy_excluding_unclear": 84.21,
  "accuracy_total": 80.0,
  "results": [
    {
      "video": "000001_Alpha_instance_000_camera.mp4",
      "frame_index": 45,
      "expected": "forward",
      "response": "forward",
      "correct": true,
      "unclear": false,
      "metadata": {...}
    }
  ]
}
```
