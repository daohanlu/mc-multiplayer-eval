# Minecraft Multiplayer VLM Evaluation Framework

A framework for evaluating video generation models on Minecraft multiplayer scenarios using Vision Language Models (VLMs). The evaluation measures whether generated videos correctly preserve multiplayer interactions that were present in the ground-truth training data.

---

# Part 1: Usage

## Quick Start

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key"

# Download the eval dataset from GCloud Bucket
gsutil -m cp -r gs://YOUR_BUCKET_NAME/.../mc_multiplayer_v2_eval_max_speed_new_sneak_max_speed mc_multiplayer_v2_eval_max_speed_max_speed

# Run evaluation on ground-truth videos
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval

# Run evaluation on generated videos
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval --generated generations/flagship_final

# Run structure evaluation
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/structureEval --generated generations/flagship_final
```

## Environment Setup

```bash
# Required
export GEMINI_API_KEY="your-api-key"

# Install dependencies
pip install google-genai opencv-python
```

## Command Reference

```
python run_eval.py <folder> [options]

Arguments:
  folder                    Path to dataset folder (e.g., mc_multiplayer_v2_eval_max_speed/turnToLookEval)
                            The /test subdirectory is automatically appended.

Options:
  --dry-run                 View keyframe info without VLM queries (no cost)
  --extract-frames          Extract frames to frame_extraction/ folder for inspection
  --generated PATH          Path to generated videos directory
  --limit N                 Limit number of episodes/queries (default: 32 video pairs)
  --api-key KEY             Gemini API key (or use GEMINI_API_KEY env var)
  --summary-json PATH       Custom path to structure summary JSON (structure datasets only)
  -o, --output PATH         Custom output path (default: auto-organized)
```

## Available Datasets

| Dataset | Description | Expected Answer |
|---------|-------------|-----------------|
| `turnToLookEval` | Both players turn to look at each other | yes |
| `turnToLookOppositeEval` | Players do NOT look at each other | no |
| `translationEval` | Detects player movement direction | forward/backward/left/right |
| `rotationEval` | Detects camera rotation direction | left/right |
| `oneLooksAwayEval` | One player looks away and back | yes |
| `bothLookAwayEval` | Both players look away and back | yes |
| `structureEval` | Structure is built and visible | yes |
| `structureNoPlaceEval` | Structure NOT built - should not be visible | no |

## Common Usage Examples

### Dry Run (Inspect Without VLM Queries)
```bash
# View keyframe detection info for first 10 episodes (no API cost)
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval --dry-run --limit 10
```

### Extract Frames for Visual Inspection
```bash
# Extract frames to frame_extraction/ folder
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval --extract-frames --limit 5
```

### Run Full Evaluation
```bash
# Evaluate ground-truth videos (sanity check)
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval

# Evaluate generated videos
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval --generated generations/flagship_final
```

### Batch Evaluation (All Models)
```bash
python run_all_evals.py
```

This scans `generations/` for model folders and runs evaluations for each model/dataset combination. Configure which datasets to evaluate by editing `ENABLED_EVAL_TYPES` in `run_all_evals.py`.

## Output Location

Results are automatically organized:
- **Ground-truth evaluations**: `results_json/real/{dataset_name}.json`
- **Generated video evaluations**: `results_json/generated/{model_name}_{dataset_name}.json`

## Result Format

```json
{
  "vlm_model_name": "gemini-3-flash-preview",
  "our_model_name": "flagship_final",
  "thinking_enabled": false,
  "total_queries": 64,
  "unclear_count": 2,
  "unclear_percentage": 3.12,
  "evaluable_queries": 62,
  "correct": 58,
  "accuracy_excluding_unclear": 93.55,
  "accuracy_total": 90.62,
  "breakdown_by_query_type": {...},
  "results": [...]
}
```

---

# Part 2: Implementation

## Code Flow Overview

When you run:
```bash
python run_eval.py ./mc_multiplayer_v2_eval_max_speed/turnToLookEval --generated generations/flagship_final --limit 10
```

The execution flows through these steps:

### 1. Argument Parsing (`main()`)
- Parses command-line arguments
- Automatically appends `/test` to the dataset path (e.g., `turnToLookEval` → `turnToLookEval/test`)
- Extracts the dataset name from the path for handler identification

### 2. Handler Identification (`identify_handler()`)
- Matches the dataset name against each handler's `DATASET_NAMES` attribute
- Returns an instance of the appropriate handler class
- For structure handlers (`structureEval`, `structureNoPlaceEval`), automatically loads the required ground-truth summary JSON from `assets/hard_coded_gt/`

### 3. Video Pair Discovery (`find_mc_video_pairs()`)
- Scans the dataset folder for video files matching the pattern:
  ```
  {episode}_{Alpha|Bravo}_instance_{instance}_camera.mp4
  ```
- Groups matching files into `VideoPair` objects containing:
  - `alpha_video`, `bravo_video`: Paths to the video files
  - `alpha_json`, `bravo_json`: Paths to corresponding metadata JSON files
  - `episode_num`, `instance_num`: Identifiers for the episode
- Returns pairs sorted by episode and instance number
- **Default limit**: 32 video pairs (override with `--limit`)

### 4. Generated Video Matching (if `--generated` is provided)
- Calls `find_generated_video_subdir()` to locate the correct subdirectory within the generations folder
- Generated videos are named `video_{N}_side_by_side.mp4` where N corresponds to the video pair index
- The generated video is a 1280x720 composite with 4 quadrants:
  - **Top-left**: Alpha ground-truth
  - **Top-right**: Alpha generated
  - **Bottom-left**: Bravo ground-truth  
  - **Bottom-right**: Bravo generated

### 5. Keyframe Extraction (`handler.extract_keyframes()`)
- Each handler implements its own keyframe extraction logic
- Reads the JSON metadata files to determine:
  - Which frames contain the relevant action/event
  - Which bot's perspective to evaluate from
  - What the expected answer should be
- Returns a list of `KeyframeQuery` objects with:
  - `video_path`: Which video to extract from
  - `frame_index`: Which frame to evaluate
  - `expected_answer`: The correct answer for validation
  - `metadata`: Additional context (variant, frame indices, etc.)

### 6. VLM Evaluation Loop (`run_evaluation()`)
For each keyframe query:
1. **Extract frame(s)** using `extract_frame()` or `extract_frame_from_generated()`
   - Ground-truth: Extracts directly from the video file
   - Generated: Extracts from the appropriate quadrant (top-right for alpha, bottom-right for bravo)
2. **Query VLM** using `query_vlm()` with the handler's prompt and extracted frame(s)
3. **Validate response** using `handler.validate_response()`
4. **Record result** with correctness and metadata

### 7. Result Saving (`save_results()`)
- Saves comprehensive JSON with:
  - Overall statistics (accuracy, unclear count, etc.)
  - Per-query breakdown (query type if applicable)
  - Individual result details
- Auto-organized output paths:
  - Ground-truth: `results_json/real/{dataset_name}.json`
  - Generated: `results_json/generated/{model_name}_{dataset_name}.json`

## Handler Architecture

Each handler extends `EpisodeTypeHandler` and implements:

| Method/Attribute | Description |
|------------------|-------------|
| `DATASET_NAMES` | List of exact dataset folder names this handler supports |
| `get_prompt()` | Returns the VLM prompt for evaluation |
| `extract_keyframes(video_pair)` | Determines which frames to evaluate and returns `KeyframeQuery` objects |
| `validate_response(response, expected)` | Compares VLM response to expected answer |
| `enable_vlm_thinking` | Whether to enable VLM thinking mode (default: False) |

Available handlers:
- `MinecraftTurnToLookHandler`
- `MinecraftTurnToLookOppositeHandler`
- `MinecraftTranslationHandler`
- `MinecraftRotationHandler`
- `MinecraftLooksAwayHandler`
- `MinecraftBothLookAwayHandler`
- `MinecraftStructureBuildingHandler`
- `MinecraftStructureNoPlaceHandler`

## Structure Evaluation: Special Requirements

The `structureEval` and `structureNoPlaceEval` datasets require hard-coded ground-truth files because the structure type (wall, tower, square) and which bot builds are **randomly selected during data generation**.

### How Structure GT Works

1. **During data collection**, logs are generated that record:
   - Which bot (Alpha or Bravo) performed the building action
   - What structure type was randomly selected (wall_4x1, tower_2x1, wall_2x2)

2. **`parse_structure_logs.py`** parses these logs to create:
   - `assets/hard_coded_gt/structure_building_summary.json` (for structureEval)
   - `assets/hard_coded_gt/structure_building_no_place_summary.json` (for structureNoPlaceEval)

3. **The JSON format** maps instance → episode → builder/structure info:
   ```json
   {
     "instance_0": {
       "episode_0": {
         "builder": "alpha",
         "structure": "wall_4x1",
         "alpha_structure": "wall_4x1",
         "alpha_builds": true,
         "bravo_structure": "wall_4x1", 
         "bravo_builds": false
       }
     }
   }
   ```

4. **During evaluation**, the structure handler:
   - Looks up which bot is the observer (non-builder)
   - Extracts frames from the observer's perspective
   - Asks the VLM if a structure is visible

### Important: Episode Order Consistency

The ground-truth JSON files assume episodes are processed in **sorted order by episode number and instance**. If the episodes in your dataset are shuffled or reordered, the structure type lookups will be incorrect. The `find_mc_video_pairs()` function ensures this by returning pairs sorted by `(episode_num, instance_num)`.

### Regenerating Structure GT Files

If you need to regenerate the structure ground-truth files from new logs:

```bash
# For structureEval
python parse_structure_logs.py /path/to/structureEval/logs_directory

# For structureNoPlaceEval  
python parse_structure_logs.py /path/to/structureNoPlaceEval/logs_directory
```

The script automatically detects the eval type from log contents and outputs to `assets/hard_coded_gt/`.

## Video Format

### Ground-Truth Videos
```
{episode}_{Alpha|Bravo}_instance_{instance}_camera.mp4  # Video file
{episode}_{Alpha|Bravo}_instance_{instance}.json        # Metadata (frame-by-frame state)
```

### Generated Videos (Side-by-Side)
```
video_{N}_side_by_side.mp4   # 1280x720, 4 quadrants
```

Layout:
```
┌─────────────────┬──────────────────┐
│  Alpha GT       │  Alpha Generated │
│  (top-left)     │  (top-right)     │
├─────────────────┼──────────────────┤
│  Bravo GT       │  Bravo Generated │
│  (bottom-left)  │  (bottom-right)  │
└─────────────────┴──────────────────┘
```

## Project Structure

```
├── run_eval.py                  # Main entry point for all evaluations
├── vlm_utils.py                 # Core utilities (VLM queries, frame extraction, data classes)
├── run_all_evals.py             # Batch evaluation across all models
├── parse_structure_logs.py      # Parses structure building logs to create GT files
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
│   ├── mc_multiplayer_handler_turn_to_look_opposite.py
│   └── camera_utils.py
│
├── assets/
│   └── hard_coded_gt/           # Hard-coded ground truth for structure evaluations
│       ├── structure_building_summary.json
│       └── structure_building_no_place_summary.json
│
├── mc_multiplayer_v2_eval_max_speed_*/    # Ground-truth video datasets
│   └── {evalType}/
│       └── test/
│           ├── {episode}_{Alpha|Bravo}_instance_{instance}_camera.mp4
│           └── {episode}_{Alpha|Bravo}_instance_{instance}.json
│
├── generations/                 # Generated videos from models
│   └── {model_name}/            # e.g., SF_CONCAT_FINAL_2
│       └── step_{N}_multiplayer_v2_eval_{type}/
│           └── video_{N}_side_by_side.mp4
│
├── results_json/                # Evaluation results
│   ├── real/                    # Results for ground-truth videos
│   └── generated/               # Results for generated videos
│
└── frame_extraction/            # Extracted frames (from --extract-frames)
```
