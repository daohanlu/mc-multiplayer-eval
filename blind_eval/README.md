# Blind Evaluation Tool

A tool for blind human evaluation of turn-to-look tasks, comparing human judgments against VLM results.

## Quick Start

### 1. Prepare the evaluation

```bash
python blind_eval_tool.py prepare --models flagship no_kv_cache_backprop
```

This creates:
- `blind_eval/cropped_images/` - Generated perspectives only (no GT, no expected answer)
- `blind_eval/blind_eval.html` - Self-contained HTML interface with embedded images
- `blind_eval/question_mapping.json` - Maps shuffled IDs to original metadata

### 2. Run the evaluation

**Option A: Local browser**
Simply open `blind_eval/blind_eval.html` in your browser.

**Option B: Remote server (e.g., via SSH)**
```bash
cd blind_eval
python -m http.server 8080
```
Then access `http://<your-server>:8080/blind_eval.html` in your browser.

> The HTML is self-contained with all images embedded as base64, so `python -m http.server` works fine - no special backend needed.

### 3. Answer questions

- **Y** = Yes (perspectives show same scenery)
- **N** = No (perspectives show different scenery)
- **S** = Skip
- **A/D** or arrow keys = Navigate between questions

### 4. Important: Progress & Saving

- **Progress is saved automatically** to your browser's `localStorage` after each answer
- You can close the tab and return later - your progress will be restored
- **localStorage is browser-specific** - if you switch browsers or clear data, progress is lost

**When finished:**
1. Click **"Download Results"** to save `human_answers_YYYY-MM-DD.json`
2. This JSON file is your permanent record - keep it safe!

### 5. Compare against VLM

```bash
python compare_human_vlm.py blind_eval/human_answers_2026-01-26.json

# Compare against a different VLM trial
python compare_human_vlm.py blind_eval/human_answers_2026-01-26.json --trial 2
```

## File Reference

| File | Purpose |
|------|---------|
| `blind_eval_tool.py` | Prepare evaluation & verify answers |
| `compare_human_vlm.py` | Compare human vs VLM answers |
| `blind_eval.html` | Annotation interface |
| `question_mapping.json` | Maps question IDs to metadata |
| `human_answers_*.json` | Your saved answers (download from webpage) |
