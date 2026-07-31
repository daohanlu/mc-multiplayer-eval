"""
Replay May 1 GT structureEval trial inputs against the gemini-3-flash-preview
endpoint TODAY, frame-for-frame and prompt-for-prompt.

Goal: rule out everything except endpoint drift. We decode the SAME
source mp4 at the SAME frame_index that the May 1 run used, send the SAME
prompt with thinking_enabled=False, and compare today's response against
May 1's recorded response.

Usage:
    GEMINI_API_KEY=... python -u replay_drift_check.py \\
        --baseline results_json/real/structureEval/trial_1.json \\
        --videos-dir mc_multiplayer_v2_eval_new_sneak_combined/structureEval/test \\
        --out replay_drift_check_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
from vlm_utils import query_vlm


def encode_frame_png(video_path: Path, frame_index: int) -> bytes:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"Failed to PNG-encode frame {frame_index} from {video_path}")
        return buf.tobytes()
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to May 1 trial JSON")
    ap.add_argument("--videos-dir", required=True, help="Directory containing source mp4s")
    ap.add_argument("--out", required=True, help="Where to write replay results JSON")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on rows (0 = all)")
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    rows: List[Dict] = baseline["results"]
    if args.limit:
        rows = rows[: args.limit]
    videos_dir = Path(args.videos_dir)

    replayed = []
    same_response = 0
    today_correct = 0
    baseline_correct = 0
    flips_to_wrong = 0
    flips_to_right = 0

    for i, r in enumerate(rows):
        video_name = r["video"]
        frame_index = r["frame_index"]
        prompt = r["metadata"]["prompt"]
        expected = r["expected"]
        baseline_resp = r["response"]
        baseline_was_correct = bool(r["correct"])

        video_path = videos_dir / video_name
        img_bytes = encode_frame_png(video_path, frame_index)

        today_resp = query_vlm(prompt, img_bytes, enable_thinking=False)
        today_resp_norm = today_resp.strip().lower()
        baseline_resp_norm = baseline_resp.strip().lower()
        is_correct_today = today_resp_norm == expected.strip().lower() or (
            expected != "yes" and expected != "no" and today_resp_norm == "yes"
        )
        # The "yes"/"no" structure prompt: ground-truth label uses the
        # structure name (tower_2x1, etc.) but the prompt only asks yes/no.
        # The original eval treated "yes" as correct (since expected was the
        # presence of the structure). Keep the same logic: correct iff the
        # response is "yes".
        is_correct_today = today_resp_norm == "yes"
        baseline_was_correct = baseline_resp_norm == "yes"

        same = today_resp_norm == baseline_resp_norm
        same_response += int(same)
        today_correct += int(is_correct_today)
        baseline_correct += int(baseline_was_correct)
        if baseline_was_correct and not is_correct_today:
            flips_to_wrong += 1
        if (not baseline_was_correct) and is_correct_today:
            flips_to_right += 1

        replayed.append({
            "video": video_name,
            "frame_index": frame_index,
            "expected": expected,
            "baseline_response": baseline_resp,
            "today_response": today_resp,
            "baseline_correct": baseline_was_correct,
            "today_correct": is_correct_today,
            "same_response": same,
        })
        flag = "OK" if same else "DIFF"
        print(f"[{i+1:>2}/{len(rows)}] {video_name} f={frame_index:<4} "
              f"base={baseline_resp_norm!r:>6} today={today_resp_norm!r:>6} {flag}")

    summary = {
        "total": len(rows),
        "same_response_count": same_response,
        "different_response_count": len(rows) - same_response,
        "baseline_accuracy": baseline_correct / len(rows) * 100,
        "today_accuracy": today_correct / len(rows) * 100,
        "flips_correct_to_wrong": flips_to_wrong,
        "flips_wrong_to_correct": flips_to_right,
    }
    print("\n" + "=" * 70)
    print("REPLAY SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    Path(args.out).write_text(json.dumps({"summary": summary, "rows": replayed}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
