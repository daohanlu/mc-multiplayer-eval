#!/usr/bin/env python3
"""
Quick sanity check to print all offsets for entire dataset.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_eval import find_mc_video_pairs, identify_handler, extract_quadrant, find_generated_video_subdir


def find_best_match_frame(target_frame: np.ndarray, video_path: Path, quadrant: str,
                          center_frame: int, window: int = 30) -> tuple:
    """Find the frame with minimum pixel difference."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, center_frame - window)
    end_frame = min(total_frames, center_frame + window + 1)

    min_diff = float('inf')
    best_idx = -1

    for i in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        quad_frame = extract_quadrant(frame, quadrant)
        diff = np.abs(target_frame.astype(float) - quad_frame.astype(float)).mean()

        if diff < min_diff:
            min_diff = diff
            best_idx = i

    cap.release()
    return best_idx, min_diff


def run_quick_check(dataset_folder: str, generated_path: str, test_offset: int = 40):
    """Run quick check and print all offsets."""
    folder = Path(dataset_folder)
    generated_base = Path(generated_path)

    # Setup
    dataset_name = folder.parent.name if folder.name == "test" else folder.name
    handler = identify_handler(dataset_name)
    video_pairs = find_mc_video_pairs(folder)

    print(f"Dataset: {dataset_name}")
    print(f"Total video pairs: {len(video_pairs)}")
    print(f"Testing frame: frame1_idx + {test_offset}\n")

    generated_subdir = find_generated_video_subdir(generated_base, dataset_name)
    if not generated_subdir:
        print(f"Error: Could not find generated video subdirectory")
        return

    print(f"Generated subdirectory: {generated_subdir.name}\n")
    print(f"{'='*100}")
    print(f"{'Video':<8} {'Episode':<10} {'Inst':<6} {'Variant':<8} {'frame1':<8} {'test_fr':<8} {'Expected':<10} {'Found':<8} {'Offset':<8} {'PixDiff':<10}")
    print(f"{'='*100}")

    all_offsets = []
    all_pixel_diffs = []

    for video_idx in range(len(video_pairs)):
        pair = video_pairs[video_idx]
        queries = handler.extract_keyframes(pair)

        if not queries:
            continue

        meta = queries[0].metadata
        frame1_idx = meta['frame1']
        episode = meta['episode']
        instance = meta['instance']
        test_frame_idx = frame1_idx + test_offset + 1
        expected_offset = test_offset

        generated_video = generated_subdir / f"video_{video_idx}_side_by_side.mp4"
        if not generated_video.exists():
            continue

        for query in queries[:2]:  # Alpha and Bravo
            variant = query.metadata['variant']
            real_video = query.video_path

            # Extract frame from real video
            cap_real = cv2.VideoCapture(str(real_video))
            cap_real.set(cv2.CAP_PROP_POS_FRAMES, test_frame_idx)
            ret, real_frame = cap_real.read()
            cap_real.release()

            if not ret:
                continue

            real_frame_resized = cv2.resize(real_frame, (640, 360))

            # Find best match
            quadrant = "top-left" if variant == "alpha" else "bottom-left"
            best_idx, min_diff = find_best_match_frame(
                real_frame_resized, generated_video, quadrant, expected_offset
            )

            offset_diff = best_idx - expected_offset
            all_offsets.append(offset_diff)
            all_pixel_diffs.append(min_diff)

            match_str = "✓" if offset_diff == 0 else ""
            print(f"{video_idx:<8} {episode:<10} {instance:<6} {variant:<8} {frame1_idx:<8} {test_frame_idx:<8} {expected_offset:<10} {best_idx:<8} {offset_diff:<8} {min_diff:<10.2f} {match_str}")

    print(f"{'='*100}")
    print(f"\nSUMMARY STATISTICS:")
    print(f"{'='*100}")

    if all_offsets:
        all_offsets = np.array(all_offsets)
        all_pixel_diffs = np.array(all_pixel_diffs)

        print(f"Total tests: {len(all_offsets)}")
        print(f"\nOffset Statistics:")
        print(f"  Mean:   {all_offsets.mean():.3f} frames")
        print(f"  Median: {np.median(all_offsets):.0f} frames")
        print(f"  Std:    {all_offsets.std():.3f} frames")
        print(f"  Min:    {all_offsets.min():.0f} frames")
        print(f"  Max:    {all_offsets.max():.0f} frames")

        print(f"\nPixel Difference Statistics:")
        print(f"  Mean:   {all_pixel_diffs.mean():.3f}")
        print(f"  Median: {np.median(all_pixel_diffs):.3f}")
        print(f"  Min:    {all_pixel_diffs.min():.3f}")
        print(f"  Max:    {all_pixel_diffs.max():.3f}")

        print(f"\nOffset Distribution:")
        unique, counts = np.unique(all_offsets, return_counts=True)
        for val, count in zip(unique, counts):
            pct = count / len(all_offsets) * 100
            print(f"  {int(val):+3d} frames: {count:3d} ({pct:5.1f}%)")

        perfect_matches = np.sum(all_offsets == 0)
        print(f"\nPerfect matches (offset = 0): {perfect_matches}/{len(all_offsets)} ({perfect_matches/len(all_offsets)*100:.1f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quick offset check for all videos")
    parser.add_argument("dataset_folder", help="Path to dataset folder")
    parser.add_argument("generated_path", help="Path to generated videos directory")
    parser.add_argument("--offset", type=int, default=40, help="Offset from frame1_idx to test (default: 40)")

    args = parser.parse_args()
    run_quick_check(args.dataset_folder, args.generated_path, args.offset)


if __name__ == "__main__":
    main()
