#!/usr/bin/env python3
"""
Sanity check script to verify alignment between ground-truth videos
and generated videos' ground-truth quadrants.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from typing import Tuple

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_eval import find_mc_video_pairs, identify_handler, extract_quadrant, find_generated_video_subdir


def find_most_similar_frame(target_frame: np.ndarray, video_path: Path, quadrant: str, max_search: int = 300) -> Tuple[int, float, np.ndarray]:
    """
    Find the frame in the video's quadrant that is most similar to target_frame.

    Args:
        target_frame: Target frame to match (already resized to 640x360)
        video_path: Path to the side-by-side video
        quadrant: Which quadrant to search ("top-left" or "bottom-left")
        max_search: Maximum number of frames to search

    Returns:
        Tuple of (best_frame_index, min_difference, best_matching_frame)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    search_frames = min(total_frames, max_search)

    min_diff = float('inf')
    best_idx = -1
    best_frame = None

    for i in range(search_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract the appropriate quadrant
        quad_frame = extract_quadrant(frame, quadrant)

        # Calculate absolute pixel difference
        diff = np.abs(target_frame.astype(float) - quad_frame.astype(float)).mean()

        if diff < min_diff:
            min_diff = diff
            best_idx = i
            best_frame = quad_frame.copy()

    cap.release()
    return best_idx, min_diff, best_frame


def visualize_comparison(real_frame: np.ndarray, generated_frame: np.ndarray,
                        episode: str, instance: str, variant: str,
                        real_idx: int, gen_idx: int, frame1_idx: int, frame2_idx: int,
                        diff: float, output_path: Path):
    """
    Create a visualization comparing real and generated frames.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Convert BGR to RGB for display
    real_rgb = cv2.cvtColor(real_frame, cv2.COLOR_BGR2RGB)
    gen_rgb = cv2.cvtColor(generated_frame, cv2.COLOR_BGR2RGB)

    axes[0].imshow(real_rgb)
    axes[0].set_title(f'Real GT Video\nFrame {real_idx}\n({variant.upper()})')
    axes[0].axis('off')

    axes[1].imshow(gen_rgb)
    axes[1].set_title(f'Generated GT Quadrant\nFrame {gen_idx}\n(Most Similar)')
    axes[1].axis('off')

    # Add info text
    expected_offset = real_idx - frame1_idx
    actual_offset = gen_idx
    offset_match = "✓" if expected_offset == actual_offset else "✗"

    fig.suptitle(
        f'Episode {episode}, Instance {instance}, {variant.upper()}\n'
        f'Real frame: {real_idx}, Expected gen frame: {expected_offset}, Found gen frame: {gen_idx} {offset_match}\n'
        f'frame1_idx: {frame1_idx}, frame2_idx: {frame2_idx}\n'
        f'Mean pixel difference: {diff:.2f}',
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_sanity_check(dataset_folder: str, generated_path: str, num_videos: int = 3):
    """
    Run sanity check on alignment between real and generated videos.

    Args:
        dataset_folder: Path to dataset folder (e.g., mc_multiplayer_eval_translation/test)
        generated_path: Path to generated videos directory
        num_videos: Number of videos to check
    """
    folder = Path(dataset_folder)
    generated_base = Path(generated_path)

    # Identify handler and get video pairs
    dataset_name = folder.parent.name if folder.name == "test" else folder.name
    handler = identify_handler(dataset_name)
    video_pairs = find_mc_video_pairs(folder)

    print(f"Dataset: {dataset_name}")
    print(f"Handler: {handler.__class__.__name__}")
    print(f"Found: {len(video_pairs)} video pairs")
    print(f"Testing first {num_videos} videos\n")

    # Find generated video subdirectory
    generated_subdir = find_generated_video_subdir(generated_base, dataset_name)
    if not generated_subdir:
        print(f"Error: Could not find generated video subdirectory for dataset '{dataset_name}'")
        return

    print(f"Generated video subdirectory: {generated_subdir.name}\n")

    # Create output directory for visualizations
    output_dir = Path("sanity_check_visualizations")
    output_dir.mkdir(exist_ok=True)

    print(f"{'='*80}")
    print("SANITY CHECK: Alignment Verification")
    print(f"{'='*80}\n")

    # Test first N videos
    for video_idx in range(min(num_videos, len(video_pairs))):
        pair = video_pairs[video_idx]

        # Extract keyframes to get frame indices
        queries = handler.extract_keyframes(pair)
        if not queries:
            print(f"Video {video_idx}: No movement found, skipping\n")
            continue

        # Get metadata from first query
        meta = queries[0].metadata
        frame1_idx = meta['frame1']
        frame2_idx = meta['frame2']
        episode = meta['episode']
        instance = meta['instance']

        print(f"Video {video_idx}: Episode {episode}, Instance {instance}")
        print(f"  frame1_idx (sneak+5): {frame1_idx}")
        print(f"  frame2_idx (movement+40): {frame2_idx}")

        # Test frame index (frame2_idx - 40 should be close to movement frame)
        test_frame_idx = frame2_idx - 40
        print(f"  Testing with frame: {test_frame_idx}")
        print(f"  Expected offset in generated video: {test_frame_idx - frame1_idx}")

        # Get generated video path
        generated_video = generated_subdir / f"video_{video_idx}_side_by_side.mp4"
        if not generated_video.exists():
            print(f"  ✗ Generated video not found: {generated_video.name}\n")
            continue

        # Test both alpha and bravo
        for query in queries[:2]:  # Only take first 2 (alpha and bravo)
            variant = query.metadata['variant']
            real_video = query.video_path

            # Extract frame from real video
            cap_real = cv2.VideoCapture(str(real_video))
            cap_real.set(cv2.CAP_PROP_POS_FRAMES, test_frame_idx)
            ret, real_frame = cap_real.read()
            cap_real.release()

            if not ret:
                print(f"    {variant.upper()}: ✗ Failed to extract frame {test_frame_idx} from real video")
                continue

            # Resize to 640x360
            real_frame_resized = cv2.resize(real_frame, (640, 360))

            # Find most similar frame in generated video's GT quadrant
            quadrant = "top-left" if variant == "alpha" else "bottom-left"
            best_idx, min_diff, best_frame = find_most_similar_frame(
                real_frame_resized, generated_video, quadrant
            )

            # Calculate expected offset
            expected_offset = test_frame_idx - frame1_idx
            offset_match = "✓" if best_idx == expected_offset else "✗"

            print(f"    {variant.upper()}: Found best match at frame {best_idx} (expected {expected_offset}) {offset_match}")
            print(f"            Mean pixel diff: {min_diff:.2f}")

            # Create visualization
            viz_path = output_dir / f"video{video_idx}_{episode}_{instance}_{variant}.png"
            visualize_comparison(
                real_frame_resized, best_frame,
                episode, instance, variant,
                test_frame_idx, best_idx, frame1_idx, frame2_idx,
                min_diff, viz_path
            )
            print(f"            Saved visualization: {viz_path}")

        print()

    print(f"{'='*80}")
    print(f"Visualizations saved to: {output_dir.absolute()}")
    print(f"{'='*80}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanity check alignment between real and generated videos"
    )
    parser.add_argument(
        "dataset_folder",
        help="Path to dataset folder (e.g., mc_multiplayer_eval_translation/test)"
    )
    parser.add_argument(
        "generated_path",
        help="Path to generated videos directory (e.g., generations/flagship_final_v2_1B_multiplayer_final)"
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=3,
        help="Number of videos to check (default: 3)"
    )

    args = parser.parse_args()

    run_sanity_check(args.dataset_folder, args.generated_path, args.num_videos)


if __name__ == "__main__":
    main()
