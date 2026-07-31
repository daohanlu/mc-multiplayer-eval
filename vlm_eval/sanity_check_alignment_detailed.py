#!/usr/bin/env python3
"""
Enhanced sanity check script with detailed difference curve visualization.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from typing import Tuple, List

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_eval import find_mc_video_pairs, identify_handler, extract_quadrant, find_generated_video_subdir


def compute_difference_curve(target_frame: np.ndarray, video_path: Path, quadrant: str,
                             center_frame: int, window: int = 30) -> Tuple[List[int], List[float], List[np.ndarray]]:
    """
    Compute pixel difference for frames around a center frame.

    Args:
        target_frame: Target frame to match (already resized to 640x360)
        video_path: Path to the side-by-side video
        quadrant: Which quadrant to search ("top-left" or "bottom-left")
        center_frame: Frame to center the search around
        window: Number of frames to search on each side

    Returns:
        Tuple of (frame_indices, differences, frames)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, center_frame - window)
    end_frame = min(total_frames, center_frame + window + 1)

    frame_indices = []
    differences = []
    frames = []

    for i in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # Extract the appropriate quadrant
        quad_frame = extract_quadrant(frame, quadrant)

        # Calculate absolute pixel difference
        diff = np.abs(target_frame.astype(float) - quad_frame.astype(float)).mean()

        frame_indices.append(i)
        differences.append(diff)
        frames.append(quad_frame.copy())

    cap.release()
    return frame_indices, differences, frames


def visualize_detailed_comparison(real_frame: np.ndarray, frame_indices: List[int],
                                 differences: List[float], frames: List[np.ndarray],
                                 episode: str, instance: str, variant: str,
                                 real_idx: int, expected_idx: int, frame1_idx: int, frame2_idx: int,
                                 output_path: Path):
    """
    Create a detailed visualization with difference curve and frame comparisons.
    """
    # Find best match
    best_idx_in_list = np.argmin(differences)
    best_frame_idx = frame_indices[best_idx_in_list]
    best_diff = differences[best_idx_in_list]
    best_frame = frames[best_idx_in_list]

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Top row: Difference curve
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(frame_indices, differences, 'b-', linewidth=2, label='Pixel Difference')
    ax_curve.axvline(expected_idx, color='g', linestyle='--', linewidth=2, label=f'Expected ({expected_idx})')
    ax_curve.axvline(best_frame_idx, color='r', linestyle='--', linewidth=2, label=f'Found ({best_frame_idx})')
    ax_curve.set_xlabel('Frame Index in Generated Video')
    ax_curve.set_ylabel('Mean Pixel Difference')
    ax_curve.set_title('Pixel Difference Curve')
    ax_curve.legend()
    ax_curve.grid(True, alpha=0.3)

    # Middle row: Real frame and best match
    ax_real = fig.add_subplot(gs[1, 0])
    real_rgb = cv2.cvtColor(real_frame, cv2.COLOR_BGR2RGB)
    ax_real.imshow(real_rgb)
    ax_real.set_title(f'Real GT Video\nFrame {real_idx}\n({variant.upper()})')
    ax_real.axis('off')

    ax_best = fig.add_subplot(gs[1, 1])
    best_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    ax_best.imshow(best_rgb)
    ax_best.set_title(f'Best Match\nFrame {best_frame_idx}\nDiff: {best_diff:.2f}')
    ax_best.axis('off')

    # Expected frame if different from best
    if expected_idx in frame_indices:
        expected_idx_in_list = frame_indices.index(expected_idx)
        expected_frame = frames[expected_idx_in_list]
        expected_diff = differences[expected_idx_in_list]

        ax_expected = fig.add_subplot(gs[1, 2])
        expected_rgb = cv2.cvtColor(expected_frame, cv2.COLOR_BGR2RGB)
        ax_expected.imshow(expected_rgb)
        ax_expected.set_title(f'Expected Frame\nFrame {expected_idx}\nDiff: {expected_diff:.2f}')
        ax_expected.axis('off')
    else:
        ax_expected = fig.add_subplot(gs[1, 2])
        ax_expected.text(0.5, 0.5, 'Expected frame\nout of range',
                        ha='center', va='center', fontsize=12)
        ax_expected.axis('off')

    # Bottom row: Show a few frames around the best match
    num_show = 3
    show_indices = []
    for offset in range(-num_show//2, num_show//2 + 1):
        idx = best_frame_idx + offset
        if idx in frame_indices:
            show_indices.append(idx)

    for i, show_idx in enumerate(show_indices[:3]):
        ax = fig.add_subplot(gs[2, i])
        show_idx_in_list = frame_indices.index(show_idx)
        show_frame = frames[show_idx_in_list]
        show_diff = differences[show_idx_in_list]
        show_rgb = cv2.cvtColor(show_frame, cv2.COLOR_BGR2RGB)
        ax.imshow(show_rgb)
        marker = '★' if show_idx == best_frame_idx else ''
        ax.set_title(f'Frame {show_idx} {marker}\nDiff: {show_diff:.2f}')
        ax.axis('off')

    # Add overall info
    offset_diff = best_frame_idx - expected_idx
    offset_str = f"+{offset_diff}" if offset_diff > 0 else str(offset_diff)

    fig.suptitle(
        f'Episode {episode}, Instance {instance}, {variant.upper()}\n'
        f'Real frame: {real_idx}, Expected gen frame: {expected_idx}, Found gen frame: {best_frame_idx} (offset: {offset_str})\n'
        f'frame1_idx: {frame1_idx}, frame2_idx: {frame2_idx}',
        fontsize=12, fontweight='bold'
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_detailed_sanity_check(dataset_folder: str, generated_path: str, num_videos: int = 3):
    """
    Run detailed sanity check with difference curves.
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
    output_dir = Path("sanity_check_detailed")
    output_dir.mkdir(exist_ok=True)

    print(f"{'='*80}")
    print("DETAILED SANITY CHECK: Alignment Verification")
    print(f"{'='*80}\n")

    # Collect all offsets for summary
    all_offsets = []

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

        # Test frame index (use frame1_idx + 40 for more visual variation during movement)
        test_frame_idx = frame1_idx + 40
        print(f"  Testing with frame: {test_frame_idx} (frame1_idx + 40)")
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

            # Compute difference curve
            quadrant = "top-left" if variant == "alpha" else "bottom-left"
            expected_offset = test_frame_idx - frame1_idx

            frame_indices, differences, frames = compute_difference_curve(
                real_frame_resized, generated_video, quadrant,
                center_frame=expected_offset, window=30
            )

            # Find best match
            best_idx_in_list = np.argmin(differences)
            best_frame_idx = frame_indices[best_idx_in_list]
            best_diff = differences[best_idx_in_list]

            offset_diff = best_frame_idx - expected_offset
            offset_str = f"+{offset_diff}" if offset_diff > 0 else str(offset_diff)
            match_str = "✓" if offset_diff == 0 else "✗"

            print(f"    {variant.upper()}: Found best match at frame {best_frame_idx} (expected {expected_offset}, offset: {offset_str}) {match_str}")
            print(f"            Mean pixel diff: {best_diff:.2f}")

            all_offsets.append(offset_diff)

            # Create detailed visualization
            viz_path = output_dir / f"video{video_idx}_{episode}_{instance}_{variant}_detailed.png"
            visualize_detailed_comparison(
                real_frame_resized, frame_indices, differences, frames,
                episode, instance, variant,
                test_frame_idx, expected_offset, frame1_idx, frame2_idx,
                viz_path
            )
            print(f"            Saved visualization: {viz_path}")

        print()

    # Print summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if all_offsets:
        all_offsets = np.array(all_offsets)
        print(f"Offset statistics across all tests:")
        print(f"  Mean offset: {all_offsets.mean():.2f} frames")
        print(f"  Median offset: {np.median(all_offsets):.0f} frames")
        print(f"  Min offset: {all_offsets.min():.0f} frames")
        print(f"  Max offset: {all_offsets.max():.0f} frames")
        print(f"  Std dev: {all_offsets.std():.2f} frames")
        print(f"\nOffset distribution: {dict(zip(*np.unique(all_offsets, return_counts=True)))}")
    print(f"\nVisualizations saved to: {output_dir.absolute()}")
    print(f"{'='*80}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detailed sanity check for alignment with difference curves"
    )
    parser.add_argument(
        "dataset_folder",
        help="Path to dataset folder (e.g., mc_multiplayer_eval_translation/test)"
    )
    parser.add_argument(
        "generated_path",
        help="Path to generated videos directory"
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=3,
        help="Number of videos to check (default: 3)"
    )

    args = parser.parse_args()

    run_detailed_sanity_check(args.dataset_folder, args.generated_path, args.num_videos)


if __name__ == "__main__":
    main()
