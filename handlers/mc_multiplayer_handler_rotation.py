#!/usr/bin/env python3
"""
Handler for mc_multiplayer_eval_rotation dataset (camera rotation task).
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery


class MinecraftRotationHandler(EpisodeTypeHandler):
    """
    Handler for Minecraft rotation evaluation.

    Similar to translation evaluation, but detects camera rotation instead of
    directional movement.
    """

    DATASET_NAMES = ["mc_multiplayer_eval_rotation"]

    def get_prompt(self) -> str:
        return (
            "Here is a Minecraft screenshot potentially showing another player on the screen. "
            "Where is the player located on the screen? "
            "Answer with a single word from \"left\", \"right\", \"center\". "
            "If there is no player on the screen, answer \"no player\"."
        )

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes based on sneak and camera rotation.

        Logic is the same as movement detection, but looks for camera
        rotation instead of forward/back/left/right.
        """
        queries = []

        # Load JSON data
        with open(video_pair.alpha_json) as f:
            alpha_data = json.load(f)
        with open(video_pair.bravo_json) as f:
            bravo_data = json.load(f)

        # Determine which bot is rotating (has sneak)
        alpha_sneak_frame = self._find_last_sneak_frame(alpha_data)
        bravo_sneak_frame = self._find_last_sneak_frame(bravo_data)

        if alpha_sneak_frame is not None:
            rotating_data = alpha_data
            rotating_video = video_pair.alpha_video
            sneak_frame = alpha_sneak_frame
            variant = "alpha"
        elif bravo_sneak_frame is not None:
            rotating_data = bravo_data
            rotating_video = video_pair.bravo_video
            sneak_frame = bravo_sneak_frame
            variant = "bravo"
        else:
            # No rotation found
            return queries

        # Find the camera rotation frame (first frame after sneak with camera movement)
        rotation_frame, rotation_direction = self._find_camera_rotation_frame(
            rotating_data, sneak_frame
        )

        if rotation_frame is None or rotation_direction is None:
            # No camera rotation found
            return queries

        # Calculate keyframe indices
        frame1_idx = sneak_frame + 5
        frame2_idx = rotation_frame + 140

        # Calculate expected answer based on yaw difference
        try:
            expected_answer = self._calculate_expected_answer(
                rotating_data, frame1_idx, frame2_idx
            )
        except ValueError as e:
            # Skip this episode if yaw difference doesn't match expected patterns
            print(f"  ⚠ Skipping episode {video_pair.episode_num} instance {video_pair.instance_num}: {e}")
            return queries

        # Create keyframe queries for BOTH bot perspectives
        for video_path, video_variant in [
            (video_pair.alpha_video, "alpha"),
            (video_pair.bravo_video, "bravo")
        ]:
            queries.append(KeyframeQuery(
                video_path=video_path,
                frame_index=frame1_idx,
                expected_answer=expected_answer,
                metadata={
                    "variant": video_variant,
                    "rotating_bot": variant,
                    "sneak_frame": sneak_frame,
                    "rotation_frame": rotation_frame,
                    "rotation_direction": rotation_direction,
                    "frame1": frame1_idx,
                    "frame2": frame2_idx,
                    "yaw1": rotating_data[frame1_idx].get("yaw", 0),
                    "yaw2": rotating_data[frame2_idx].get("yaw", 0),
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num
                }
            ))

        return queries

    def _find_last_sneak_frame(self, data: List[dict]) -> Optional[int]:
        """Find the last frame where sneak is true."""
        last_sneak = None
        for i, frame in enumerate(data):
            if frame.get("action", {}).get("sneak", False):
                last_sneak = i
        return last_sneak

    def _calculate_expected_answer(
        self, data: List[dict], frame1_idx: int, frame2_idx: int
    ) -> str:
        """
        Calculate expected answer based on yaw difference between two frames.

        Args:
            data: JSON data containing frame information
            frame1_idx: Index of first frame
            frame2_idx: Index of second frame

        Returns:
            Expected answer: "left", "right", or "no player"

        Raises:
            ValueError: If yaw difference doesn't match expected patterns
        """
        import math

        # Get yaw values
        yaw1 = data[frame1_idx].get("yaw", 0)
        yaw2 = data[frame2_idx].get("yaw", 0)

        # Calculate yaw difference
        yaw_diff = yaw2 - yaw1

        # Normalize to [-180, 180] range
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi

        # Convert to degrees
        yaw_diff_deg = math.degrees(yaw_diff)

        # Check if it's a ~40 degree rotation (+/- 5 degrees)
        if 35 <= abs(yaw_diff_deg) <= 45:
            # Positive = turned left (camera moved left, player appears right)
            # Negative = turned right (camera moved right, player appears left)
            if yaw_diff_deg > 0:
                return "right"
            else:
                return "left"

        # Check if it's a ~180 degree rotation (+/- 5 degrees)
        elif 175 <= abs(yaw_diff_deg) <= 185:
            return "no player"

        # Unexpected yaw difference
        else:
            raise ValueError(
                f"Unexpected yaw difference: {yaw_diff_deg:.2f} degrees "
                f"(frame {frame1_idx}: {math.degrees(yaw1):.2f}° -> "
                f"frame {frame2_idx}: {math.degrees(yaw2):.2f}°)"
            )

    def _find_camera_rotation_frame(
        self, data: List[dict], start_frame: int, threshold: float = 0.0001
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the first frame after start_frame with camera rotation.

        Returns:
            Tuple of (frame_index, direction) where direction describes the
            camera rotation (e.g., "left", "right", "up", "down")
        """
        for i in range(start_frame, len(data)):
            action = data[i].get("action", {})
            camera = action.get("camera", [0, 0])

            # Check if there's significant camera movement
            if abs(camera[0]) > threshold or abs(camera[1]) > threshold:
                # Determine direction based on which axis has more movement
                if abs(camera[0]) > abs(camera[1]):
                    # Horizontal rotation (yaw)
                    direction = "left" if camera[0] < 0 else "right"
                else:
                    # Vertical rotation (pitch)
                    direction = "down" if camera[1] < 0 else "up"

                return i, direction

        return None, None


# Reuse the test functions from mc_multiplayer_handler but with new handler
if __name__ == "__main__":
    import argparse
    import re
    # VideoPair already imported at top of file

    def find_mc_video_pairs(folder: Path):
        """Find video pairs in mc_multiplayer format."""
        pattern = re.compile(r'(\d+)_(Alpha|Bravo)_instance_(\d+)_camera\.mp4')

        files = {}
        for file in folder.iterdir():
            match = pattern.match(file.name)
            if match:
                episode_num, variant, instance_num = match.groups()
                key = (episode_num, instance_num)

                if key not in files:
                    files[key] = {}

                if variant == "Alpha":
                    files[key]["alpha_video"] = file
                    json_file = folder / f"{episode_num}_Alpha_instance_{instance_num}.json"
                    if json_file.exists():
                        files[key]["alpha_json"] = json_file
                else:
                    files[key]["bravo_video"] = file
                    json_file = folder / f"{episode_num}_Bravo_instance_{instance_num}.json"
                    if json_file.exists():
                        files[key]["bravo_json"] = json_file

        pairs = []
        for (episode_num, instance_num), file_dict in files.items():
            required_keys = ["alpha_video", "bravo_video", "alpha_json", "bravo_json"]
            if all(key in file_dict for key in required_keys):
                pairs.append(VideoPair(
                    episode_num=episode_num,
                    instance_num=instance_num,
                    alpha_video=file_dict["alpha_video"],
                    bravo_video=file_dict["bravo_video"],
                    alpha_json=file_dict["alpha_json"],
                    bravo_json=file_dict["bravo_json"]
                ))

        return sorted(pairs, key=lambda p: (p.episode_num, p.instance_num))

    def test_and_extract(test_folder: str, num_episodes: int, extract_frames: bool, output_dir: str):
        """Test keyframe extraction and optionally extract frames."""
        import cv2

        test_path = Path(test_folder)
        handler = MinecraftCameraRotationHandler()

        video_pairs = find_mc_video_pairs(test_path)
        video_pairs = video_pairs[:num_episodes]

        print(f"Testing keyframe extraction on {len(video_pairs)} episodes")
        print("=" * 80)

        results = []
        for i, pair in enumerate(video_pairs, 1):
            print(f"\nEpisode {i}: {pair.episode_num} (instance {pair.instance_num})")
            print("-" * 80)

            queries = handler.extract_keyframes(pair)

            if not queries:
                print("  ⚠ No camera rotation found in this episode")
                continue

            meta = queries[0].metadata
            print(f"  Rotating bot: {meta['rotating_bot'].upper()}")
            print(f"  Sneak frame: {meta['sneak_frame']}")
            print(f"  Rotation frame: {meta['rotation_frame']}")
            print(f"  Rotation direction: {meta['rotation_direction']}")
            print(f"  Keyframe 1: frame {meta['frame1']}")
            print(f"  Keyframe 2: frame {meta['frame2']}")
            print(f"  Expected answer: {queries[0].expected_answer}")
            print(f"  Extracting from: BOTH Alpha and Bravo perspectives")

            for query in queries:
                query_meta = query.metadata
                results.append({
                    "episode": pair.episode_num,
                    "instance": pair.instance_num,
                    "variant": query_meta['variant'],
                    "frame1": query_meta['frame1'],
                    "frame2": query_meta['frame2'],
                    "rotation": query_meta['rotation_direction'],
                    "expected": query.expected_answer,
                    "video_path": str(query.video_path)
                })

        # Extract frames if requested
        if extract_frames and results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"\n{'=' * 80}")
            print(f"Extracting frames to: {output_path}")
            print("=" * 80)

            for i, result in enumerate(results, 1):
                video_path = result['video_path']
                frame1_idx = result['frame1']
                frame2_idx = result['frame2']
                episode = result['episode']
                instance = result['instance']
                variant = result['variant']

                print(f"\nEpisode {i}: {episode} (instance {instance}) - {variant}")

                cap = cv2.VideoCapture(video_path)

                # Extract frame 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
                ret1, frame1 = cap.read()

                # Extract frame 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
                ret2, frame2 = cap.read()

                cap.release()

                if ret1 and ret2:
                    frame1_path = output_path / f"ep{episode}_inst{instance}_{variant}_frame1.png"
                    frame2_path = output_path / f"ep{episode}_inst{instance}_{variant}_frame2.png"

                    cv2.imwrite(str(frame1_path), frame1)
                    cv2.imwrite(str(frame2_path), frame2)

                    print(f"  ✓ Saved frame 1 (idx {frame1_idx}): {frame1_path.name}")
                    print(f"  ✓ Saved frame 2 (idx {frame2_idx}): {frame2_path.name}")
                    print(f"  Expected answer: {result['expected']}")
                else:
                    print(f"  ✗ Failed to extract frames")

            print(f"\n{'=' * 80}")
            print(f"Frames saved to: {output_path.absolute()}")
            print("=" * 80)

    parser = argparse.ArgumentParser(
        description="Test Minecraft camera rotation keyframe extraction"
    )
    parser.add_argument(
        "--test-folder",
        default="/home/dl3957/Documents/mp_eval_datasets/mc_multiplayer_eval_dev_2/test",
        help="Path to test folder"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to test"
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract and save actual frames for visual inspection"
    )
    parser.add_argument(
        "--output-dir",
        default="test_frames_dev2",
        help="Output directory for extracted frames"
    )

    args = parser.parse_args()

    test_and_extract(args.test_folder, args.num_episodes, args.extract_frames, args.output_dir)
