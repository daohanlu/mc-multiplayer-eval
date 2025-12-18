#!/usr/bin/env python3
"""
Handler for mc_multiplayer_eval_structure dataset (structure building task).
"""

import json
from pathlib import Path
from typing import List, Optional
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery


class MinecraftStructureBuildingHandler(EpisodeTypeHandler):
    """
    Handler for Minecraft structure building evaluation.

    Evaluates from the perspective of the non-building bot to see if they
    can observe the structure being built.
    """

    DATASET_NAMES = ["structureEval"]

    def __init__(self, summary_json_path: str):
        """
        Initialize handler with path to structure building summary JSON.
        
        Args:
            summary_json_path: Path to structure_building_summary.json
        """
        self.summary_json_path = summary_json_path
        self.summary_data = self._load_summary()

    def _load_summary(self) -> dict:
        """Load the structure building summary JSON."""
        with open(self.summary_json_path) as f:
            return json.load(f)

    def get_prompt(self) -> str:
        return (
            "Here is a Minecraft screenshot. "
            "Can you tell me whether there is a visible structure built about 6 blocks away from the player? "
            "Answer with a single word from \"yes\", \"no\"."
        )

    def validate_response(self, response: str, expected: str) -> bool:
        """
        Validate the VLM response against the expected structure type.
        
        Maps structure types from JSON format to prompt format:
        - wall_4x1 → strip (1x4 horizontal)
        - tower_2 → tower (4x1 vertical) 
        - wall_2x2 → square (2x2)
        
        Args:
            response: VLM response (should be "strip", "tower", "square", or "unclear")
            expected: Expected structure from JSON (e.g., "wall_4x1", "tower_2", "wall_2x2")
            
        Returns:
            True if response matches expected structure type, False otherwise
        """
        # Map JSON structure names to prompt answer format
        structure_mapping = {
            "wall_4x1": "yes",
            "tower_2x1": "yes",
            "wall_2x2": "yes"
        }
        
        # Normalize the response
        normalized_response = response.strip().lower()
        
        # Map expected structure to prompt format
        expected_answer = structure_mapping.get(expected.strip().lower(), "unclear")
        
        # Check if response matches expected answer
        return normalized_response == expected_answer

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes from the non-building bot's perspective.
        
        Args:
            video_pair: Pair of videos and JSON files for alpha and bravo
            
        Returns:
            List of keyframe queries (one per episode, from observer's perspective)
        """
        queries = []

        # Get episode and instance info
        # Strip leading zeros from episode and instance numbers for JSON lookup
        episode_num = str(int(video_pair.episode_num))
        instance_num = str(int(video_pair.instance_num))
        
        episode_key = f"episode_{episode_num}"
        instance_key = f"instance_{instance_num}"

        # Get builder info from summary
        if instance_key not in self.summary_data:
            print(f"  ⚠ Instance {video_pair.instance_num} not found in summary")
            return queries

        if episode_key not in self.summary_data[instance_key]:
            print(f"  ⚠ Episode {video_pair.episode_num} not found in instance {video_pair.instance_num}")
            return queries

        episode_data = self.summary_data[instance_key][episode_key]
        builder = episode_data["builder"]
        structure = episode_data["structure"]
        
        # Determine which bot is observing (not building)
        if builder == "alpha":
            observer = "bravo"
            observer_video = video_pair.bravo_video
            observer_json = video_pair.bravo_json
        else:
            observer = "alpha"
            observer_video = video_pair.alpha_video
            observer_json = video_pair.alpha_json

        # Load observer JSON to verify it has enough frames
        with open(observer_json) as f:
            observer_data = json.load(f)

        # Calculate keyframe indices
        # Start from frame 20 to allow scene to stabilize
        frame1_idx = 20
        frame2_idx = frame1_idx + 240

        # Check if we have enough frames
        if frame2_idx >= len(observer_data):
            print(f"  ⚠ Episode {video_pair.episode_num} instance {video_pair.instance_num}: "
                  f"Not enough frames (need {frame2_idx}, have {len(observer_data)})")
            return queries

        # Create keyframe query only for the observer (non-building bot)
        queries.append(KeyframeQuery(
            video_path=observer_video,
            frame_index=frame2_idx,
            expected_answer=structure,
            metadata={
                "variant": observer,
                "builder": builder,
                "structure": structure,
                "alpha_structure": episode_data["alpha_structure"],
                "bravo_structure": episode_data["bravo_structure"],
                "alpha_builds": episode_data["alpha_builds"],
                "bravo_builds": episode_data["bravo_builds"],
                "frame1": frame1_idx,
                "frame2": frame2_idx,
                "episode": video_pair.episode_num,
                "instance": video_pair.instance_num
            }
        ))

        return queries


# Test functions
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

    def test_and_extract(test_folder: str, summary_json: str, num_episodes: int, 
                        extract_frames: bool, output_dir: str):
        """Test keyframe extraction and optionally extract frames."""
        import cv2

        test_path = Path(test_folder)
        handler = MinecraftStructureBuildingHandler(summary_json)

        video_pairs = find_mc_video_pairs(test_path)
        
        if num_episodes > 0:
            video_pairs = video_pairs[:num_episodes]

        print(f"Testing keyframe extraction on {len(video_pairs)} episodes")
        print("=" * 80)

        results = []
        for i, pair in enumerate(video_pairs, 1):
            print(f"\nEpisode {i}: {pair.episode_num} (instance {pair.instance_num})")
            print("-" * 80)

            queries = handler.extract_keyframes(pair)

            if not queries:
                print("  ⚠ No valid keyframes found in this episode")
                continue

            query = queries[0]
            meta = query.metadata
            print(f"  Builder: {meta['builder'].upper()}")
            print(f"  Observer: {meta['variant'].upper()}")
            print(f"  Structure: {meta['structure']}")
            print(f"  Alpha structure: {meta['alpha_structure']} (builds: {meta['alpha_builds']})")
            print(f"  Bravo structure: {meta['bravo_structure']} (builds: {meta['bravo_builds']})")
            print(f"  Keyframe 1: frame {meta['frame1']}")
            print(f"  Keyframe 2: frame {meta['frame2']}")
            print(f"  Expected answer: {query.expected_answer}")
            print(f"  Extracting from: {meta['variant'].upper()} perspective (observer)")

            results.append({
                "episode": pair.episode_num,
                "instance": pair.instance_num,
                "variant": meta['variant'],
                "builder": meta['builder'],
                "frame1": meta['frame1'],
                "frame2": meta['frame2'],
                "structure": meta['structure'],
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
                structure = result['structure']

                print(f"\nEpisode {i}: {episode} (instance {instance}) - {variant} observing")

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
                    print(f"  Structure: {structure}")
                    print(f"  Expected answer: {result['expected']}")
                else:
                    print(f"  ✗ Failed to extract frames")

            print(f"\n{'=' * 80}")
            print(f"Frames saved to: {output_path.absolute()}")
            print("=" * 80)

    parser = argparse.ArgumentParser(
        description="Test Minecraft structure building keyframe extraction"
    )
    parser.add_argument(
        "--test-folder",
        default="/home/dl3957/Documents/mp_eval_datasets/mc_multiplayer_eval_structure/test",
        help="Path to test folder"
    )
    parser.add_argument(
        "--summary-json",
        default="/home/dl3957/Documents/mp_eval_datasets/structure_building_summary.json",
        help="Path to structure building summary JSON"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Number of episodes to test (0 = all)"
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract and save actual frames for visual inspection"
    )
    parser.add_argument(
        "--output-dir",
        default="frame_extraction/mc_multiplayer_eval_structure",
        help="Output directory for extracted frames"
    )

    args = parser.parse_args()

    test_and_extract(args.test_folder, args.summary_json, args.num_episodes, 
                    args.extract_frames, args.output_dir)

