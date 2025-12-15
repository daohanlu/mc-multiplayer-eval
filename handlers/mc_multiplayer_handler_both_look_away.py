#!/usr/bin/env python3
"""
Handler for mc_multiplayer_eval_both_look_away dataset.

Both bots look away and then back. Similar timing to one_looks_away.
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery


class MinecraftBothLookAwayHandler(EpisodeTypeHandler):
    """
    Handler for Minecraft "both look away" evaluation.

    Both bots rotate away and then back.
    """

    DATASET_NAMES = ["mc_multiplayer_eval_both_look_away"]

    def get_prompt(self, query_type: str = "player_position") -> str:
        if query_type == "player_visible":
            return (
                "You will be shown two Minecraft screenshots. "
                "Is there another player visible on-screen in the second screenshot? "
                "Answer with a single word: \"yes\", \"no\"."
            )
        else:  # player_position or player_position_final
            return (
                "Here is a Minecraft screenshot potentially showing another player on the screen. "
                "Where is the player located on the screen? "
                "Answer with a single word from \"left\", \"right\", \"center\". "
                "If there is no player on the screen, answer \"no player\"."
            )

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes based on sneak and camera rotation.

        Since both bots look away, we extract keyframes from both perspectives.
        The starting frame (frame1) is based on the LATEST of the two sneak frames.
        """
        queries = []

        # Load JSON data
        with open(video_pair.alpha_json) as f:
            alpha_data = json.load(f)
        with open(video_pair.bravo_json) as f:
            bravo_data = json.load(f)

        # Check both bots for sneak frames
        alpha_sneak_frame = self._find_last_sneak_frame(alpha_data)
        bravo_sneak_frame = self._find_last_sneak_frame(bravo_data)

        # Ensure both bots have sneak frames
        if alpha_sneak_frame is None or bravo_sneak_frame is None:
            return queries

        # Use the LATEST sneak frame as the starting point for both bots
        latest_sneak_frame = max(alpha_sneak_frame, bravo_sneak_frame)

        # Process each bot that has rotation
        for data, video_path, variant, bot_sneak_frame in [
            (alpha_data, video_pair.alpha_video, "alpha", alpha_sneak_frame),
            (bravo_data, video_pair.bravo_video, "bravo", bravo_sneak_frame)
        ]:
            # Find the camera rotation frame
            rotation_frame, rotation_direction = self._find_camera_rotation_frame(
                data, bot_sneak_frame
            )

            if rotation_frame is None or rotation_direction is None:
                continue

            # Calculate keyframe indices using the LATEST sneak frame
            frame1_idx = latest_sneak_frame + 5
            frame2_idx = rotation_frame + 200

            # Find intermediate frame (when bot stops turning)
            intermediate_frame = self._find_stop_turning_frame(data, frame1_idx)
            if intermediate_frame is None:
                print(f"  ⚠ Skipping {variant} for episode {video_pair.episode_num} instance {video_pair.instance_num}: No stop turning frame found")
                continue

            # For the final frame query, bot should be back at center
            final_position_answer = "center"

            # Determine expected answer for player visibility at intermediate frame
            # At intermediate frame, bot is looking away, so player should be visible
            player_visible_answer = "yes"

            # Calculate player position frame and expected answer
            position_frame_idx = rotation_frame + 15
            try:
                position_answer = self._calculate_position_answer(
                    data, frame1_idx, position_frame_idx
                )
                assert position_answer in ["left", "right"]
            except (ValueError, AssertionError) as e:
                print(f"  ⚠ Skipping {variant} for episode {video_pair.episode_num} instance {video_pair.instance_num}: {e}")
                continue

            # Create first query: player visibility at intermediate frame
            queries.append(KeyframeQuery(
                video_path=video_path,
                frame_index=frame1_idx,
                expected_answer=player_visible_answer,
                metadata={
                    "variant": variant,
                    "rotating_bot": variant,  # In this case, the bot itself is rotating
                    "query_type": "player_visible",
                    "sneak_frame": bot_sneak_frame,
                    "latest_sneak_frame": latest_sneak_frame,  # Track which sneak frame was used for frame1
                    "rotation_frame": rotation_frame,
                    "rotation_direction": rotation_direction,
                    "frame1": frame1_idx,
                    "frame2": intermediate_frame,
                    "yaw1": data[frame1_idx].get("yaw", 0),
                    "yaw2": data[intermediate_frame].get("yaw", 0),
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num
                }
            ))

            # Create second query: player position at rotation_frame + 27
            queries.append(KeyframeQuery(
                video_path=video_path,
                frame_index=frame1_idx,
                expected_answer=position_answer,
                metadata={
                    "variant": variant,
                    "rotating_bot": variant,  # In this case, the bot itself is rotating
                    "query_type": "player_position",
                    "sneak_frame": bot_sneak_frame,
                    "latest_sneak_frame": latest_sneak_frame,  # Track which sneak frame was used for frame1
                    "rotation_frame": rotation_frame,
                    "rotation_direction": rotation_direction,
                    "frame1": frame1_idx,
                    "frame2": position_frame_idx,
                    "yaw1": data[frame1_idx].get("yaw", 0),
                    "yaw2": data[position_frame_idx].get("yaw", 0),
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num
                }
            ))

            # Create third query: player position at final frame (should be back at center)
            queries.append(KeyframeQuery(
                video_path=video_path,
                frame_index=frame1_idx,
                expected_answer=final_position_answer,
                metadata={
                    "variant": variant,
                    "rotating_bot": variant,  # In this case, the bot itself is rotating
                    "query_type": "player_position_final",
                    "sneak_frame": bot_sneak_frame,
                    "latest_sneak_frame": latest_sneak_frame,  # Track which sneak frame was used for frame1
                    "rotation_frame": rotation_frame,
                    "rotation_direction": rotation_direction,
                    "frame1": frame1_idx,
                    "frame2": frame2_idx,
                    "yaw1": data[frame1_idx].get("yaw", 0),
                    "yaw2": data[frame2_idx].get("yaw", 0),
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num
                }
            ))

        return queries

    def _find_stop_turning_frame(
        self, data: List[dict], frame1_idx: int, threshold: float = 0.0001
    ) -> Optional[int]:
        """
        Find the frame where the bot stops turning.

        Starts searching 40 frames after frame1_idx, finds the first frame
        with no camera movement, and adds 10.

        Args:
            data: JSON data containing frame information
            frame1_idx: Starting frame index
            threshold: Threshold for camera movement detection

        Returns:
            Frame index where bot has stopped turning, or None if not found
        """
        search_start = frame1_idx + 40

        for i in range(search_start, len(data)):
            action = data[i].get("action", {})
            camera = action.get("camera", [0, 0])

            # Check if there's no significant camera movement
            if abs(camera[0]) <= threshold and abs(camera[1]) <= threshold:
                return i + 10

        return None

    def _calculate_position_answer(
        self, data: List[dict], frame1_idx: int, frame2_idx: int
    ) -> str:
        """
        Calculate expected player position based on yaw difference.

        Args:
            data: JSON data containing frame information
            frame1_idx: Index of first frame
            frame2_idx: Index of second frame

        Returns:
            Expected answer: "left", "right", or "no player"

        Raises:
            ValueError: If yaw difference doesn't match expected patterns
        """
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
        print(f"yaw_diff_deg: {yaw_diff_deg}")

        # Check if it's a ~40 degree rotation (+/- 5 degrees)
        if 35 <= abs(yaw_diff_deg) <= 45:
            # Positive = turned right (camera moved right, player appears left)
            # Negative = turned left (camera moved left, player appears right)
            if yaw_diff_deg > 0:
                return "left"
            else:
                return "right"

        # Check if it's a 90 degree rotation (+/- 30 degrees)
        elif 90-30 <= abs(yaw_diff_deg) <= 90+30:
            return "no player"

        # Unexpected yaw difference
        else:
            raise ValueError(
                f"Unexpected yaw difference for position query: {yaw_diff_deg:.2f} degrees "
                f"(frame {frame1_idx}: {math.degrees(yaw1):.2f}° -> "
                f"frame {frame2_idx}: {math.degrees(yaw2):.2f}°)"
            )

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
            Expected answer: "left", "right", "no player", or "same position"

        Raises:
            ValueError: If yaw difference doesn't match expected patterns
        """
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
            # Positive = turned right (camera moved right, player appears left)
            # Negative = turned left (camera moved left, player appears right)
            if yaw_diff_deg > 0:
                return "left"
            else:
                return "right"

        # Check if it's a ~180 degree rotation (+/- 5 degrees)
        elif 175 <= abs(yaw_diff_deg) <= 185:
            return "no player"

        # Check if yaw returned to near original position (looks away and back)
        # Small difference means back to original
        elif abs(yaw_diff_deg) <= 5:
            return "same position"

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
