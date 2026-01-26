#!/usr/bin/env python3
"""
Handler for mc_multiplayer_eval_both_look_away dataset.

Both bots look away and then turn back. This handler creates three queries
per bot (6 total) in chronological order:

1. player_position_during_turn: Bot is mid-turn, other player should be visible
   to the left or right of the screen (single frame query)
   
2. player_invisible_looked_away: Bot has fully turned away, other player should
   NOT be visible on screen (both bots turn away from each other) (single frame query)
   
3. player_position_turned_back: Bot has turned back to original orientation,
   other player should be back at center (single frame query)
"""

import json
from pathlib import Path
from typing import List
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery
from handlers.camera_utils import (
    get_accumulated_yaw,
    find_end_of_first_sneak_chunk,
    find_camera_rotation_frame,
    find_stop_turning_frame,
)


class MinecraftBothLookAwayHandler(EpisodeTypeHandler):
    """
    Handler for Minecraft "both look away" evaluation.

    Both bots rotate away from each other and then back.
    """

    DATASET_NAMES = ["bothLookAwayEval"]

    def get_prompt(self, query_type: str = "player_position_during_turn") -> str:
        # Single frame query: bot has turned away but other player should still be visible
        # (because both bots turn away, they still face each other)
        return (
            "Here is a Minecraft screenshot. "
            "Is there another player visible on-screen? "
            "Answer with a single word: \"yes\", \"no\"."
        )

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes based on sneak and camera rotation.

        Creates three queries per bot in chronological order:
        1. player_position_during_turn: While bot is turning, other player visible left/right
        2. player_invisible_looked_away: When bot has turned away, other player NOT visible
        3. player_position_turned_back: When bot turns back, other player at center again

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
        alpha_sneak_frame = find_end_of_first_sneak_chunk(alpha_data)
        bravo_sneak_frame = find_end_of_first_sneak_chunk(bravo_data)

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
            rotation_frame, rotation_direction = find_camera_rotation_frame(
                data, bot_sneak_frame
            )

            if rotation_frame is None or rotation_direction is None:
                continue

            # Calculate keyframe indices using the LATEST sneak frame
            frame1_idx = latest_sneak_frame  # Reference frame (before turning)
            
            
            # Frame when fully turned away: find when camera stops moving
            looked_away_frame_idx = find_stop_turning_frame(data, frame1_idx)
            if looked_away_frame_idx is None:
                raise ValueError(f"No stop turning frame found for {variant} in episode {video_pair.episode_num} instance {video_pair.instance_num}")
            
            # Frame when turned back: rotation_frame + 200 (bot has returned to original orientation)
            turned_back_frame_idx = rotation_frame + 200



            # Query 1 (chronological): Fully looked away - player NOT visible
            # Note: Single-frame query, only frame_index is sent to VLM
            queries.append(KeyframeQuery(
                video_path=video_path,
                frame_index=looked_away_frame_idx,
                expected_answer="no",
                metadata={
                    "variant": variant,
                    "rotating_bot": variant,
                    "query_type": "player_invisible_looked_away",
                    "sneak_frame": bot_sneak_frame,
                    "latest_sneak_frame": latest_sneak_frame,
                    "rotation_frame": rotation_frame,
                    "rotation_direction": rotation_direction,
                    "frame1": frame1_idx,
                    "yaw1": get_accumulated_yaw(data, frame1_idx),
                    "yaw2": get_accumulated_yaw(data, looked_away_frame_idx),
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num
                }
            ))

            # Query 2 (chronological): Turned back - player should be back at center
            # Note: Single-frame query, only frame_index is sent to VLM
            queries.append(KeyframeQuery(
                video_path=video_path,
                frame_index=turned_back_frame_idx,
                expected_answer="yes",
                metadata={
                    "variant": variant,
                    "rotating_bot": variant,
                    "query_type": "player_position_turned_back",
                    "sneak_frame": bot_sneak_frame,
                    "latest_sneak_frame": latest_sneak_frame,
                    "rotation_frame": rotation_frame,
                    "rotation_direction": rotation_direction,
                    "frame1": frame1_idx,
                    "yaw1": get_accumulated_yaw(data, frame1_idx),
                    "yaw2": get_accumulated_yaw(data, turned_back_frame_idx),
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num
                }
            ))

        return queries
