#!/usr/bin/env python3
"""Handler for the co-movement evals, where BOTH bots move simultaneously.

This is translationEval's harder sibling. In translationEval exactly one bot
sneaks and moves while the other stands still, so the answer follows directly
from the mover's action label and is the same from both cameras. Here both bots
sneak and both move at the same time, which changes two things:

1. The answer is a *relative* quantity. Half the episodes pair actions that
   cancel — alpha forward with bravo back, or alpha left with bravo right —
   and because the bots face each other those move them the same way through
   the world, leaving the on-screen relationship unchanged. The correct answer
   there is "no motion" even though both players are walking.

2. The two cameras can disagree, so each perspective gets its own expected
   answer rather than sharing one.

Expected answers are therefore computed geometrically from the recorded x/z/yaw
rather than from action labels: the other bot's displacement is projected onto
the observer's forward and right axes and the dominant component wins. That
method reproduces translationEval's action-derived answers 64/64, which is what
justifies trusting it here (see analyze_comovement.py).

Episode structure, consistent across all 64 pairs of both datasets:

    ~f48   both bots sneak                        (episode start)
    ~f49   chunk 0: both walk forward             (shared approach)
    ~f90   chunk 1: the tested co-movement        <- what we query
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery
from handlers.camera_utils import find_end_of_first_sneak_chunk

DIRECTIONS = ("forward", "back", "left", "right")

# Frames after the tested chunk begins, matching translationEval's +40.
QUERY_OFFSET = 40

# Displacement below this in both axes reads as no motion (blocks).
DEADZONE = 0.75


class MinecraftCoMovementHandler(EpisodeTypeHandler):
    """Both bots move at once; each camera is judged on its own."""

    DATASET_NAMES = ["coMovementEval", "coMovementWithDividerEval"]

    def get_prompt(self, query_type: str = "co_movement") -> str:
        """Screen-relative phrasing, chosen by A/B against ground truth.

        translationEval's wording ("did the player move ... on-screen?") reads
        as a question about the world, which is a reasonable reading when the
        other player is visibly walking in every episode. It cost most of the
        "no motion" class: 56% recall here, and 3% on the divider variant,
        where a static block gives the model something to measure against.

        Making the frame of reference explicit — compare position and size
        *within the picture*, ignore blocks and landmarks, "no motion" holds
        even when the scenery moves — took "no motion" recall from 56% to 97%
        with all four motion classes still at 100% (prompt_ab_comovement.py,
        64 GT queries per variant).

        Deliberately says nothing about the action structure. A line such as
        "if both players walk the same way, answer no motion" would give away
        the answer for half the queries and inflate the score without
        measuring anything. The answer vocabulary matches translationEval so
        the two evals stay comparable.
        """
        return (
            "These are two screenshots from one player's camera, taken at two different "
            "moments. Another player is visible in both. "
            "The camera itself may have moved between the two screenshots, so the "
            "ground, the sky and any blocks or structures may shift between the two "
            "images. Ignore all of that. Do not judge the other player's position "
            "relative to any block, structure or landmark. "
            "Judge only how the other player appears within the picture itself: "
            "compare their position in the frame and how large they appear, in the "
            "first screenshot versus the second. "
            "If the other player appears at the same place in the frame and at the "
            "same size in both screenshots, answer \"no motion\", even if the scenery "
            "around them has moved. "
            "Answer with a single word from \"closer\", \"farther\", \"left\", \"right\", or \"no motion\"."
        )

    # --- geometry ---------------------------------------------------------

    @staticmethod
    def _camera_axes(yaw: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Forward and right unit vectors in (x, z).

        Derived from the data by holding each direction key and comparing the
        travel angle with yaw: forward is yaw+180 deg, right is yaw+90 deg.
        """
        return (-math.sin(yaw), -math.cos(yaw)), (math.cos(yaw), -math.sin(yaw))

    @classmethod
    def _apparent_motion(cls, obs: List[dict], other: List[dict],
                         f1: int, f2: int) -> Tuple[str, dict]:
        """Classify how ``other`` appears to move in ``obs``'s camera."""
        fwd, right = cls._camera_axes(obs[f1]["yaw"])

        def rel(f):
            return other[f]["x"] - obs[f]["x"], other[f]["z"] - obs[f]["z"]

        x1, z1 = rel(f1)
        x2, z2 = rel(f2)
        d_depth = (x2 * fwd[0] + z2 * fwd[1]) - (x1 * fwd[0] + z1 * fwd[1])
        d_lat = (x2 * right[0] + z2 * right[1]) - (x1 * right[0] + z1 * right[1])

        if abs(d_depth) < DEADZONE and abs(d_lat) < DEADZONE:
            answer = "no motion"
        elif abs(d_depth) >= abs(d_lat):
            answer = "closer" if d_depth < 0 else "farther"
        else:
            answer = "right" if d_lat > 0 else "left"

        return answer, {
            "delta_depth": round(d_depth, 3),
            "delta_lateral": round(d_lat, 3),
            "distance_frame1": round(math.hypot(x1, z1), 3),
            "distance_frame2": round(math.hypot(x2, z2), 3),
        }

    # --- keyframes --------------------------------------------------------

    @staticmethod
    def _movement_chunks(data: List[dict]) -> List[Tuple[int, str, int]]:
        out: List[list] = []
        for i, frame in enumerate(data):
            action = frame.get("action", {})
            for d in DIRECTIONS:
                if action.get(d):
                    if not out or out[-1][1] != d or i > out[-1][2] + 1:
                        out.append([i, d, i])
                    else:
                        out[-1][2] = i
        return [tuple(c) for c in out]

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        with open(video_pair.alpha_json) as f:
            alpha = json.load(f)
        with open(video_pair.bravo_json) as f:
            bravo = json.load(f)

        alpha_sneak = find_end_of_first_sneak_chunk(alpha)
        bravo_sneak = find_end_of_first_sneak_chunk(bravo)
        if alpha_sneak is None or bravo_sneak is None:
            raise ValueError(
                f"co-movement expects both bots to sneak; got alpha={alpha_sneak} "
                f"bravo={bravo_sneak} in episode {video_pair.episode_num} "
                f"instance {video_pair.instance_num}")

        alpha_chunks = self._movement_chunks(alpha)
        bravo_chunks = self._movement_chunks(bravo)
        if len(alpha_chunks) < 2 or len(bravo_chunks) < 2:
            raise ValueError(
                f"expected a shared approach then a tested co-movement; got "
                f"{len(alpha_chunks)}/{len(bravo_chunks)} chunks in episode "
                f"{video_pair.episode_num} instance {video_pair.instance_num}")

        # chunk 0 is the approach both bots share; chunk 1 is the tested move.
        alpha_start, alpha_dir, _ = alpha_chunks[1]
        bravo_start, bravo_dir, _ = bravo_chunks[1]

        # The bots start within a frame or two of each other; take the earlier
        # so the "before" frame precedes both.
        query_frame1 = min(alpha_start, bravo_start)
        usable = min(len(alpha), len(bravo))
        query_frame2 = min(query_frame1 + QUERY_OFFSET, usable - 1)
        if query_frame2 <= query_frame1:
            raise ValueError(
                f"not enough frames after the tested move in episode "
                f"{video_pair.episode_num} instance {video_pair.instance_num}")

        # The episode starts at the sneak, which is what generated videos are
        # rendered from; `frame1` keeps that meaning for offset arithmetic.
        episode_start = min(alpha_sneak, bravo_sneak)

        queries = []
        for variant, obs, other, video in (
            ("alpha", alpha, bravo, video_pair.alpha_video),
            ("bravo", bravo, alpha, video_pair.bravo_video),
        ):
            expected, detail = self._apparent_motion(
                obs, other, query_frame1, query_frame2)
            queries.append(KeyframeQuery(
                video_path=video,
                frame_index=query_frame1,
                second_frame_index=query_frame2,
                expected_answer=expected,
                metadata={
                    "variant": variant,
                    "query_type": "co_movement",
                    "alpha_direction": alpha_dir,
                    "bravo_direction": bravo_dir,
                    "observer_direction": alpha_dir if variant == "alpha" else bravo_dir,
                    "other_direction": bravo_dir if variant == "alpha" else alpha_dir,
                    "tested_chunk_start": query_frame1,
                    "query_frame1": query_frame1,
                    "query_frame2": query_frame2,
                    # Episode start, i.e. generated frame 0 is GT frame1 + 1.
                    "frame1": episode_start,
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num,
                    **detail,
                }
            ))
        return queries

    def validate_response(self, response: str, expected: str) -> bool:
        return response.strip().lower() == expected.strip().lower()
