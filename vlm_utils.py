#!/usr/bin/env python3
"""
VLM Evaluation Utilities

This module provides utility functions and data structures for VLM evaluation:
- Data structures for video pairs, queries, and results
- Frame extraction from videos (both ground-truth and generated)
- VLM querying via Gemini API
- Result saving and formatting
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class VideoPair:
    """Represents a pair of videos (Alpha/Bravo) with the same episode and instance number."""
    episode_num: str
    instance_num: str
    alpha_video: Path
    bravo_video: Path
    alpha_json: Path
    bravo_json: Path

    def __repr__(self):
        return f"VideoPair(episode={self.episode_num}, instance={self.instance_num})"


@dataclass
class KeyframeQuery:
    """Represents a single keyframe query with expected answer."""
    video_path: Path
    frame_index: int
    expected_answer: str
    metadata: Optional[Dict] = None  # Additional context for the query

    def __repr__(self):
        return f"KeyframeQuery(video={self.video_path.name}, frame={self.frame_index})"


@dataclass
class EvalResult:
    """Results from evaluating a single keyframe query."""
    query: KeyframeQuery
    vlm_response: str
    is_correct: bool
    is_unclear: bool = False  # Whether the VLM response was "unclear"
    metadata: Optional[Dict] = None


class EpisodeTypeHandler(ABC):
    """
    Abstract base class for episode type handlers.

    Each episode type should implement its own handler that defines:
    - How to extract keyframes from video pairs
    - What prompt to use for the VLM
    - How to parse and validate VLM responses
    """

    # List of exact dataset names this handler supports
    # Subclasses should override this with their specific dataset names
    DATASET_NAMES: List[str] = []

    @abstractmethod
    def get_prompt(self) -> str:
        """
        Return the prompt template to use with the VLM.

        The prompt can use placeholders like {frame_description} if needed.
        """
        pass

    @abstractmethod
    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes from a video pair and return queries with expected answers.

        This method should:
        1. Read the JSON files for the video pair
        2. Determine which frames to extract based on the actions
        3. Return a list of KeyframeQuery objects with expected answers

        Args:
            video_pair: A VideoPair object containing paths to videos and JSONs

        Returns:
            List of KeyframeQuery objects
        """
        pass

    def validate_response(self, response: str, expected: str) -> bool:
        """
        Validate the VLM response against the expected answer.

        Default implementation does exact string matching.
        Override for custom validation logic.
        """
        return response.strip().lower() == expected.strip().lower()


def extract_frame(video_path: Path, frame_index: int) -> bytes:
    """
    Extract a specific frame from a video as image bytes, resized to 640x360.

    Args:
        video_path: Path to the video file
        frame_index: Zero-indexed frame number to extract

    Returns:
        Image bytes (PNG format), resized to 640x360
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Resize to 640x360 to match generated video quadrant size
        frame_resized = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.png', frame_resized)
        return buffer.tobytes()
    else:
        raise ValueError(f"Could not extract frame {frame_index} from {video_path}")


def extract_quadrant(frame, quadrant: str):
    """
    Extract a specific quadrant from a side-by-side video frame.

    Args:
        frame: Full video frame (1280x720) for Oasis or (1280x704) for Matrix-Game 2
        quadrant: One of "top-left", "top-right", "bottom-left", "bottom-right"

    Returns:
        Extracted quadrant frame (640x360) for Oasis or (640x352) for Matrix-Game 2
    """
    height, width = frame.shape[:2]

    # Verify expected dimensions
    if width != 1280 or (height != 720 and height != 704):
        raise ValueError(f"Expected frame size 1280x720, got {width}x{height}")

    quad_width = 640
    quad_height = 360 if height == 720 else 352

    if quadrant == "top-left":
        return frame[0:quad_height, 0:quad_width]
    elif quadrant == "top-right":
        return frame[0:quad_height, quad_width:width]
    elif quadrant == "bottom-left":
        return frame[quad_height:height, 0:quad_width]
    elif quadrant == "bottom-right":
        return frame[quad_height:height, quad_width:width]
    else:
        raise ValueError(f"Invalid quadrant: {quadrant}")


def find_generated_video_subdir(generated_base_path: Path, dataset_name: str) -> Optional[Path]:
    """
    Find the generated video subdirectory that matches the dataset name.

    Args:
        generated_base_path: Base path to generated videos
        dataset_name: Dataset name (e.g., "mc_multiplayer_eval_translation")

    Returns:
        Path to the subdirectory containing generated videos, or None if not found
    """
    # Extract the key part of the dataset name (e.g., "translation", "rotation")
    dataset_key = None
    for key in ["translation", "rotation", "both_look_away", "one_looks_away", "looks_away", "structure", "turn_to_look", "turnToLook"]:
        if key in dataset_name:
            dataset_key = key
            break

    if not dataset_key:
        return None

    # Look for subdirectories matching the pattern
    for subdir in generated_base_path.iterdir():
        if subdir.is_dir() and dataset_key in subdir.name:
            return subdir

    return None


def extract_frame_from_generated(
    generated_video_path: Path,
    frame_index_gt: int,
    frame1_idx_gt: int,
    variant: str
) -> bytes:
    """
    Extract a frame from a generated side-by-side video.

    Args:
        generated_video_path: Path to the generated video_X_side_by_side.mp4
        frame_index_gt: Ground-truth frame index to extract
        frame1_idx_gt: Ground-truth frame1_idx (sneak_frame + 5) - this is frame 0 in generated video
        variant: "alpha" or "bravo" to determine which quadrant to extract

    Returns:
        Image bytes (PNG format) of the extracted quadrant
    """
    import cv2

    # Calculate the frame index in the generated video
    # Generated video frame 0 corresponds to GT frame (frame1_idx + 1)
    generated_frame_idx = frame_index_gt - frame1_idx_gt - 1

    if generated_frame_idx < 0:
        raise ValueError(
            f"Frame index {frame_index_gt} is before the generated video start "
            f"(starts at GT frame {frame1_idx_gt + 1})"
        )

    # Open the generated video
    cap = cv2.VideoCapture(str(generated_video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, generated_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(
            f"Could not extract frame {generated_frame_idx} from {generated_video_path}"
        )

    # Extract the appropriate quadrant
    # Top-right: alpha generated
    # Bottom-right: bravo generated
    quadrant = "top-right" if variant == "alpha" else "bottom-right"
    extracted_frame = extract_quadrant(frame, quadrant)

    # Encode to PNG
    _, buffer = cv2.imencode('.png', extracted_frame)
    return buffer.tobytes()


def query_vlm(prompt: str, image_bytes: bytes, image_bytes_2: Optional[bytes] = None) -> str:
    """
    Query the VLM (e.g., Gemini) with a prompt and image(s).

    Args:
        prompt: The text prompt for the VLM
        image_bytes: First image data as bytes
        image_bytes_2: Optional second image data as bytes

    Returns:
        VLM response as string
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    # Build content parts
    content_parts = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png',
        )
    ]

    # Add second image if provided
    if image_bytes_2 is not None:
        content_parts.append(
            types.Part.from_bytes(
                data=image_bytes_2,
                mime_type='image/png',
            )
        )

    # Add the prompt
    content_parts.append(prompt)

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=content_parts,
        config=types.GenerateContentConfig(
            system_instruction='You are a helpful assistant that evaluates on-screen motion of Minecraft characters between two screenshots.',
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
        ),
    )

    return response.text.strip().lower()


def save_results(results: List[EvalResult], output_path: str):
    """
    Save evaluation results to a JSON file.

    Args:
        results: List of EvalResult objects
        output_path: Path to save the JSON file
    """
    # Calculate overall statistics
    total = len(results)
    unclear_count = sum(1 for r in results if r.is_unclear)
    evaluable = total - unclear_count
    correct = sum(1 for r in results if r.is_correct and not r.is_unclear)
    accuracy_excluding_unclear = (correct / evaluable * 100) if evaluable > 0 else 0
    accuracy_total = (correct / total * 100) if total > 0 else 0  # Treats unclear as incorrect

    # Calculate breakdown by query type
    query_types = set(r.metadata.get('query_type', 'default') for r in results if r.metadata)
    breakdown_by_query_type = {}

    for qtype in sorted(query_types):
        type_results = [r for r in results if r.metadata and r.metadata.get('query_type', 'default') == qtype]
        type_total = len(type_results)
        type_unclear = sum(1 for r in type_results if r.is_unclear)
        type_evaluable = type_total - type_unclear
        type_correct = sum(1 for r in type_results if r.is_correct and not r.is_unclear)

        breakdown_by_query_type[qtype] = {
            "total": type_total,
            "unclear_count": type_unclear,
            "unclear_percentage": (type_unclear / type_total * 100) if type_total > 0 else 0,
            "evaluable": type_evaluable,
            "correct": type_correct,
            "accuracy_excluding_unclear": (type_correct / type_evaluable * 100) if type_evaluable > 0 else 0,
            "accuracy_total": (type_correct / type_total * 100) if type_total > 0 else 0
        }

    output_data = {
        "total_queries": total,
        "unclear_count": unclear_count,
        "unclear_percentage": (unclear_count / total * 100) if total > 0 else 0,
        "evaluable_queries": evaluable,
        "correct": correct,
        "accuracy_excluding_unclear": accuracy_excluding_unclear,
        "accuracy_total": accuracy_total,
        "breakdown_by_query_type": breakdown_by_query_type,
        "results": [
            {
                "video": str(r.query.video_path.name),
                "frame_index": r.query.frame_index,
                "expected": r.query.expected_answer,
                "response": r.vlm_response,
                "correct": r.is_correct,
                "unclear": r.is_unclear,
                "metadata": r.metadata
            }
            for r in results
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
