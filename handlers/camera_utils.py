#!/usr/bin/env python3
"""
Camera utilities for computing absolute yaw/pitch values from action deltas.

The raw "yaw" field in frame data is unreliable. Instead, we accumulate
the camera action deltas from frame 0 to compute absolute camera orientation.

All values are in RADIANS:
- action['camera'][0] is yaw delta in radians (positive = turn right)
- action['camera'][1] is pitch delta in radians (positive = look down)
"""

import math
from typing import List, Optional, Tuple

# The delay (in frames) after the sneak chunk ends when the episode actually starts.
# This buffer accounts for settling time after the sneak action completes.
SNEAK_FRAME_START_DELAY = 25


def normalize_radians(angle):
    """
    Normalizes a given angle (in radians) to the range (-pi, pi].
    """
    two_pi = 2 * math.pi
    shifted_angle = angle + math.pi
    wrapped_angle = shifted_angle % two_pi
    normalized_angle = wrapped_angle - math.pi
    return normalized_angle


def normalize_degrees(angle: float) -> float:
    """
    Normalize an angle in degrees to the range [-180, 180].
    """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    return angle


def get_accumulated_yaw(data: List[dict], frame_idx: int) -> float:
    """
    Compute the absolute yaw at a given frame by accumulating camera deltas.
    
    Args:
        data: List of frame dictionaries containing action data
        frame_idx: Target frame index
        
    Returns:
        Accumulated yaw value in radians
    """
    accumulated_yaw = 0.0
    for i in range(min(frame_idx + 1, len(data))):
        camera = data[i].get("action", {}).get("camera", [0, 0])
        accumulated_yaw += camera[0]
    return accumulated_yaw


def get_accumulated_pitch(data: List[dict], frame_idx: int) -> float:
    """
    Compute the absolute pitch at a given frame by accumulating camera deltas.
    
    Args:
        data: List of frame dictionaries containing action data
        frame_idx: Target frame index
        
    Returns:
        Accumulated pitch value in radians
    """
    accumulated_pitch = 0.0
    for i in range(min(frame_idx + 1, len(data))):
        camera = data[i].get("action", {}).get("camera", [0, 0])
        accumulated_pitch += camera[1]
    return accumulated_pitch


def get_accumulated_camera(data: List[dict], frame_idx: int) -> Tuple[float, float]:
    """
    Compute both absolute yaw and pitch at a given frame by accumulating camera deltas.
    
    Args:
        data: List of frame dictionaries containing action data
        frame_idx: Target frame index
        
    Returns:
        Tuple of (accumulated_yaw, accumulated_pitch) in radians
    """
    accumulated_yaw = 0.0
    accumulated_pitch = 0.0
    for i in range(min(frame_idx + 1, len(data))):
        camera = data[i].get("action", {}).get("camera", [0, 0])
        accumulated_yaw += camera[0]
        accumulated_pitch += camera[1]
    return accumulated_yaw, accumulated_pitch


def get_yaw_difference(data: List[dict], frame1_idx: int, frame2_idx: int) -> float:
    """
    Compute the yaw difference between two frames by accumulating camera deltas.
    
    More efficient than computing two full accumulations when you only need
    the difference.
    
    Args:
        data: List of frame dictionaries containing action data
        frame1_idx: First frame index
        frame2_idx: Second frame index
        
    Returns:
        Yaw difference in radians (yaw2 - yaw1)
    """
    # Only need to sum deltas between the two frames
    if frame1_idx > frame2_idx:
        # Swap and negate
        start, end = frame2_idx, frame1_idx
        sign = -1
    else:
        start, end = frame1_idx, frame2_idx
        sign = 1
    
    yaw_diff = 0.0
    for i in range(start + 1, min(end + 1, len(data))):
        camera = data[i].get("action", {}).get("camera", [0, 0])
        yaw_diff += camera[0]
    
    return sign * yaw_diff


def find_end_of_first_sneak_chunk(
    data: List[dict], 
    buffer: int = SNEAK_FRAME_START_DELAY
) -> Optional[int]:
    """
    Find the frame where the episode actually starts (after the first sneak chunk).
    
    Episodes may have multiple sneak chunks, but we only want the end of the
    first one, which marks when the episode actually begins. A buffer is added
    to account for settling time after the sneak action completes.
    
    Args:
        data: List of frame dictionaries containing action data
        buffer: Number of frames to add after the sneak chunk ends (default: 25).
                Set to 0 to get the raw last frame of the sneak chunk.
        
    Returns:
        Index of the episode start frame (last sneak frame + buffer),
        or None if no sneak frames found
    """
    in_sneak_chunk = False
    last_sneak_in_chunk = None
    
    for i, frame in enumerate(data):
        is_sneaking = frame.get("action", {})["sneak"]
        
        if is_sneaking:
            in_sneak_chunk = True
            last_sneak_in_chunk = i
        elif in_sneak_chunk:
            # We were in a sneak chunk but now sneak is False
            # This means the first chunk has ended
            break
    
    if last_sneak_in_chunk is None:
        raise ValueError(f"No sneak frames found in data!")
    return last_sneak_in_chunk + buffer


def find_camera_rotation_frame(
    data: List[dict], start_frame: int, threshold: float = 0.0001
) -> Tuple[Optional[int], Optional[str]]:
    """
    Find the first frame after start_frame with camera rotation.

    Args:
        data: List of frame dictionaries containing action data
        start_frame: Frame index to start searching from
        threshold: Minimum camera movement to consider as rotation

    Returns:
        Tuple of (frame_index, direction) where direction describes the
        camera rotation (e.g., "left", "right", "up", "down").
        Returns (None, None) if no rotation found.
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


def find_stop_turning_frame(
    data: List[dict], frame1_idx: int, threshold: float = 0.0001
) -> Optional[int]:
    """
    Find the frame where the bot stops turning.

    Starts at frame1_idx, waits for camera to start turning, then returns
    the first frame where turning stops + 10 (+0.5s).

    Args:
        data: List of frame dictionaries containing action data
        frame1_idx: Starting frame index
        threshold: Threshold for camera movement detection

    Returns:
        Frame index where bot has stopped turning, or None if not found
    """
    # First, find when the camera starts turning
    turning_started = False
    for i in range(frame1_idx, len(data)):
        action = data[i].get("action", {})
        camera = action.get("camera", [0, 0])

        is_turning = abs(camera[0]) > threshold or abs(camera[1]) > threshold

        if not turning_started:
            if is_turning:
                turning_started = True
        else:
            # We were turning, check if we've stopped
            if not is_turning:
                return i + 10  #Wait 0.5s for the camera to stabilize

    return None


def calculate_position_answer(
    data: List[dict], frame1_idx: int, frame2_idx: int
) -> str:
    """
    Calculate expected player position based on yaw difference between two frames.

    Uses accumulated camera deltas to compute reliable yaw values, then determines
    the expected player position based on how much the camera has rotated.

    Args:
        data: List of frame dictionaries containing action data
        frame1_idx: Index of first (reference) frame
        frame2_idx: Index of second frame

    Returns:
        Expected answer: "left", "right", or "no player"

    Raises:
        ValueError: If yaw difference doesn't match expected patterns
    """
    yaw_diff_rad = get_yaw_difference(data, frame1_idx, frame2_idx)
    
    # Normalize to [-π, π] range and convert to degrees
    yaw_diff_rad = normalize_radians(yaw_diff_rad)
    yaw_diff_deg = math.degrees(yaw_diff_rad)
    print(f"yaw_diff_deg: {yaw_diff_deg}")

    # Check if it's a ~40 degree rotation (+/- 5 degrees)
    if 35 <= abs(yaw_diff_deg) <= 45:
        # Negative = turned right (camera moved right, player appears left)
        # Positive = turned left (camera moved left, player appears right)
        if yaw_diff_deg < 0:
            return "left"
        else:
            return "right"

    # Check if it's a ~90 degree rotation (+/- 30 degrees) -> no player (one/both look away eval episode)
    elif 90 - 30 <= abs(yaw_diff_deg) <= 90 + 30:
        return "no player"

    # Check if it's a ~180 degree rotation (+/- 5 degrees) -> no player (rotation eval episode)
    elif 175 <= abs(yaw_diff_deg) <= 185:
        return "no player"

    # Unexpected yaw difference
    else:
        raise ValueError(
            f"Unexpected yaw difference for position query: {yaw_diff_deg:.2f} degrees "
            f"(frame {frame1_idx} -> frame {frame2_idx})"
        )

