#!/usr/bin/env python3
"""
Unified evaluation script for all mc_multiplayer datasets.

Usage:
  python run_eval.py <folder> [options]

The folder name determines which handler to use:
  - mc_multiplayer_eval_dev     -> MinecraftMultiplayerHandler (movement)
  - mc_multiplayer_eval_dev_2   -> MinecraftCameraRotationHandler (camera rotation)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vlm_utils import (
    VideoPair,
    extract_frame,
    query_vlm,
    save_results,
    EvalResult,
    extract_quadrant,
    find_generated_video_subdir,
    extract_frame_from_generated,
    VLM_MODEL_NAME
)


def find_mc_video_pairs(folder: Path) -> List[VideoPair]:
    """
    Find video pairs in mc_multiplayer format (with _camera.mp4 suffix).

    Matches: {episode}_{Alpha|Bravo}_instance_{instance}_camera.mp4
    """
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


def identify_handler(folder_name: str, summary_json_path: str = None):
    """
    Identify which handler to use based on folder name.

    Uses exact dataset name matching based on handler's DATASET_NAMES attribute.

    Args:
        folder_name: Name of the dataset folder
        summary_json_path: Optional path to structure_building_summary.json (for structure handler)

    Returns:
        Handler instance
    """
    from handlers import (
        MinecraftTranslationHandler,
        MinecraftRotationHandler,
        MinecraftLooksAwayHandler,
        MinecraftBothLookAwayHandler,
        MinecraftStructureBuildingHandler,
        MinecraftStructureNoPlaceHandler,
        MinecraftTurnToLookHandler,
        MinecraftTurnToLookOppositeHandler
    )

    # List of all handler classes (order doesn't matter for exact matching)
    handler_classes = [
        MinecraftTranslationHandler,
        MinecraftRotationHandler,
        MinecraftLooksAwayHandler,
        MinecraftBothLookAwayHandler,
        MinecraftTurnToLookHandler,
        MinecraftTurnToLookOppositeHandler,
    ]

    # Structure handlers require summary_json_path
    structure_handler_classes = [
        MinecraftStructureBuildingHandler,
        MinecraftStructureNoPlaceHandler,
    ]

    # Check each handler for exact dataset name match
    for handler_class in handler_classes:
        if folder_name in handler_class.DATASET_NAMES:
            return handler_class()

    # Special case for structure handlers (require summary_json_path)
    for handler_class in structure_handler_classes:
        if folder_name in handler_class.DATASET_NAMES:
            if not summary_json_path:
                # Use correct default summary JSON based on handler type
                if handler_class == MinecraftStructureNoPlaceHandler:
                    summary_json_path = str(Path(__file__).parent / "assets" / "hard_coded_gt" / "structure_building_no_place_summary.json")
                else:
                    summary_json_path = str(Path(__file__).parent / "assets" / "hard_coded_gt" / "structure_building_summary.json")
            return handler_class(summary_json_path)

    # No exact match found
    all_dataset_names = []
    for handler_class in handler_classes:
        all_dataset_names.extend(handler_class.DATASET_NAMES)
    for handler_class in structure_handler_classes:
        all_dataset_names.extend(handler_class.DATASET_NAMES)

    raise ValueError(
        f"Cannot identify handler for dataset: {folder_name}\n"
        f"No handler found with matching DATASET_NAMES.\n"
        f"Available dataset names: {sorted(all_dataset_names)}"
    )


def dry_run(handler, video_pairs: List[VideoPair], limit: Optional[int] = None):
    """
    View keyframe info without extracting frames or querying VLM.

    Args:
        handler: Episode type handler
        video_pairs: List of video pairs
        limit: Optional limit on number of episodes
    """
    if limit:
        video_pairs = video_pairs[:limit]

    print(f"\n{'='*80}")
    print(f"DRY RUN: Keyframe Detection")
    print(f"Handler: {handler.__class__.__name__}")
    print(f"Episodes: {len(video_pairs)}")
    print(f"{'='*80}\n")

    for i, pair in enumerate(video_pairs, 1):
        print(f"Episode {i}: {pair.episode_num} (instance {pair.instance_num})")
        print("-" * 80)

        queries = handler.extract_keyframes(pair)

        if not queries:
            print("  ⚠ No movement/rotation found in this episode\n")
            continue

        # Print info from first query (they share same metadata)
        meta = queries[0].metadata
        
        # Handle different handler types
        if 'builder' in meta:
            # Structure handler
            print(f"  Builder bot: {meta['builder'].upper()}")
            print(f"  Observer bot: {meta['variant'].upper()}")
            print(f"  Structure: {meta['structure']}")
        elif 'moving_bot' in meta:
            # Translation handler
            print(f"  Moving bot: {meta['moving_bot'].upper()}")
            print(f"  Sneak frame: {meta['sneak_frame']}")
            print(f"  Movement frame: {meta['movement_frame']}")
            print(f"  Movement direction: {meta['movement_direction']}")
        elif 'rotating_bot' in meta:
            # Rotation/looks_away handlers
            print(f"  Rotating bot: {meta['rotating_bot'].upper()}")
            print(f"  Sneak frame: {meta['sneak_frame']}")
            print(f"  Rotation frame: {meta['rotation_frame']}")
            print(f"  Rotation direction: {meta['rotation_direction']}")
            if 'yaw1' in meta and 'yaw2' in meta:
                import math
                yaw_diff = math.degrees(meta['yaw2'] - meta['yaw1'])
                print(f"  Yaw difference: {yaw_diff:.2f}°")

        print(f"  Frame 1: {meta['frame1']}")
        print(f"  Frame 2: {meta['frame2']}")
        print(f"  Expected answer: {queries[0].expected_answer}")
        print(f"  Perspectives: {len(queries)} (both Alpha and Bravo)\n")


def extract_frames_mode(handler, video_pairs: List[VideoPair], output_dir: str, limit: Optional[int] = None, generated_path: Optional[Path] = None, dataset_name: Optional[str] = None):
    """
    Extract frames for visual inspection.

    Args:
        handler: Episode type handler
        video_pairs: List of video pairs
        output_dir: Directory to save frames
        limit: Optional limit on number of episodes
        generated_path: Optional path to generated videos directory
        dataset_name: Dataset name (needed when using generated videos)
    """
    import cv2

    if limit:
        video_pairs = video_pairs[:limit]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXTRACTING FRAMES")
    print(f"Handler: {handler.__class__.__name__}")
    print(f"Output: {output_path}")
    print(f"Episodes: {len(video_pairs)}")
    if generated_path:
        print(f"Using generated videos from: {generated_path}")
    print(f"{'='*80}\n")

    # Find generated video subdirectory if using generated videos
    generated_subdir = None
    if generated_path:
        generated_subdir = find_generated_video_subdir(generated_path, dataset_name)
        if not generated_subdir:
            print(f"Error: Could not find generated video subdirectory for dataset '{dataset_name}'")
            return
        print(f"Found generated video subdirectory: {generated_subdir.name}")

        # Count available generated videos
        generated_videos = list(generated_subdir.glob("video_*_side_by_side.mp4"))
        num_generated = len(generated_videos)
        print(f"Found {num_generated} generated videos")

        # Limit video pairs to number of generated videos available
        if num_generated < len(video_pairs):
            print(f"Limiting extraction to first {num_generated} video pairs (fewer generated videos than GT)\n")
            video_pairs = video_pairs[:num_generated]
        else:
            print()

    # Extract keyframes
    all_queries = []
    for pair in video_pairs:
        queries = handler.extract_keyframes(pair)
        all_queries.extend(queries)

    print(f"Total frames to extract: {len(all_queries) * 2}")  # frame1 and frame2

    # Extract frames
    # Track which video pair we're currently processing
    current_video_id = -1
    last_episode_instance = None

    for i, query in enumerate(all_queries, 1):
        meta = query.metadata
        episode = meta['episode']
        instance = meta['instance']
        variant = meta['variant']
        frame1_idx = meta['frame1']
        frame2_idx = meta['frame2']

        print(f"[{i}/{len(all_queries)}] Episode {episode} instance {instance} - {variant}", end="... ")

        try:
            # Check if this is a co-observation query (needs frames from both videos)
            is_turn_to_look = meta.get('is_turn_to_look', False)

            if is_turn_to_look:
                # For co-observation, extract frame2 from both alpha and bravo
                alpha_video = Path(meta['alpha_video'])
                bravo_video = Path(meta['bravo_video'])
                alpha_frame = meta['alpha_frame']
                bravo_frame = meta['bravo_frame']

                if generated_path and generated_subdir:
                    # Use generated videos
                    episode_instance = (episode, instance)
                    if episode_instance != last_episode_instance:
                        current_video_id += 1
                        last_episode_instance = episode_instance

                    video_id = current_video_id
                    generated_video = generated_subdir / f"video_{video_id}_side_by_side.mp4"

                    if not generated_video.exists():
                        print(f"✗ Generated video not found: {generated_video.name}")
                        continue

                    # Extract frames from both alpha and bravo quadrants
                    alpha_bytes = extract_frame_from_generated(generated_video, alpha_frame, frame1_idx, "alpha")
                    bravo_bytes = extract_frame_from_generated(generated_video, bravo_frame, frame1_idx, "bravo")

                    # Save frames
                    alpha_path = output_path / f"ep{episode}_inst{instance}_alpha_frame2.png"
                    bravo_path = output_path / f"ep{episode}_inst{instance}_bravo_frame2.png"

                    with open(alpha_path, 'wb') as f:
                        f.write(alpha_bytes)
                    with open(bravo_path, 'wb') as f:
                        f.write(bravo_bytes)

                    print("✓")
                else:
                    # Use ground-truth videos
                    # Extract from alpha video
                    cap_alpha = cv2.VideoCapture(str(alpha_video))
                    cap_alpha.set(cv2.CAP_PROP_POS_FRAMES, alpha_frame)
                    ret_alpha, frame_alpha = cap_alpha.read()
                    cap_alpha.release()

                    # Extract from bravo video
                    cap_bravo = cv2.VideoCapture(str(bravo_video))
                    cap_bravo.set(cv2.CAP_PROP_POS_FRAMES, bravo_frame)
                    ret_bravo, frame_bravo = cap_bravo.read()
                    cap_bravo.release()

                    if ret_alpha and ret_bravo:
                        # Resize to 640x360 to match generated video resolution
                        frame_alpha_resized = cv2.resize(frame_alpha, (640, 360))
                        frame_bravo_resized = cv2.resize(frame_bravo, (640, 360))

                        alpha_path = output_path / f"ep{episode}_inst{instance}_alpha_frame2.png"
                        bravo_path = output_path / f"ep{episode}_inst{instance}_bravo_frame2.png"

                        cv2.imwrite(str(alpha_path), frame_alpha_resized)
                        cv2.imwrite(str(bravo_path), frame_bravo_resized)

                        print("✓")
                    else:
                        print("✗ Failed to read frames")
            else:
                # Standard handling for other query types
                if generated_path and generated_subdir:
                    # Use generated videos
                    # Track which video pair we're on by watching for new (episode, instance) combinations
                    episode_instance = (episode, instance)
                    if episode_instance != last_episode_instance:
                        current_video_id += 1
                        last_episode_instance = episode_instance

                    # The video pair index corresponds to the video_X in video_X_side_by_side.mp4
                    video_id = current_video_id
                    generated_video = generated_subdir / f"video_{video_id}_side_by_side.mp4"

                    if not generated_video.exists():
                        print(f"✗ Generated video not found: {generated_video.name}")
                        continue

                    # Extract frames from generated video with proper quadrant and offset
                    import io
                    from PIL import Image

                    # For generated videos, use frame1_idx + 1 as first frame (since gen video starts there)
                    frame1_bytes = extract_frame_from_generated(generated_video, frame1_idx + 1, frame1_idx, variant)
                    frame2_bytes = extract_frame_from_generated(generated_video, frame2_idx, frame1_idx, variant)

                    # Save frames
                    frame1_path = output_path / f"ep{episode}_inst{instance}_{variant}_frame1.png"
                    frame2_path = output_path / f"ep{episode}_inst{instance}_{variant}_frame2.png"

                    with open(frame1_path, 'wb') as f:
                        f.write(frame1_bytes)
                    with open(frame2_path, 'wb') as f:
                        f.write(frame2_bytes)

                    print("✓")
                else:
                    # Use ground-truth videos
                    cap = cv2.VideoCapture(str(query.video_path))

                    # Extract frame 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
                    ret1, frame1 = cap.read()

                    # Extract frame 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
                    ret2, frame2 = cap.read()

                    cap.release()

                    if ret1 and ret2:
                        # Resize to 640x360 to match generated video resolution
                        frame1_resized = cv2.resize(frame1, (640, 360))
                        frame2_resized = cv2.resize(frame2, (640, 360))

                        frame1_path = output_path / f"ep{episode}_inst{instance}_{variant}_frame1.png"
                        frame2_path = output_path / f"ep{episode}_inst{instance}_{variant}_frame2.png"

                        cv2.imwrite(str(frame1_path), frame1_resized)
                        cv2.imwrite(str(frame2_path), frame2_resized)

                        print("✓")
                    else:
                        print("✗ Failed to read frames")
        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n{'='*80}")
    print(f"Frames saved to: {output_path.absolute()}")
    print(f"{'='*80}")


def run_evaluation(handler, video_pairs: List[VideoPair], output_file: str, limit: Optional[int] = None, generated_path: Optional[Path] = None, dataset_name: Optional[str] = None, model_name: str = "ground_truth"):
    """
    Run full VLM evaluation.

    Args:
        handler: Episode type handler
        video_pairs: List of video pairs
        output_file: Path to save results JSON
        limit: Optional limit on number of queries
        generated_path: Optional path to generated videos directory
        dataset_name: Dataset name (needed when using generated videos)
        model_name: Name of our video generation model being evaluated, or "ground_truth" for GT videos
    """
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠ ERROR: GEMINI_API_KEY environment variable not set!")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Use handler's enable_vlm_thinking property
    enable_thinking = handler.enable_vlm_thinking
    thinking_status = "default" if enable_thinking else "disabled"

    print(f"\n{'='*80}")
    print(f"VLM EVALUATION")
    print(f"Handler: {handler.__class__.__name__}")
    print(f"Model: {VLM_MODEL_NAME} (thinking {thinking_status})")
    print(f"Output: {output_file}")
    if generated_path:
        print(f"Using generated videos from: {generated_path}")
    print(f"{'='*80}\n")

    # Find generated video subdirectory if using generated videos
    generated_subdir = None
    if generated_path:
        generated_subdir = find_generated_video_subdir(generated_path, dataset_name)
        if not generated_subdir:
            print(f"Error: Could not find generated video subdirectory for dataset '{dataset_name}'")
            return []
        print(f"Found generated video subdirectory: {generated_subdir.name}")

        # Count available generated videos
        generated_videos = list(generated_subdir.glob("video_*_side_by_side.mp4"))
        num_generated = len(generated_videos)
        print(f"Found {num_generated} generated videos")

        # Limit video pairs to number of generated videos available
        if num_generated < len(video_pairs):
            print(f"Limiting evaluation to first {num_generated} video pairs (fewer generated videos than GT)\n")
            video_pairs = video_pairs[:num_generated]
        else:
            print()

    # Extract keyframes
    print("Extracting keyframe queries...")
    all_queries = []
    for pair in video_pairs:
        queries = handler.extract_keyframes(pair)

        # For rotation and one_looks_away, only query from rotating bot's perspective
        # For both_look_away, keep both perspectives since both bots rotate
        # For structure, only query from observer's perspective (already filtered by handler)
        handler_name = handler.__class__.__name__
        if "Rotation" in handler_name and "Both" not in handler_name and queries:
            rotating_bot = queries[0].metadata['rotating_bot']
            queries = [q for q in queries if q.metadata['variant'] == rotating_bot]
        elif "LooksAway" in handler_name and "Both" not in handler_name and queries:
            rotating_bot = queries[0].metadata['rotating_bot']
            queries = [q for q in queries if q.metadata['variant'] == rotating_bot]
        # Structure handler already filters to observer only, no additional filtering needed

        all_queries.extend(queries)

    print(f"Total queries: {len(all_queries)}")

    if limit:
        all_queries = all_queries[:limit]
        print(f"Limiting to: {limit} queries")

    # Query VLM
    print(f"\n{'='*80}")
    print("QUERYING VLM")
    print(f"{'='*80}\n")

    results = []

    # Track which video pair we're currently evaluating
    # (needed because filtering may reduce queries per pair)
    current_video_id = -1
    last_episode_instance = None

    for i, query in enumerate(all_queries, 1):
        meta = query.metadata

        # Get query-specific prompt if handler supports it
        query_type = meta.get('query_type', 'default')
        if hasattr(handler, 'get_prompt') and 'query_type' in meta:
            prompt = handler.get_prompt(query_type)
        else:
            prompt = handler.get_prompt()

        query_type_display = f" ({query_type})" if query_type != 'default' else ""
        print(f"[{i}/{len(all_queries)}] Episode {meta['episode']}, Instance {meta['instance']}, {meta['variant'].upper()}{query_type_display}")
        print(f"  Expected: {query.expected_answer}", end="... ")

        try:
            # Determine which frames to extract based on handler type
            handler_name = handler.__class__.__name__
            is_turn_to_look = meta.get('is_turn_to_look', False)

            if is_turn_to_look:
                # Co-observation: extract frame2 from both alpha and bravo videos
                alpha_video = Path(meta['alpha_video'])
                bravo_video = Path(meta['bravo_video'])
                alpha_frame = meta['alpha_frame']
                bravo_frame = meta['bravo_frame']
                frame1_idx = meta['frame1']

                if generated_path and generated_subdir:
                    # Use generated videos
                    episode_instance = (meta['episode'], meta['instance'])
                    if episode_instance != last_episode_instance:
                        current_video_id += 1
                        last_episode_instance = episode_instance

                    video_id = current_video_id
                    generated_video = generated_subdir / f"video_{video_id}_side_by_side.mp4"

                    if not generated_video.exists():
                        raise FileNotFoundError(f"Generated video not found: {generated_video.name}")

                    # Extract frames from both alpha and bravo quadrants
                    image_bytes_alpha = extract_frame_from_generated(generated_video, alpha_frame, frame1_idx, "alpha")
                    image_bytes_bravo = extract_frame_from_generated(generated_video, bravo_frame, frame1_idx, "bravo")
                    vlm_response = query_vlm(prompt, image_bytes_alpha, image_bytes_bravo, enable_thinking=enable_thinking)
                else:
                    # Use ground-truth videos
                    image_bytes_alpha = extract_frame(alpha_video, alpha_frame)
                    image_bytes_bravo = extract_frame(bravo_video, bravo_frame)
                    vlm_response = query_vlm(prompt, image_bytes_alpha, image_bytes_bravo, enable_thinking=enable_thinking)

            elif generated_path and generated_subdir:
                # Use generated videos
                # Track which video pair we're on by watching for new (episode, instance) combinations
                episode_instance = (meta['episode'], meta['instance'])
                if episode_instance != last_episode_instance:
                    current_video_id += 1
                    last_episode_instance = episode_instance

                # The video pair index corresponds to the video_X in video_X_side_by_side.mp4
                video_id = current_video_id
                generated_video = generated_subdir / f"video_{video_id}_side_by_side.mp4"

                if not generated_video.exists():
                    raise FileNotFoundError(f"Generated video not found: {generated_video.name}")

                frame1_idx = meta['frame1']
                frame2_idx = meta['frame2']
                variant = meta['variant']

                # Check query type for looks_away/both_look_away handlers
                if query_type in ("player_position_during_turn", "player_position_turned_back",
                                  "player_invisible_looked_away"):
                    # All looks_away queries use only frame2
                    image_bytes = extract_frame_from_generated(generated_video, frame2_idx, frame1_idx, variant)
                    vlm_response = query_vlm(prompt, image_bytes, enable_thinking=enable_thinking)
                elif "LooksAway" in handler_name or "BothLookAway" in handler_name:
                    # looks_away and both_look_away (fallback for no query_type): both frames (to compare if they look the same)
                    # Use frame1_idx + 1 as first frame since generated video starts there
                    image_bytes_1 = extract_frame_from_generated(generated_video, frame1_idx + 1, frame1_idx, variant)
                    image_bytes_2 = extract_frame_from_generated(generated_video, frame2_idx, frame1_idx, variant)
                    vlm_response = query_vlm(prompt, image_bytes_1, image_bytes_2, enable_thinking=enable_thinking)
                elif "Rotation" in handler_name or "Structure" in handler_name:
                    # Rotation and structure: only frame2
                    image_bytes = extract_frame_from_generated(generated_video, frame2_idx, frame1_idx, variant)
                    vlm_response = query_vlm(prompt, image_bytes, enable_thinking=enable_thinking)
                else:
                    # Translation: both frames
                    # Use frame1_idx + 1 as first frame since generated video starts there
                    image_bytes_1 = extract_frame_from_generated(generated_video, frame1_idx + 1, frame1_idx, variant)
                    image_bytes_2 = extract_frame_from_generated(generated_video, frame2_idx, frame1_idx, variant)
                    vlm_response = query_vlm(prompt, image_bytes_1, image_bytes_2, enable_thinking=enable_thinking)
            else:
                # Use ground-truth videos
                # Check query type for looks_away/both_look_away handlers
                if query_type in ("player_position_during_turn", "player_position_turned_back",
                                  "player_invisible_looked_away"):
                    # All looks_away queries use only frame2
                    frame2_idx = meta['frame2']
                    image_bytes = extract_frame(query.video_path, frame2_idx)
                    vlm_response = query_vlm(prompt, image_bytes, enable_thinking=enable_thinking)
                elif "LooksAway" in handler_name or "BothLookAway" in handler_name:
                    # looks_away and both_look_away (fallback for no query_type): both frames (to compare if they look the same)
                    frame1_idx = meta['frame1']
                    frame2_idx = meta['frame2']
                    image_bytes_1 = extract_frame(query.video_path, frame1_idx)
                    image_bytes_2 = extract_frame(query.video_path, frame2_idx)
                    vlm_response = query_vlm(prompt, image_bytes_1, image_bytes_2, enable_thinking=enable_thinking)
                elif "Rotation" in handler_name or "Structure" in handler_name:
                    # Rotation and structure: only frame2
                    frame2_idx = meta['frame2']
                    image_bytes = extract_frame(query.video_path, frame2_idx)
                    vlm_response = query_vlm(prompt, image_bytes, enable_thinking=enable_thinking)
                else:
                    # Translation: both frames
                    frame1_idx = meta['frame1']
                    frame2_idx = meta['frame2']
                    image_bytes_1 = extract_frame(query.video_path, frame1_idx)
                    image_bytes_2 = extract_frame(query.video_path, frame2_idx)
                    vlm_response = query_vlm(prompt, image_bytes_1, image_bytes_2, enable_thinking=enable_thinking)

            # Check if response is "unclear"
            is_unclear = vlm_response.strip().lower() == "unclear"
            
            # Validate response (only meaningful if not unclear)
            is_correct = handler.validate_response(vlm_response, query.expected_answer) if not is_unclear else False

            result = EvalResult(
                query=query,
                vlm_response=vlm_response,
                is_correct=is_correct,
                is_unclear=is_unclear,
                metadata={
                    "prompt": prompt,
                    "handler": handler.__class__.__name__,
                    "using_generated": generated_path is not None,
                    **meta
                }
            )
            results.append(result)

            if is_unclear:
                print(f"Got: '{vlm_response}' UNCLEAR")
            else:
                status = "CORRECT" if is_correct else "WRONG"
                print(f"Got: '{vlm_response}' {status}")

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append(EvalResult(
                query=query,
                vlm_response=f"ERROR: {e}",
                is_correct=False,
                is_unclear=False,
                metadata={"error": str(e), **meta}
            ))

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    save_results(results, output_file, VLM_MODEL_NAME, model_name, thinking_enabled=enable_thinking)

    # Print summary
    total = len(results)
    unclear_count = sum(1 for r in results if r.is_unclear)
    evaluable = total - unclear_count
    correct = sum(1 for r in results if r.is_correct and not r.is_unclear)
    incorrect = evaluable - correct
    accuracy_excluding_unclear = correct / evaluable * 100 if evaluable > 0 else 0
    accuracy_total = correct / total * 100 if total > 0 else 0  # Treats unclear as incorrect
    unclear_percentage = unclear_count / total * 100 if total > 0 else 0

    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total queries: {total}")
    print(f"Unclear responses: {unclear_count} ({unclear_percentage:.2f}%)")
    print(f"Evaluable queries: {evaluable}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy (excluding unclear): {accuracy_excluding_unclear:.2f}%")
    print(f"Accuracy (total, unclear=wrong): {accuracy_total:.2f}%")

    # Break down by query type if multiple types exist
    query_types = set(r.metadata.get('query_type', 'default') for r in results)
    if len(query_types) > 1:
        print(f"\n{'='*80}")
        print("BREAKDOWN BY QUERY TYPE")
        print(f"{'='*80}")
        for qtype in sorted(query_types):
            type_results = [r for r in results if r.metadata.get('query_type', 'default') == qtype]
            type_total = len(type_results)
            type_unclear = sum(1 for r in type_results if r.is_unclear)
            type_evaluable = type_total - type_unclear
            type_correct = sum(1 for r in type_results if r.is_correct and not r.is_unclear)
            type_incorrect = type_evaluable - type_correct
            type_acc_excl_unclear = type_correct / type_evaluable * 100 if type_evaluable > 0 else 0
            type_acc_total = type_correct / type_total * 100 if type_total > 0 else 0

            print(f"\nQuery Type: {qtype}")
            print(f"  Total: {type_total}")
            print(f"  Unclear: {type_unclear}")
            print(f"  Evaluable: {type_evaluable}")
            print(f"  Correct: {type_correct}")
            print(f"  Incorrect: {type_incorrect}")
            print(f"  Accuracy (excluding unclear): {type_acc_excl_unclear:.2f}%")
            print(f"  Accuracy (total, unclear=wrong): {type_acc_total:.2f}%")

    print(f"{'='*80}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified evaluation script for mc_multiplayer datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (view keyframe info for first 10 episodes)
  python run_eval.py mc_multiplayer_v2_eval_max_speed/turnToLookEval --dry-run --limit 10

  # Extract frames for visual inspection
  python run_eval.py mc_multiplayer_v2_eval_max_speed/turnToLookOppositeEval --extract-frames --limit 5

  # Run full evaluation on ground-truth videos
  # (auto-saves to results_json/real/turnToLookEval.json)
  python run_eval.py mc_multiplayer_v2_eval_max_speed/turnToLookEval

  # Run evaluation on generated videos
  # (auto-saves to results_json/generated/{model_name}_turnToLookEval.json)
  python run_eval.py mc_multiplayer_v2_eval_max_speed/turnToLookEval --generated generations/flagship_final_v2_1B_multiplayer_final

  # Run evaluation for structure dataset
  python run_eval.py mc_multiplayer_eval_structure

  # Run evaluation with custom summary JSON
  python run_eval.py mc_multiplayer_eval_structure --summary-json path/to/summary.json
        """
    )

    parser.add_argument(
        "folder",
        help="Path to dataset folder (e.g., mc_multiplayer_v2_eval_max_speed/turnToLookEval). The /test subdirectory is automatically appended."
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="View keyframe info without extraction or evaluation"
    )
    mode_group.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames to frame_extraction/ folder"
    )

    # Common options
    parser.add_argument(
        "--output", "-o",
        default="eval_results.json",
        help="Output JSON file (auto-organized: results_json/generated/{model}_{dataset}.json or results_json/real/{dataset}.json)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of episodes/queries to process"
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key (optional, will use GEMINI_API_KEY env var if not provided)"
    )
    parser.add_argument(
        "--summary-json",
        help="Path to structure_building_summary.json (only needed for structure dataset)"
    )
    parser.add_argument(
        "--generated",
        help="Path to generated videos directory (e.g., generations/flagship_final_v2_1B_multiplayer_final)"
    )

    args = parser.parse_args()

    # Set API key if provided
    if args.api_key:
        os.environ['GEMINI_API_KEY'] = args.api_key

    # Get folder path - automatically append /test if not already present
    folder = Path(args.folder)
    if folder.name != "test":
        dataset_name = folder.name
        folder = folder / "test"
    else:
        dataset_name = folder.parent.name

    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return 1

    try:
        handler = identify_handler(dataset_name, summary_json_path=args.summary_json)
        print(f"Dataset: {dataset_name}")
        print(f"Handler: {handler.__class__.__name__}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Find video pairs
    video_pairs = find_mc_video_pairs(folder)
    print(f"Found: {len(video_pairs)} video pairs")

    if not video_pairs:
        print("Error: No video pairs found")
        return 1

    # Apply default limit of 32 video pairs if no limit specified
    DEFAULT_VIDEO_PAIR_LIMIT = 32
    if len(video_pairs) > DEFAULT_VIDEO_PAIR_LIMIT and args.limit is None:
        print(f"Limiting to first {DEFAULT_VIDEO_PAIR_LIMIT} video pairs (use --limit to override)")
        video_pairs = video_pairs[:DEFAULT_VIDEO_PAIR_LIMIT]

    # Parse generated path if provided
    generated_path = None
    model_name = "ground_truth"
    if args.generated:
        generated_path = Path(args.generated)
        if not generated_path.exists():
            print(f"Error: Generated videos path not found: {generated_path}")
            return 1
        # Extract model name from path (e.g., "flagship_final_v2_1B_multiplayer_final")
        model_name = generated_path.name

    # Determine output file path for evaluation
    output_file = args.output
    if not args.dry_run and not args.extract_frames:
        # Only auto-generate output path if using the default value
        if args.output == "eval_results.json":
            if generated_path:
                # Save to results_json/generated/{model_name}_{dataset_name}.json
                output_dir = Path("results_json/generated")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = str(output_dir / f"{model_name}_{dataset_name}.json")
            else:
                # Save to results_json/real/{dataset_name}.json
                output_dir = Path("results_json/real")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = str(output_dir / f"{dataset_name}.json")
        print(f"Output file: {output_file}")

    # Execute based on mode
    if args.dry_run:
        dry_run(handler, video_pairs, limit=args.limit)

    elif args.extract_frames:
        # Determine output directory based on dataset
        if generated_path:
            output_dir = f"frame_extraction/{folder.parent.name}_generated"
        else:
            output_dir = f"frame_extraction/{folder.parent.name}"
        extract_frames_mode(handler, video_pairs, output_dir, limit=args.limit,
                          generated_path=generated_path, dataset_name=dataset_name)

    else:
        # Run full evaluation
        run_evaluation(handler, video_pairs, output_file, limit=args.limit,
                      generated_path=generated_path, dataset_name=dataset_name,
                      model_name=model_name)

    return 0


if __name__ == "__main__":
    exit(main())
