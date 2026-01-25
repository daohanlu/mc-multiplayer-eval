#!/usr/bin/env python3
"""
Extract frames for all models and tasks, creating side-by-side comparisons.

This script runs frame extraction for all model variants in mc_multiplayer_v2_generations
against all tasks in mc_multiplayer_v2_eval_new_sneak_combined.

Usage:
    python extract_all_frames.py [--limit N] [--models-dir DIR] [--dataset-dir DIR]
    
Examples:
    # Extract frames with default limit of 6 episodes
    python extract_all_frames.py
    
    # Extract frames for 10 episodes
    python extract_all_frames.py --limit 10
    
    # Use custom directories
    python extract_all_frames.py --models-dir my_generations --dataset-dir my_dataset
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames for all models and tasks with side-by-side comparisons"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Number of episodes to extract per task (default: 6)"
    )
    parser.add_argument(
        "--models-dir",
        default="mc_multiplayer_v2_generations",
        help="Directory containing model generation folders (default: mc_multiplayer_v2_generations)"
    )
    parser.add_argument(
        "--dataset-dir",
        default="mc_multiplayer_v2_eval_new_sneak_combined",
        help="Directory containing evaluation task folders (default: mc_multiplayer_v2_eval_new_sneak_combined)"
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    dataset_dir = Path(args.dataset_dir)
    
    # Validate directories exist
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return 1
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    # Find all model folders (directories in models_dir)
    model_folders = sorted([d for d in models_dir.iterdir() if d.is_dir()])
    if not model_folders:
        print(f"Error: No model folders found in {models_dir}")
        return 1
    
    # Find all task folders (directories in dataset_dir that contain a 'test' subfolder)
    task_folders = sorted([
        d for d in dataset_dir.iterdir() 
        if d.is_dir() and (d / "test").exists()
    ])
    if not task_folders:
        print(f"Error: No task folders found in {dataset_dir}")
        return 1
    
    print(f"{'='*80}")
    print("FRAME EXTRACTION FOR ALL MODELS AND TASKS")
    print(f"{'='*80}")
    print(f"Models directory: {models_dir}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Limit: {args.limit} episodes per task")
    print(f"\nFound {len(model_folders)} model(s):")
    for m in model_folders:
        print(f"  - {m.name}")
    print(f"\nFound {len(task_folders)} task(s):")
    for t in task_folders:
        print(f"  - {t.name}")
    print(f"{'='*80}\n")
    
    # Also extract GT (real) frames for each task
    print("Step 1: Extracting ground-truth frames...")
    print("-" * 40)
    for task_folder in task_folders:
        task_name = task_folder.name
        print(f"\n[GT] Task: {task_name}")
        
        cmd = [
            sys.executable, "run_eval.py",
            str(task_folder),
            "--extract-frames",
            "--limit", str(args.limit),
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  Warning: GT extraction failed for {task_name}")
    
    # Run extraction for each model-task combination
    print(f"\n{'='*80}")
    print("Step 2: Extracting generated frames with side-by-side comparisons...")
    print("-" * 40)
    
    total_combinations = len(model_folders) * len(task_folders)
    current = 0
    
    for model_folder in model_folders:
        model_name = model_folder.name
        
        for task_folder in task_folders:
            current += 1
            task_name = task_folder.name
            
            print(f"\n[{current}/{total_combinations}] Model: {model_name}, Task: {task_name}")
            
            cmd = [
                sys.executable, "run_eval.py",
                str(task_folder),
                "--extract-frames",
                "--limit", str(args.limit),
                "--generated", str(model_folder),
            ]
            
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"  Warning: Extraction failed for {model_name}/{task_name}")
    
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nFrames saved to:")
    print(f"  - frame_extraction/          (individual frames)")
    print(f"  - frame_extraction_side_by_side/  (GT vs Generated comparisons)")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
