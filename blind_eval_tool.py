#!/usr/bin/env python3
"""
Blind evaluation tool for turn-to-look evaluations.

Creates a shuffled set of cropped images (generated perspectives only) and an HTML
interface for manual evaluation. The human evaluator answers yes/no without seeing
the expected answer or GT frames.

Usage:
    python blind_eval_tool.py --prepare --models flagship no_kv_cache_backprop
    python blind_eval_tool.py --verify human_answers.json
"""

import argparse
import json
import random
import base64
from pathlib import Path
from typing import List, Dict, Any
import re


def crop_to_generated_only(image_path: Path, output_path: Path) -> bool:
    """
    Crop a side-by-side comparison image to show only the generated perspectives.
    
    The original image has:
    - Left half: GT (Alpha and Bravo stacked)
    - Right half: Generated (Alpha and Bravo stacked)
    - Bottom: Expected answer text bar
    
    We want to extract only the right half without the bottom bar.
    """
    import cv2
    import numpy as np
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  Warning: Could not read {image_path}")
        return False
    
    h, w = img.shape[:2]
    
    # The bottom bar is approximately 40-60 pixels (depends on text length)
    # We'll detect it by looking for the dark gray bar (RGB ~40,40,40)
    # Scan from bottom to find where the gray bar starts
    bottom_bar_height = 0
    for y in range(h - 1, max(0, h - 100), -1):
        row = img[y, :]
        # Check if row is mostly dark gray (the expected answer bar)
        mean_color = row.mean(axis=0)
        if np.all(mean_color < 60):  # Dark gray threshold
            bottom_bar_height = h - y
        else:
            break
    
    # If we didn't find a bar, assume a default
    if bottom_bar_height < 20:
        bottom_bar_height = 50  # Default estimate
    
    # Crop: right half, excluding bottom bar
    # The image is split horizontally: left half is GT, right half is Generated
    content_height = h - bottom_bar_height
    half_width = w // 2
    
    # Extract right half (generated)
    cropped = img[0:content_height, half_width:w]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cropped)
    return True


def prepare_blind_evaluation(
    models: List[str],
    eval_types: List[str] = ["turnToLookEval", "turnToLookOppositeEval"],
    samples_per_eval: int = 32,
    output_dir: Path = Path("blind_eval"),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Prepare the blind evaluation dataset.
    
    1. Find side-by-side images for each model and eval type
    2. Crop to generated-only
    3. Create a shuffled question list
    4. Generate HTML interface
    """
    random.seed(seed)
    
    sbs_base = Path("frame_extraction_side_by_side")
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir = output_dir / "cropped_images"
    cropped_dir.mkdir(exist_ok=True)
    
    questions = []
    question_id = 0
    
    for model in models:
        for eval_type in eval_types:
            sbs_dir = sbs_base / eval_type / model / "default"
            if not sbs_dir.exists():
                print(f"Warning: {sbs_dir} does not exist, skipping")
                continue
            
            # Find all comparison images
            images = sorted(sbs_dir.glob("*_comparison.png"))
            print(f"Found {len(images)} images in {sbs_dir}")
            
            # Limit to samples_per_eval
            images = images[:samples_per_eval]
            
            for img_path in images:
                # Parse filename: ep000000_inst000_turn_to_look_comparison.png
                match = re.match(r"ep(\d+)_inst(\d+)_(.+)_comparison\.png", img_path.name)
                if not match:
                    print(f"  Warning: Could not parse {img_path.name}")
                    continue
                
                episode, instance, variant = match.groups()
                
                # Crop image
                cropped_filename = f"q{question_id:04d}.png"
                cropped_path = cropped_dir / cropped_filename
                
                if crop_to_generated_only(img_path, cropped_path):
                    questions.append({
                        "id": question_id,
                        "cropped_image": cropped_filename,
                        "original_image": str(img_path),
                        "model": model,
                        "eval_type": eval_type,
                        "episode": episode,
                        "instance": instance,
                        "variant": variant,
                    })
                    question_id += 1
    
    print(f"\nTotal questions prepared: {len(questions)}")
    
    # Shuffle questions
    random.shuffle(questions)
    
    # Save question mapping (for later verification)
    mapping_path = output_dir / "question_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({
            "seed": seed,
            "models": models,
            "eval_types": eval_types,
            "questions": questions,
        }, f, indent=2)
    print(f"Saved question mapping to {mapping_path}")
    
    # Generate HTML interface
    generate_html_interface(questions, cropped_dir, output_dir)
    
    return {"questions": questions, "mapping_path": mapping_path}


def generate_html_interface(questions: List[Dict], cropped_dir: Path, output_dir: Path):
    """Generate an HTML interface for the blind evaluation."""
    
    # Embed images as base64 for self-contained HTML
    images_base64 = {}
    for q in questions:
        img_path = cropped_dir / q["cropped_image"]
        if img_path.exists():
            with open(img_path, "rb") as f:
                images_base64[q["cropped_image"]] = base64.b64encode(f.read()).decode()
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Blind Evaluation - Turn to Look</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .progress {{
            font-size: 18px;
            margin-bottom: 20px;
        }}
        .question-container {{
            text-align: center;
        }}
        .image-container {{
            margin: 20px 0;
            background: #2a2a2a;
            padding: 10px;
            border-radius: 8px;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 70vh;
        }}
        .prompt {{
            font-size: 18px;
            margin: 20px 0;
            padding: 15px;
            background: #333;
            border-radius: 8px;
        }}
        .buttons {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 30px 0;
        }}
        .btn {{
            padding: 15px 60px;
            font-size: 24px;
            cursor: pointer;
            border: none;
            border-radius: 8px;
            transition: transform 0.1s;
        }}
        .btn:hover {{
            transform: scale(1.05);
        }}
        .btn-yes {{
            background: #4CAF50;
            color: white;
        }}
        .btn-no {{
            background: #f44336;
            color: white;
        }}
        .btn-skip {{
            background: #666;
            color: white;
            padding: 10px 30px;
            font-size: 16px;
        }}
        .completed {{
            text-align: center;
            padding: 50px;
        }}
        .download-btn {{
            padding: 20px 40px;
            font-size: 20px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin: 20px;
        }}
        .nav-buttons {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }}
        .nav-btn {{
            padding: 8px 20px;
            font-size: 14px;
            background: #555;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .answer-indicator {{
            margin-top: 10px;
            font-size: 14px;
            color: #888;
        }}
        .answered {{
            color: #4CAF50;
        }}
        .keyboard-hint {{
            margin-top: 15px;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Blind Evaluation: Turn to Look</h1>
        <p>Do these two Minecraft screenshots show the same scenery?</p>
    </div>
    
    <div id="evaluation" class="question-container">
        <div class="progress">
            Question <span id="current">1</span> of <span id="total">{len(questions)}</span>
            (<span id="answered-count">0</span> answered)
        </div>
        
        <div class="image-container">
            <img id="question-image" src="" alt="Question image">
        </div>
        
        <div class="prompt">
            Do these two screenshots show the same scenery?
        </div>
        
        <div class="buttons">
            <button class="btn btn-yes" onclick="answer('yes')">Yes (Y)</button>
            <button class="btn btn-no" onclick="answer('no')">No (N)</button>
        </div>
        
        <div class="nav-buttons">
            <button class="nav-btn" onclick="prevQuestion()">&larr; Previous (A)</button>
            <button class="btn-skip nav-btn" onclick="skipQuestion()">Skip (S)</button>
            <button class="nav-btn" onclick="nextQuestion()">Next (D) &rarr;</button>
        </div>
        
        <div id="answer-indicator" class="answer-indicator"></div>
        <div class="keyboard-hint">Keyboard: Y=Yes, N=No, S=Skip, A=Prev, D=Next</div>
    </div>
    
    <div id="completed" class="completed" style="display: none;">
        <h2>Evaluation Complete!</h2>
        <p>You answered <span id="final-count">0</span> out of {len(questions)} questions.</p>
        <button class="download-btn" onclick="downloadResults()">Download Results</button>
        <button class="nav-btn" onclick="goBack()">Go Back to Review</button>
    </div>

    <script>
        const questions = {json.dumps(questions)};
        const images = {json.dumps(images_base64)};
        
        let currentIndex = 0;
        let answers = {{}};
        
        // Load saved progress from localStorage
        const savedAnswers = localStorage.getItem('blindEvalAnswers');
        if (savedAnswers) {{
            try {{
                answers = JSON.parse(savedAnswers);
                console.log('Loaded saved progress:', Object.keys(answers).length, 'answers');
            }} catch (e) {{
                console.error('Failed to load saved progress');
            }}
        }}
        
        function saveProgress() {{
            localStorage.setItem('blindEvalAnswers', JSON.stringify(answers));
        }}
        
        function updateDisplay() {{
            const q = questions[currentIndex];
            document.getElementById('current').textContent = currentIndex + 1;
            document.getElementById('total').textContent = questions.length;
            document.getElementById('answered-count').textContent = Object.keys(answers).length;
            
            const imgSrc = images[q.cropped_image];
            document.getElementById('question-image').src = 'data:image/png;base64,' + imgSrc;
            
            const indicator = document.getElementById('answer-indicator');
            if (answers[q.id] !== undefined) {{
                indicator.textContent = 'Your answer: ' + answers[q.id].toUpperCase();
                indicator.className = 'answer-indicator answered';
            }} else {{
                indicator.textContent = 'Not answered yet';
                indicator.className = 'answer-indicator';
            }}
            
            // Check if all answered
            if (Object.keys(answers).length === questions.length) {{
                document.getElementById('evaluation').style.display = 'none';
                document.getElementById('completed').style.display = 'block';
                document.getElementById('final-count').textContent = Object.keys(answers).length;
            }}
        }}
        
        function answer(response) {{
            const q = questions[currentIndex];
            answers[q.id] = response;
            saveProgress();
            
            if (currentIndex < questions.length - 1) {{
                currentIndex++;
                updateDisplay();
            }} else {{
                updateDisplay();
            }}
        }}
        
        function skipQuestion() {{
            if (currentIndex < questions.length - 1) {{
                currentIndex++;
                updateDisplay();
            }}
        }}
        
        function prevQuestion() {{
            if (currentIndex > 0) {{
                currentIndex--;
                updateDisplay();
            }}
        }}
        
        function nextQuestion() {{
            if (currentIndex < questions.length - 1) {{
                currentIndex++;
                updateDisplay();
            }}
        }}
        
        function goBack() {{
            document.getElementById('evaluation').style.display = 'block';
            document.getElementById('completed').style.display = 'none';
            currentIndex = 0;
            updateDisplay();
        }}
        
        function downloadResults() {{
            const timestamp = new Date().toISOString();
            const results = {{
                timestamp: timestamp,
                description: "Human blind evaluation answers for turn-to-look tasks. Can be re-verified against updated VLM results using: python blind_eval_tool.py verify <this_file>",
                total_questions: questions.length,
                answered: Object.keys(answers).length,
                models_evaluated: [...new Set(questions.map(q => q.model))],
                eval_types: [...new Set(questions.map(q => q.eval_type))],
                answers: questions.map(q => ({{
                    question_id: q.id,
                    model: q.model,
                    eval_type: q.eval_type,
                    episode: q.episode,
                    instance: q.instance,
                    variant: q.variant,
                    original_image: q.original_image,
                    human_answer: answers[q.id] || null
                }}))
            }};
            
            const blob = new Blob([JSON.stringify(results, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            // Include timestamp in filename for versioning
            const dateStr = timestamp.slice(0, 10);  // YYYY-MM-DD
            a.href = url;
            a.download = `human_answers_${{dateStr}}.json`;
            a.click();
            URL.revokeObjectURL(url);
            
            // Also save to localStorage with timestamp key for backup
            localStorage.setItem('blindEvalAnswers_' + timestamp, JSON.stringify(results));
            console.log('Answers backed up to localStorage with key: blindEvalAnswers_' + timestamp);
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            const key = e.key.toLowerCase();
            if (key === 'y') answer('yes');
            else if (key === 'n') answer('no');
            else if (key === 's') skipQuestion();
            else if (key === 'a' || key === 'arrowleft') prevQuestion();
            else if (key === 'd' || key === 'arrowright') nextQuestion();
        }});
        
        // Initialize
        updateDisplay();
    </script>
</body>
</html>
'''
    
    html_path = output_dir / "blind_eval.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Generated HTML interface: {html_path}")
    print(f"\nTo start the evaluation, open {html_path} in a browser.")
    print("In VSCode, you can right-click the file and select 'Open with Live Server'")
    print("or use the 'Simple Browser' extension.")


def verify_answers(human_answers_path: Path, question_mapping_path: Path = None, trial: int = 1, results_dir: Path = None):
    """
    Compare human answers against VLM results.
    
    Args:
        human_answers_path: Path to human_answers.json
        question_mapping_path: Path to question_mapping.json (optional if answers contain full info)
        trial: Which VLM trial to compare against (default: 1)
        results_dir: Custom results directory (default: results_json/generated)
    """
    results_base = results_dir or Path("results_json/generated")
    
    # Load human answers
    with open(human_answers_path) as f:
        human_data = json.load(f)
    
    print(f"Loaded human answers from: {human_answers_path}")
    print(f"  Timestamp: {human_data.get('timestamp', 'unknown')}")
    print(f"  Total questions: {human_data.get('total_questions', len(human_data.get('answers', [])))}")
    print(f"  Answered: {human_data.get('answered', 'unknown')}")
    print(f"  Models: {human_data.get('models_evaluated', 'unknown')}")
    print(f"  Eval types: {human_data.get('eval_types', 'unknown')}")
    print(f"\nComparing against VLM trial_{trial}.json files...")
    
    # The new format has all info in answers, so question_mapping is optional
    # Build lookup from question_id to question info (from answers themselves)
    question_lookup = {a["question_id"]: a for a in human_data["answers"]}
    
    # Process each answer
    comparisons = []
    vlm_cache = {}  # Cache loaded VLM results
    
    for answer_entry in human_data["answers"]:
        if answer_entry["human_answer"] is None:
            continue
        
        model = answer_entry["model"]
        eval_type = answer_entry["eval_type"]
        episode = answer_entry["episode"]
        instance = answer_entry["instance"]
        human_answer = answer_entry["human_answer"]
        q_id = answer_entry["question_id"]
        
        # Load VLM results for this model/eval (with caching)
        cache_key = f"{model}_{eval_type}"
        vlm_results_dir = results_base / cache_key
        vlm_trial_path = vlm_results_dir / f"trial_{trial}.json"
        
        if not vlm_trial_path.exists():
            print(f"Warning: VLM results not found at {vlm_trial_path}")
            continue
        
        # Use cached VLM data if available
        if cache_key not in vlm_cache:
            with open(vlm_trial_path) as f:
                vlm_cache[cache_key] = json.load(f)
        vlm_data = vlm_cache[cache_key]
        
        # Find matching VLM result
        vlm_answer = None
        expected_answer = None
        for r in vlm_data["results"]:
            if r["metadata"]["episode"] == episode and r["metadata"]["instance"] == instance:
                vlm_answer = r["response"]
                expected_answer = r["expected"]
                break
        
        if vlm_answer is None:
            print(f"Warning: Could not find VLM result for ep{episode}_inst{instance}")
            continue
        
        comparisons.append({
            "question_id": q_id,
            "model": model,
            "eval_type": eval_type,
            "episode": episode,
            "instance": instance,
            "expected": expected_answer,
            "human": human_answer,
            "vlm": vlm_answer,
            "human_correct": human_answer.lower() == expected_answer.lower(),
            "vlm_correct": vlm_answer.lower() == expected_answer.lower(),
            "human_vlm_agree": human_answer.lower() == vlm_answer.lower(),
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    total = len(comparisons)
    human_correct = sum(1 for c in comparisons if c["human_correct"])
    vlm_correct = sum(1 for c in comparisons if c["vlm_correct"])
    agreement = sum(1 for c in comparisons if c["human_vlm_agree"])
    
    print(f"\nTotal questions compared: {total}")
    print(f"\nHuman accuracy:     {human_correct}/{total} ({100*human_correct/total:.1f}%)")
    print(f"VLM accuracy:       {vlm_correct}/{total} ({100*vlm_correct/total:.1f}%)")
    print(f"Human-VLM agreement: {agreement}/{total} ({100*agreement/total:.1f}%)")
    
    # Breakdown by model
    models = set(c["model"] for c in comparisons)
    for model in sorted(models):
        model_comps = [c for c in comparisons if c["model"] == model]
        m_total = len(model_comps)
        m_human = sum(1 for c in model_comps if c["human_correct"])
        m_vlm = sum(1 for c in model_comps if c["vlm_correct"])
        m_agree = sum(1 for c in model_comps if c["human_vlm_agree"])
        
        print(f"\n--- Model: {model} ---")
        print(f"  Human accuracy:     {m_human}/{m_total} ({100*m_human/m_total:.1f}%)")
        print(f"  VLM accuracy:       {m_vlm}/{m_total} ({100*m_vlm/m_total:.1f}%)")
        print(f"  Agreement:          {m_agree}/{m_total} ({100*m_agree/m_total:.1f}%)")
    
    # Breakdown by eval type
    eval_types = set(c["eval_type"] for c in comparisons)
    for eval_type in sorted(eval_types):
        eval_comps = [c for c in comparisons if c["eval_type"] == eval_type]
        e_total = len(eval_comps)
        e_human = sum(1 for c in eval_comps if c["human_correct"])
        e_vlm = sum(1 for c in eval_comps if c["vlm_correct"])
        e_agree = sum(1 for c in eval_comps if c["human_vlm_agree"])
        
        expected = eval_comps[0]["expected"] if eval_comps else "?"
        print(f"\n--- Eval Type: {eval_type} (expected: {expected}) ---")
        print(f"  Human accuracy:     {e_human}/{e_total} ({100*e_human/e_total:.1f}%)")
        print(f"  VLM accuracy:       {e_vlm}/{e_total} ({100*e_vlm/e_total:.1f}%)")
        print(f"  Agreement:          {e_agree}/{e_total} ({100*e_agree/e_total:.1f}%)")
    
    # Show disagreements
    disagreements = [c for c in comparisons if not c["human_vlm_agree"]]
    if disagreements:
        print(f"\n{'=' * 80}")
        print(f"DISAGREEMENTS ({len(disagreements)} cases)")
        print("=" * 80)
        for d in disagreements[:20]:  # Show first 20
            status = ""
            if d["human_correct"] and not d["vlm_correct"]:
                status = "[Human RIGHT, VLM wrong]"
            elif not d["human_correct"] and d["vlm_correct"]:
                status = "[Human wrong, VLM RIGHT]"
            else:
                status = "[Both wrong]"
            print(f"  {d['model']}/{d['eval_type']} ep{d['episode']}_inst{d['instance']}: "
                  f"Human={d['human']}, VLM={d['vlm']}, Expected={d['expected']} {status}")
    
    # Save detailed results
    output_path = human_answers_path.parent / f"verification_results_trial{trial}.json"
    with open(output_path, "w") as f:
        json.dump({
            "human_answers_file": str(human_answers_path),
            "vlm_trial": trial,
            "results_dir": str(results_base),
            "summary": {
                "total": total,
                "human_correct": human_correct,
                "human_accuracy": human_correct / total if total > 0 else 0,
                "vlm_correct": vlm_correct,
                "vlm_accuracy": vlm_correct / total if total > 0 else 0,
                "agreement": agreement,
                "agreement_rate": agreement / total if total > 0 else 0,
            },
            "comparisons": comparisons,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")
    print(f"\nTo re-verify against a different VLM trial, run:")
    print(f"  python blind_eval_tool.py verify {human_answers_path} --trial 2")


def main():
    parser = argparse.ArgumentParser(description="Blind evaluation tool for turn-to-look evaluations")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare blind evaluation dataset")
    prepare_parser.add_argument(
        "--models", 
        nargs="+", 
        default=["flagship", "no_kv_cache_backprop"],
        help="Models to evaluate"
    )
    prepare_parser.add_argument(
        "--samples-per-eval",
        type=int,
        default=32,
        help="Number of samples per evaluation type per model"
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("blind_eval"),
        help="Output directory"
    )
    prepare_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify human answers against VLM")
    verify_parser.add_argument(
        "human_answers",
        type=Path,
        help="Path to human_answers.json"
    )
    verify_parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Path to question_mapping.json (optional, answers file contains full info)"
    )
    verify_parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="Which VLM trial to compare against (default: 1)"
    )
    verify_parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Custom results directory (default: results_json/generated)"
    )
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare_blind_evaluation(
            models=args.models,
            samples_per_eval=args.samples_per_eval,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.command == "verify":
        verify_answers(args.human_answers, args.mapping, args.trial, args.results_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
