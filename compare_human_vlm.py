#!/usr/bin/env python3
"""
Compare human blind evaluation answers against VLM answers.

Usage:
    python compare_human_vlm.py blind_eval/human_answers_2026-01-26.json
    python compare_human_vlm.py blind_eval/human_answers_2026-01-26.json --trial 2
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare human vs VLM answers")
    parser.add_argument("human_answers", type=Path, help="Path to human answers JSON")
    parser.add_argument("--trial", type=int, default=1, help="VLM trial to compare against (default: 1)")
    parser.add_argument("--results-dir", type=Path, default=Path("results_json/generated"),
                        help="Path to VLM results directory")
    args = parser.parse_args()

    # Load human answers
    with open(args.human_answers) as f:
        human_data = json.load(f)

    # Ground truth expected answers
    expected = {
        "turnToLookEval": "yes",
        "turnToLookOppositeEval": "no"
    }

    # Load VLM results
    results_base = args.results_dir
    vlm_cache = {}

    def get_vlm_answer(model, eval_type, episode, instance):
        cache_key = f"{model}_{eval_type}"
        if cache_key not in vlm_cache:
            trial_path = results_base / cache_key / f"trial_{args.trial}.json"
            if trial_path.exists():
                with open(trial_path) as f:
                    vlm_cache[cache_key] = json.load(f)
            else:
                return None
        
        vlm_data = vlm_cache[cache_key]
        for r in vlm_data["results"]:
            if r["metadata"]["episode"] == episode and r["metadata"]["instance"] == instance:
                return r["response"]
        return None

    # Compare human vs VLM
    comparisons = {}
    for answer in human_data["answers"]:
        model = answer["model"]
        eval_type = answer["eval_type"]
        episode = answer["episode"]
        instance = answer["instance"]
        human_answer = answer["human_answer"]
        
        vlm_answer = get_vlm_answer(model, eval_type, episode, instance)
        if vlm_answer is None:
            continue
        
        expected_answer = expected[eval_type]
        human_correct = human_answer.lower() == expected_answer.lower()
        vlm_correct = vlm_answer.lower() == expected_answer.lower()
        agree = human_answer.lower() == vlm_answer.lower()
        
        key = (model, eval_type)
        if key not in comparisons:
            comparisons[key] = {
                "total": 0,
                "human_correct": 0,
                "vlm_correct": 0,
                "agree": 0,
                "both_correct": 0,
                "both_wrong": 0,
                "human_right_vlm_wrong": 0,
                "vlm_right_human_wrong": 0,
            }
        
        c = comparisons[key]
        c["total"] += 1
        if human_correct:
            c["human_correct"] += 1
        if vlm_correct:
            c["vlm_correct"] += 1
        if agree:
            c["agree"] += 1
        if human_correct and vlm_correct:
            c["both_correct"] += 1
        elif not human_correct and not vlm_correct:
            c["both_wrong"] += 1
        elif human_correct and not vlm_correct:
            c["human_right_vlm_wrong"] += 1
        else:
            c["vlm_right_human_wrong"] += 1

    print("=" * 80)
    print(f"HUMAN vs VLM COMPARISON (Trial {args.trial})")
    print("=" * 80)

    # Overall
    total = sum(c["total"] for c in comparisons.values())
    human_correct = sum(c["human_correct"] for c in comparisons.values())
    vlm_correct = sum(c["vlm_correct"] for c in comparisons.values())
    agree = sum(c["agree"] for c in comparisons.values())

    print(f"\nOVERALL ({total} samples):")
    print(f"  Human-Eval:       {human_correct}/{total} ({100*human_correct/total:.1f}%)")
    print(f"  VLM-Eval:         {vlm_correct}/{total} ({100*vlm_correct/total:.1f}%)")
    print(f"  Agreement Rate:   {agree}/{total} ({100*agree/total:.1f}%)")

    # By model variant
    print("\n" + "-" * 80)
    print("BY MODEL VARIANT:")
    print("-" * 80)
    models = sorted(set(k[0] for k in comparisons.keys()))
    for model in models:
        m_total = sum(c["total"] for k, c in comparisons.items() if k[0] == model)
        m_human = sum(c["human_correct"] for k, c in comparisons.items() if k[0] == model)
        m_vlm = sum(c["vlm_correct"] for k, c in comparisons.items() if k[0] == model)
        m_agree = sum(c["agree"] for k, c in comparisons.items() if k[0] == model)
        print(f"\n  {model} ({m_total} samples):")
        print(f"    Human-Eval:      {m_human}/{m_total} ({100*m_human/m_total:.1f}%)")
        print(f"    VLM-Eval:        {m_vlm}/{m_total} ({100*m_vlm/m_total:.1f}%)")
        print(f"    Agreement Rate:  {m_agree}/{m_total} ({100*m_agree/m_total:.1f}%)")

    # By eval type
    print("\n" + "-" * 80)
    print("BY EVAL TYPE:")
    print("-" * 80)
    for eval_type in ["turnToLookEval", "turnToLookOppositeEval"]:
        e_total = sum(c["total"] for k, c in comparisons.items() if k[1] == eval_type)
        if e_total == 0:
            continue
        e_human = sum(c["human_correct"] for k, c in comparisons.items() if k[1] == eval_type)
        e_vlm = sum(c["vlm_correct"] for k, c in comparisons.items() if k[1] == eval_type)
        e_agree = sum(c["agree"] for k, c in comparisons.items() if k[1] == eval_type)
        exp = expected[eval_type]
        print(f"\n  {eval_type} (expected: {exp}, {e_total} samples):")
        print(f"    Human-Eval:      {e_human}/{e_total} ({100*e_human/e_total:.1f}%)")
        print(f"    VLM-Eval:        {e_vlm}/{e_total} ({100*e_vlm/e_total:.1f}%)")
        print(f"    Agreement Rate:  {e_agree}/{e_total} ({100*e_agree/e_total:.1f}%)")

    # By model variant AND eval type (detailed breakdown)
    print("\n" + "-" * 80)
    print("BY MODEL VARIANT + EVAL TYPE:")
    print("-" * 80)
    for model in models:
        print(f"\n  {model}:")
        for eval_type in ["turnToLookEval", "turnToLookOppositeEval"]:
            key = (model, eval_type)
            if key in comparisons:
                c = comparisons[key]
                exp = expected[eval_type]
                print(f"\n    {eval_type} (expected: {exp}):")
                print(f"      Human-Eval:      {c['human_correct']}/{c['total']} ({100*c['human_correct']/c['total']:.1f}%)")
                print(f"      VLM-Eval:        {c['vlm_correct']}/{c['total']} ({100*c['vlm_correct']/c['total']:.1f}%)")
                print(f"      Agreement Rate:  {c['agree']}/{c['total']} ({100*c['agree']/c['total']:.1f}%)")
                print(f"      Both correct:    {c['both_correct']}/{c['total']}")
                print(f"      Both wrong:      {c['both_wrong']}/{c['total']}")
                print(f"      Human✓ VLM✗:     {c['human_right_vlm_wrong']}/{c['total']}")
                print(f"      VLM✓ Human✗:     {c['vlm_right_human_wrong']}/{c['total']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
