#!/usr/bin/env python3
"""
Compare a simple human yes/no responses file against VLM trial results.

Human responses file format:
  First line: model,dataset,query_type
  Subsequent lines: y / n (one per line)

VLM results directory format:
  <vlm_results_dir>/<model>_<dataset>/trial_1.json
  (also supports a couple fallback folder layouts)

The script filters trial["results"] by query_type (using r["metadata"]["query_type"]),
aligns with human answers in order, and counts matches on the "response" field.

Usage:
  python compare_human_responses_yesno.py /path/to/generated /path/to/human_responses.txt
  python compare_human_responses_yesno.py /path/to/generated /path/to/human_responses.txt --trial 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _normalize_yesno(value: str) -> str:
    v = value.strip().lower()
    if v in {"y", "yes", "true", "1"}:
        return "yes"
    if v in {"n", "no", "false", "0"}:
        return "no"
    raise ValueError(f"Unrecognized yes/no value: {value!r}")


def _read_human_file(path: Path) -> Tuple[str, str, str, List[str]]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    lines = [ln for ln in lines if ln != ""]
    if not lines:
        raise ValueError(f"Human responses file is empty: {path}")

    header = lines[0].split(",")
    if len(header) != 3:
        raise ValueError(
            f"Expected first line to be 'model,dataset,query_type' (3 comma-separated fields), got: {lines[0]!r}"
        )
    model, dataset, query_type = (h.strip() for h in header)
    if not model or not dataset or not query_type:
        raise ValueError(f"Header fields must be non-empty, got: {lines[0]!r}")

    answers: List[str] = []
    for i, ln in enumerate(lines[1:], start=2):
        try:
            answers.append(_normalize_yesno(ln))
        except ValueError as e:
            raise ValueError(f"{path}:{i}: {e}") from e

    return model, dataset, query_type, answers


def _candidate_trial_paths(results_dir: Path, model: str, dataset: str, trial: int) -> Iterable[Path]:
    # Most common: <base>/<model>_<dataset>/trial_1.json
    yield results_dir / f"{model}_{dataset}" / f"trial_{trial}.json"
    # Sometimes concatenated without underscore.
    yield results_dir / f"{model}{dataset}" / f"trial_{trial}.json"
    # Sometimes nested as <base>/<model>/<dataset>/trial_1.json
    yield results_dir / model / dataset / f"trial_{trial}.json"


def _find_trial_json(results_dir: Path, model: str, dataset: str, trial: int) -> Path:
    for p in _candidate_trial_paths(results_dir, model, dataset, trial):
        if p.exists():
            return p
    tried = "\n".join(f"  - {p}" for p in _candidate_trial_paths(results_dir, model, dataset, trial))
    raise FileNotFoundError(
        f"Could not find trial_{trial}.json for model={model!r}, dataset={dataset!r} under {results_dir}.\n"
        f"Tried:\n{tried}"
    )


def _get_query_type(entry: dict) -> Optional[str]:
    # Expected layout in your example: entry["metadata"]["query_type"]
    md = entry.get("metadata")
    if isinstance(md, dict) and isinstance(md.get("query_type"), str):
        return md["query_type"]
    # Fallback if query_type is top-level.
    qt = entry.get("query_type")
    if isinstance(qt, str):
        return qt
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare human y/n answers against VLM trial_*.json results for a given query_type."
    )
    parser.add_argument("vlm_results_dir", type=Path, help="Path to VLM generated results directory")
    parser.add_argument("human_responses", type=Path, help="Path to human responses .txt file")
    parser.add_argument("--trial", type=int, default=1, help="Which trial_N.json to load (default: 1)")
    args = parser.parse_args()

    model, dataset, query_type, human_answers = _read_human_file(args.human_responses)

    trial_path = _find_trial_json(args.vlm_results_dir, model, dataset, args.trial)
    trial = json.loads(trial_path.read_text())
    results = trial.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{trial_path} missing 'results' list")

    filtered = [r for r in results if _get_query_type(r) == query_type]

    if len(filtered) != len(human_answers):
        raise ValueError(
            f"Count mismatch after filtering by query_type={query_type!r}.\n"
            f"VLM entries: {len(filtered)}\n"
            f"Human answers: {len(human_answers)}\n"
            f"Trial file: {trial_path}"
        )

    matches = 0
    total = len(human_answers)
    mismatches: List[Tuple[int, str, str, str]] = []
    for idx0, (r, human) in enumerate(zip(filtered, human_answers)):
        if "response" not in r:
            raise ValueError(f"{trial_path}: filtered results[{idx0}] missing 'response' field")
        vlm = _normalize_yesno(str(r["response"]))
        if vlm == human:
            matches += 1
        else:
            video = str(r.get("video", "<missing-video>"))
            mismatches.append((idx0 + 1, video, human, vlm))

    accuracy_pct = (100.0 * matches / total) if total else 0.0
    print(f"{matches}/{total} ({accuracy_pct:.2f}%)")
    if mismatches:
        for idx1, video, human, vlm in mismatches:
            print(f"index_1based={idx1}\tvideo={video}\thuman={human}\tvlm={vlm}")


if __name__ == "__main__":
    main()

