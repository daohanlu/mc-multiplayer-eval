#!/usr/bin/env python3
"""
Update a multiplayer eval CSV with VLM scores from generated stats.json files.

Given:
  - generated_root: directory containing subfolders named "{model}_{datasetEval}"
  - csv_path: a CSV whose "Name" column contains model names, and whose columns
    starting with "VLM" contain dataset names (often on a second line)

For each (model, VLM-column) pair, this script looks for:
  {generated_root}/{model}_{camelCase(dataset)+Eval}/stats.json
and, if found, reads the top-level "mean" field and writes it into the cell.

It writes an updated CSV next to the original with an "_update" suffix.
Warnings are printed for any missing cell/file/mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _camelcase_dataset(name: str) -> str:
    # Split on non-alphanumeric boundaries; keep letters/numbers.
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", name.strip()) if p]
    if not parts:
        return ""
    first = parts[0].lower()
    rest = [p[:1].upper() + p[1:].lower() if p else "" for p in parts[1:]]
    return "".join([first, *rest])


def _extract_dataset_phrase_from_vlm_column(col_name: str) -> str:
    """
    CSV headers in this project often look like:
      "VLM ↑ \nTranslation"
    We take the last non-empty line as the dataset phrase.
    """
    normalized = col_name.replace("\r", "").strip()
    lines = [ln.strip() for ln in normalized.split("\n") if ln.strip()]
    if not lines:
        return ""
    # Most commonly the dataset is on the last line.
    dataset = lines[-1]
    # If there's no newline and it still starts with VLM, strip a prefix.
    if dataset.upper().startswith("VLM"):
        dataset = re.sub(r"^VLM\b[^A-Za-z0-9]*", "", dataset, flags=re.IGNORECASE).strip()
    return dataset


def _is_vlm_column(col_name: str) -> bool:
    return col_name.replace("\r", "").lstrip().upper().startswith("VLM")


def _format_number(x: float) -> str:
    # Preserve useful precision but avoid noisy trailing zeros.
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s


def _read_mean(stats_path: Path) -> Optional[float]:
    try:
        with stats_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"WARNING: failed to read JSON: {stats_path} ({e})", file=sys.stderr)
        return None

    mean = data.get("mean")
    if mean is None:
        return None
    try:
        return float(mean)
    except Exception:
        return None


def _find_name_column(fieldnames: List[str]) -> Optional[str]:
    for fn in fieldnames:
        if fn.strip().lower() == "name":
            return fn
    return None


def update_csv(generated_root: Path, csv_path: Path) -> Tuple[Path, int]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV appears to have no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        name_col = _find_name_column(fieldnames)
        if name_col is None:
            raise ValueError(f'Could not find a "Name" column in: {csv_path}')

        vlm_cols = [c for c in fieldnames if _is_vlm_column(c)]
        if not vlm_cols:
            raise ValueError(f'No columns starting with "VLM" found in: {csv_path}')

        rows: List[Dict[str, str]] = []
        for row in reader:
            # Keep raw strings; csv.DictReader returns str values (or None).
            rows.append({k: (v if v is not None else "") for k, v in row.items()})

    missing_count = 0
    for row_idx, row in enumerate(rows, start=2):  # +1 for header, +1 for 1-based indexing
        model_name = (row.get(name_col) or "").strip()
        if not model_name:
            continue

        for col in vlm_cols:
            dataset_phrase = _extract_dataset_phrase_from_vlm_column(col)
            dataset_cc = _camelcase_dataset(dataset_phrase)
            if not dataset_cc:
                missing_count += 1
                print(
                    f"WARNING: could not parse dataset name from column {col!r} (row {row_idx}, model {model_name!r})",
                    file=sys.stderr,
                )
                continue

            dataset_name = f"{dataset_cc}Eval"
            subdir = generated_root / f"{model_name}_{dataset_name}"
            stats_path = subdir / "stats.json"

            mean = _read_mean(stats_path)
            if mean is None:
                missing_count += 1
                print(
                    f"WARNING: missing stats for cell (model={model_name!r}, dataset={dataset_name!r}) at {stats_path}",
                    file=sys.stderr,
                )
                continue

            row[col] = _format_number(mean)

    out_path = csv_path.with_name(f"{csv_path.stem}_update{csv_path.suffix}")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return out_path, missing_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update a multiplayer eval CSV with means from generated stats.json files."
    )
    parser.add_argument(
        "generated_root",
        type=Path,
        help='Path to generated root (e.g. "/data/oasis/fred_mc_eval/generated")',
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help='Path to the CSV to update (e.g. "/data/oasis/Solaris Experiments - Multiplayer v2.csv")',
    )
    args = parser.parse_args()

    generated_root: Path = args.generated_root
    csv_path: Path = args.csv_path

    if not generated_root.exists() or not generated_root.is_dir():
        raise SystemExit(f"generated_root is not a directory: {generated_root}")
    if not csv_path.exists() or not csv_path.is_file():
        raise SystemExit(f"csv_path is not a file: {csv_path}")

    out_path, missing = update_csv(generated_root=generated_root, csv_path=csv_path)
    print(f"Wrote: {out_path}")
    if missing:
        print(f"Warnings: {missing} cells not found on disk", file=sys.stderr)


if __name__ == "__main__":
    main()

