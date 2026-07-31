#!/usr/bin/env python3
"""
Update a multiplayer eval CSV with VLM scores from generated stats.json files.

Given:
  - generated_root: directory containing subfolders named "{model}_{datasetEval}"
  - csv_path: a CSV whose "Name" column contains model names, and whose columns
    starting with "VLM" contain dataset names (often on a second line)

For each (model, VLM-column) pair, this script looks for:
  {generated_root}/{model}_{datasetEval}/stats.json
and, if found, reads the top-level "mean" field and writes it into the cell.

It writes an updated CSV next to the original with an "_update" suffix.
It also writes a second CSV with an "_update_with_std" suffix that has a
separate "SD" column inserted next to each VLM column.
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

# Mapping from human-readable column names to the actual folder name suffixes.
# These are the special cases where the column name doesn't directly translate.
DATASET_NAME_MAPPING: Dict[str, str] = {
    "grounding": "oneLooksAwayEval_long",
    "memory": "bothLookAwayEval_long",
    "building": "structureEval",
    "translation": "translationEval",
    "rotation": "rotationEval",
    "turn to look": "turnToLookEval",
    "turn to look opposite": "turnToLookOppositeEval",
}

# Mapping from human-readable model names (in CSV) to actual folder names.
MODEL_NAME_MAPPING: Dict[str, str] = {
    "ours": "flagship",
    "concat": "concat_c",
    "ours scratch": "from_scratch",
    "ours causvid dmd": "causvid_dmd",
    "ours w/o kv-bp": "no_kv_cache_backprop",
    "ours causvid ode reg.": "causvid_regression",
}

# Columns that should be skipped (averages that user fills manually)
SKIP_COLUMNS: set[str] = {
    "movement (translation + rotation)",
    "consistency",
}


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


def _should_skip_column(dataset_phrase: str) -> bool:
    """Check if this column should be skipped (averages that user fills manually)."""
    return dataset_phrase.lower() in SKIP_COLUMNS


def _get_dataset_folder_name(dataset_phrase: str) -> Optional[str]:
    """
    Convert a dataset phrase from the column header to the actual folder name suffix.
    Uses the DATASET_NAME_MAPPING for known mappings, otherwise falls back to camelCase.
    """
    normalized = dataset_phrase.lower().strip()
    if normalized in DATASET_NAME_MAPPING:
        return DATASET_NAME_MAPPING[normalized]
    # Fallback to camelCase + "Eval"
    cc = _camelcase_dataset(dataset_phrase)
    if cc:
        return f"{cc}Eval"
    return None


def _get_model_folder_name(model_name: str) -> str:
    """
    Convert a human-readable model name from the CSV to the actual folder name.
    Uses MODEL_NAME_MAPPING if available, otherwise returns the original name.
    """
    normalized = model_name.lower().strip()
    return MODEL_NAME_MAPPING.get(normalized, model_name)


def _is_vlm_column(col_name: str) -> bool:
    return col_name.replace("\r", "").lstrip().upper().startswith("VLM")


def _format_number(x: float) -> str:
    return str(x)


def _read_stats(stats_path: Path) -> Optional[Tuple[float, Optional[float]]]:
    """Read mean and std from a stats.json file. Returns (mean, std) or None."""
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
        mean_f = float(mean)
    except Exception:
        return None

    std_val = data.get("std")
    std_f: Optional[float] = None
    if std_val is not None:
        try:
            std_f = float(std_val)
        except Exception:
            pass

    return mean_f, std_f


def _find_name_column(fieldnames: List[str]) -> Optional[str]:
    for fn in fieldnames:
        if fn.strip().lower() == "name":
            return fn
    return None


def _make_sd_column_name(vlm_col: str) -> str:
    """Create an SD column name from a VLM column name (e.g. 'VLM ↑ \nTranslation' -> 'SD\nTranslation')."""
    dataset_phrase = _extract_dataset_phrase_from_vlm_column(vlm_col)
    if dataset_phrase:
        return f"SD\n{dataset_phrase}"
    return "SD"


def update_csv(generated_root: Path, csv_path: Path) -> Tuple[Path, Path, int]:
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

    # Build fieldnames for the with-std CSV: insert an SD column after each VLM column.
    vlm_to_sd: Dict[str, str] = {}
    fieldnames_with_std: List[str] = []
    for col in fieldnames:
        fieldnames_with_std.append(col)
        if _is_vlm_column(col):
            sd_col = _make_sd_column_name(col)
            vlm_to_sd[col] = sd_col
            fieldnames_with_std.append(sd_col)

    # Deep copy rows for the with-std variant (add empty SD columns)
    rows_with_std: List[Dict[str, str]] = []
    for row in rows:
        row_std = {**row}
        for sd_col in vlm_to_sd.values():
            row_std[sd_col] = ""
        rows_with_std.append(row_std)

    missing_count = 0
    for row_idx, (row, row_std) in enumerate(zip(rows, rows_with_std), start=2):
        model_name = (row.get(name_col) or "").strip()
        if not model_name:
            continue

        # Map human-readable model name to folder name
        model_folder_name = _get_model_folder_name(model_name)

        for col in vlm_cols:
            dataset_phrase = _extract_dataset_phrase_from_vlm_column(col)

            # Skip columns that are averages (user fills manually)
            if _should_skip_column(dataset_phrase):
                continue

            dataset_name = _get_dataset_folder_name(dataset_phrase)
            if not dataset_name:
                missing_count += 1
                print(
                    f"WARNING: could not parse dataset name from column {col!r} (row {row_idx}, model {model_name!r})",
                    file=sys.stderr,
                )
                continue

            subdir = generated_root / f"{model_folder_name}_{dataset_name}"
            stats_path = subdir / "stats.json"

            result = _read_stats(stats_path)
            if result is None:
                missing_count += 1
                print(
                    f"WARNING: missing stats for cell (model={model_name!r}, dataset={dataset_name!r}) at {stats_path}",
                    file=sys.stderr,
                )
                continue

            mean, std = result
            row[col] = _format_number(mean)
            row_std[col] = _format_number(mean)
            if std is not None:
                row_std[vlm_to_sd[col]] = _format_number(std)

    # Write mean-only CSV
    out_path = csv_path.with_name(f"{csv_path.stem}_update{csv_path.suffix}")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Write CSV with separate SD columns
    out_path_std = csv_path.with_name(f"{csv_path.stem}_update_with_std{csv_path.suffix}")
    with out_path_std.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_with_std, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_with_std)

    return out_path, out_path_std, missing_count


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

    out_path, out_path_std, missing = update_csv(generated_root=generated_root, csv_path=csv_path)
    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_path_std}")
    if missing:
        print(f"Warnings: {missing} cells not found on disk", file=sys.stderr)


if __name__ == "__main__":
    main()

