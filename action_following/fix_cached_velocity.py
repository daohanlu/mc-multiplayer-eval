#!/usr/bin/env python3
"""Rewrite the ``vel`` array in every cached ``.npz``.

``commanded_body_velocity`` had its player-frame axes wrong until the convention
was fitted to the recorded data. The cache written before that fix carries the
wrong ``vel``. No report reads it, so no published number changes, but a stale
array in a cache is a trap for whoever reads it next.

This pass touches only ``vel``. It reads the action JSON and never decodes video,
so it costs seconds rather than the hour the full sweep costs.

    python3 action_following/fix_cached_velocity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from af_data import ROOT, build_index  # noqa: E402
from af_flow import commanded_body_velocity  # noqa: E402
from extract_motion import DATASETS, MODELS, out_path  # noqa: E402

CACHE = ROOT / "action_following" / "cache"


def main() -> None:
    fixed = missed = 0
    for dataset in DATASETS:
        if not (CACHE / dataset).is_dir():
            continue
        for model in MODELS:
            if not (CACHE / dataset / model).is_dir():
                continue
            for ep in build_index(dataset, model):
                for player in ("alpha", "bravo"):
                    p = out_path(dataset, model, ep.episode, ep.instance, player)
                    if not p.exists():
                        continue
                    z = dict(np.load(p, allow_pickle=True))
                    n = len(z["cam"])
                    new = commanded_body_velocity(ep.actions(player),
                                                  ep.frame1 + 1, n)
                    if np.allclose(np.nan_to_num(z.get("vel", np.zeros_like(new))),
                                   new, atol=1e-6):
                        missed += 1
                        continue
                    z["vel"] = new.astype(np.float32)
                    np.savez_compressed(p, **z)
                    fixed += 1
        print(f"  {dataset}: {fixed} rewritten so far", flush=True)
    print(f"rewritten {fixed}, already correct {missed}")


if __name__ == "__main__":
    main()
