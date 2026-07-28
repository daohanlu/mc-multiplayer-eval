#!/usr/bin/env python3
"""Carry consistency answers across a change to CONSISTENCY_MODELS.

Item ids (``c0001``, ``c0002``, …) are assigned by position in a shuffled list,
so changing the model set renumbers everything: the same id points at a
different screenshot pair before and after a rebuild. Answers therefore cannot
be matched by id. They are matched on the query itself —
``(model, eval, query_type, episode, instance)`` — which is stable across
rebuilds because it describes what the annotator actually saw.

Answers whose query no longer exists (a dropped model) are removed. Answers for
a retained model are rewritten under their new id, so an annotator who has
already judged those pairs sees their own choice again rather than a blank.

Usage
-----
::

    # after rebuilding with the new CONSISTENCY_MODELS
    python migrate_consistency_responses.py --old-key backups/<stamp>/consistency_key.OLD.json

    python migrate_consistency_responses.py --old-key ... --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESPONSES = HERE / "responses"
NEW_KEY = HERE / "data" / "consistency_key.json"

# The query identity — everything that determines which pair of screenshots the
# annotator was shown. Deliberately excludes the item id.
FIELDS = ("model", "eval", "query_type", "episode", "instance")


def signature(entry: dict) -> tuple:
    return tuple(entry[f] for f in FIELDS)


def load_key(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"key not found: {path}")
    return {i["id"]: i for i in json.loads(path.read_text())["items"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-key", required=True, type=Path,
                    help="consistency_key.json as it was BEFORE the rebuild")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    old = load_key(args.old_key)
    new = load_key(NEW_KEY)

    old_sig = {oid: signature(e) for oid, e in old.items()}
    new_by_sig = {signature(e): nid for nid, e in new.items()}

    old_models = sorted({e["model"] for e in old.values()})
    new_models = sorted({e["model"] for e in new.values()})
    print(f"models before : {old_models}")
    print(f"models after  : {new_models}")
    print(f"retained      : {sorted(set(old_models) & set(new_models))}")
    print(f"dropped       : {sorted(set(old_models) - set(new_models))}")
    print(f"added         : {sorted(set(new_models) - set(old_models))}")
    print()

    files = sorted(RESPONSES.glob("consistency__*.json"))
    if not files:
        raise SystemExit(f"no consistency responses in {RESPONSES}")

    for path in files:
        data = json.loads(path.read_text())
        answers = data.get("answers", {})

        migrated: dict[str, dict] = {}
        dropped = unknown = 0
        for old_id, ans in answers.items():
            sig = old_sig.get(old_id)
            if sig is None:
                unknown += 1        # id not in the old key at all
                continue
            new_id = new_by_sig.get(sig)
            if new_id is None:
                dropped += 1        # this model is no longer part of the task
                continue
            carried = dict(ans)
            # index is a per-annotator display position; the reshuffle
            # invalidates it and show() recomputes it on the next answer.
            carried.pop("index", None)
            carried["migrated_from"] = old_id
            migrated[new_id] = carried

        data["answers"] = migrated
        data["answered"] = len(migrated)
        data["total"] = len(new)
        data.setdefault("migrations", []).append({
            "date": datetime.date.today().isoformat(),
            "reason": "consistency models changed "
                      f"{old_models} -> {new_models}",
            "carried": len(migrated),
            "dropped_model_removed": dropped,
            "dropped_unrecognised_id": unknown,
        })

        print(f"{path.name}")
        print(f"   before  : {len(answers)}")
        print(f"   carried : {len(migrated)}")
        print(f"   dropped : {dropped} (model removed)"
              + (f", {unknown} unrecognised" if unknown else ""))
        print(f"   after   : {len(migrated)} / {len(new)}")

        if not args.dry_run:
            path.write_text(json.dumps(data, indent=2))

    if args.dry_run:
        print("\n(dry run — nothing written)")
    else:
        # deploy.sh excludes responses/ from its rsync, so it does NOT carry
        # these rewritten files to the remote. They must be pushed explicitly,
        # or the server keeps serving answers keyed to the old numbering.
        print("\nmigrated. The remote still holds the OLD ids. Push in this order:")
        print("  ./deploy.sh stop")
        print("  ./deploy.sh                 # uploads the rebuilt task, NOT responses/")
        print("  rsync -az ./responses/ \\")
        print("      \"${REMOTE:-fred@69.30.0.74}:${REMOTE_DIR:-/nas2/fred/solaris-human-eval}/responses/\"")
        print("  ./deploy.sh restart")


if __name__ == "__main__":
    main()
