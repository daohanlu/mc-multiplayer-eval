# What this tree is

A rename-and-copy of `../mc_multiplayer_v2_eval_new_sneak_combined`, plus the
three co-movement eval sets that only exist here. No clip, action log or
annotation was modified: every folder shared with the source tree is
byte-identical to it (verified with `diff -rq` per folder, 2026-07-30).

| Here | Source folder | diff |
| --- | --- | --- |
| `oneLooksAwayEval` | `oneLooksAwayEval_long` | identical |
| `bothLookAwayEval` | `bothLookAwayEval_long` | identical |
| `rotationEval` | `rotationEval` | identical |
| `structureEval` | `structureEval` | identical |
| `structureNoPlaceEval` | `structureNoPlaceEval` | identical |
| `translationEval` | `translationEval` | identical |
| `turnToLookEval` | `turnToLookEval` | identical |
| `turnToLookOppositeEval` | `turnToLookOppositeEval` | identical |
| `coMovementEval` | — (new here) | |
| `coMovementWithDividerEval` | — (new here) | |
| `coMovementAlwaysRelativeMotionEval` | — (new here) | |

So the two "look away" names drop the `_long` suffix: `oneLooksAwayEval` here
**is** the long variant the paper's Grounding column uses (the
`oneLooksAwayEval_long` handler in `build_vlm_tables.py` /
`run_all_evals.py`). The source tree's non-long variants were already retired
there as `DO_NOT_USE_one!Looks!Away!Eval` and `DO_NOT_USE_both!Look!Away!Eval`
and were not copied.

Re-verify with:

    for d in *Eval*; do diff -rq "$d" ../mc_multiplayer_v2_eval_new_sneak_combined/<source-name>; done
