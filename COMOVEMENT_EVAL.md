# Co-movement evals

Two new datasets pulled from
`gs://solaris-central1/solaris/data/neurips_eval_coMovement/` into
`mc_multiplayer_v2_eval_new_sneak_combined_simplified_naming/`:

| Dataset | Pairs | Queries | Notes |
|---|---|---|---|
| `coMovementEval` | 32 | 64 | open ground — **this is the one to report** |
| `coMovementWithDividerEval` | 32 | 64 | a divider between the bots — **excluded, see below** |

Each is 16 episodes x 2 instances, 32 pairs — the same size as the existing
evals, so the default 32-pair cap in `run_eval.py` takes all of them.

## How this differs from translationEval

|  | translationEval | co-movement |
|---|---|---|
| Sneak | one bot | **both** |
| Movement | one chunk, one bot | **two chunks, both bots** |
| Answer source | the mover's action label | **relative geometry** |
| Answer per camera | shared | **one per perspective** |

Episode structure, identical across all 64 pairs of both datasets:

```
~f48   both bots sneak                     <- episode start; generation begins here
~f49   chunk 0: both walk forward          <- shared approach, NOT the question
~f90   chunk 1: the tested co-movement     <- queried: f_start .. f_start + 40
```

The query deliberately spans only chunk 1. Spanning from the episode start
would fold the shared approach into the answer — for a `forward` + `back`
episode the pair ends up slightly closer overall, which would make the correct
answer "closer" even though the tested movement produced no relative motion at
all.

## Ground truth is geometric, not label-derived

With both bots moving, no single action label determines what a camera sees.
The other bot's displacement is projected onto the observer's forward and right
axes and the dominant component wins; below a 0.75-block deadzone in both axes
the answer is "no motion".

The camera convention was **measured, not assumed**. Holding each direction key
and comparing travel angle `atan2(dx, dz)` against yaw over translationEval
gives `forward -> yaw+180deg`, `right -> yaw+90deg`, hence

```
forward = (-sin(yaw), -cos(yaw))
right   = ( cos(yaw), -sin(yaw))
```

A guessed convention agreed with translationEval only **18/64** times; the
measured one agrees **64/64**. `analyze_comovement.py` runs that check —
re-run it after touching the geometry.

## The design is 50/50 by construction

Eight action combinations, four pairs each:

| alpha + bravo | Relative effect | Expected |
|---|---|---|
| forward + forward | converge | closer (x8) |
| back + back | separate | farther (x8) |
| left + left | separate laterally | right (x8) |
| right + right | separate laterally | left (x8) |
| forward + back | both move the same way through the world | **no motion** (x8) |
| back + forward | same | **no motion** (x8) |
| left + right | same | **no motion** (x8) |
| right + left | same | **no motion** (x8) |

Because the bots face each other, opposite *actions* mean the same *world*
direction, so the on-screen relationship is unchanged. That is what the eval is
really testing: whether the model separates ego-motion from relative motion.

**32 of 64 queries answer "no motion"**, so a model that always says "no
motion" scores 50%. Report the per-class breakdown, not just the aggregate —
`score_comovement.py` does this.

## The prompt was chosen by A/B, not by taste

translationEval's wording — *"did the player being shown move closer, farther,
to the left, or to the right on-screen?"* — reads as a question about the
world. That is a fair reading: the other player really is walking in every
episode. It cost most of the "no motion" class.

Measured over all 64 GT queries per variant (`prompt_ab_comovement.py`):

| Prompt | coMovementEval | | coMovementWithDivider | |
|---|---|---|---|---|
| | overall | no motion | overall | no motion |
| translation-style baseline | 78.1% | 56.2% | 51.6% | 3.1% |
| screen-relative | 85.9% | 71.9% | 54.7% | 9.4% |
| **screen-relative + ignore landmarks** | **98.4%** | **96.9%** | 53.1% | 6.2% |

All four motion classes stayed at 100% for every variant, so nothing was traded
away to buy the "no motion" gain.

Two things the prompt deliberately does **not** do:

* It says nothing about the action structure. A line like *"if both players
  walk the same way, answer no motion"* would hand over the answer for half the
  queries and inflate the score without measuring anything.
* It does not mention the divider, so the same prompt is used for both
  datasets and they stay comparable.

Judge prompt changes on **balanced accuracy across the five classes**, not
overall: with half the queries being "no motion", a prompt that merely biases
toward it gains several points while getting worse.

### coMovementWithDividerEval is excluded — do not report it

**Use `coMovementEval` only.** The divider set is not a reskin of it and its
numbers are not trustworthy.

Under the same prompt that takes the regular eval's "no motion" recall from 56%
to 91%, the divider set sits at 7.3% — barely above chance overall (53.6% vs a
50% baseline) — and **84 of its 89 errors answer `farther`**. That lopsidedness
is the tell: a reasoning failure would spread errors across the classes.

The cause looks like occlusion, not reasoning. The divider is a static block
between the bots, so when both move the same way the observer closes on the
block and it hides progressively more of the other player. On-screen size is
unchanged, which is why the geometry says "no motion", but more of the player
is covered and that reads as receding. Its "no motion" class is therefore
measuring how much the block hides, not whether the model separates ego-motion
from relative motion. No prompt fixes a property of the stimulus.

Consequences, already wired in:

* `score_comovement.py` skips it and prints why; `--include-unreliable`
  overrides.
* `run_all_evals.py` leaves `co_movement_divider` out of `ENABLED_EVAL_TYPES`.
* The handler still supports the dataset, so an explicit run works if the
  stimulus is ever regenerated without the confound.

The collected results stay in `results_json_comovement/real/` as the evidence
for this call, not as numbers to quote.

## Generated videos

Not yet available. One thing is already handled: the generated path in
`extract_query_frames` used to hardcode generated frame 0 as the "before"
image, which is the episode start rather than the pre-test frame this eval
needs. It now honours `query.frame_index` when that is later than the episode
start. For every other two-frame handler the two are equal, so it is a no-op —
verified across all 64 translationEval queries, and by byte-comparing extracted
generated frames before and after the change.

`metadata["frame1"]` still means *episode start* (generated frame 0 is
`frame1 + 1`), so offset arithmetic elsewhere is unaffected.

When generations land, confirm the generated clips start at the sneak like the
other evals. If they instead start at the tested chunk, `frame1` must change to
match, and the numbers should be rebuilt.

`run_all_evals.py` knows the eval types `co_movement` and `co_movement_divider`
but leaves them out of `ENABLED_EVAL_TYPES` — with no generations they would be
skipped for every model. Pass `--eval-types co_movement` explicitly.

## Reproducing

```bash
python3 analyze_comovement.py                      # validate geometry, survey both sets
python3 run_eval.py \
    mc_multiplayer_v2_eval_new_sneak_combined_simplified_naming/coMovementEval \
    --num-trials 3 --results-dir results_json_comovement
python3 score_comovement.py                        # per-class breakdown
```
