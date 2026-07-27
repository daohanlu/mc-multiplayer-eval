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

## Results on generated videos

Pulled from `gs://solaris-east5/outputs/neurips_eval_coMovement/co_movement/`
into `generations_comovement/co_movement/` (32 clips, 1280x704, 257 frames —
same geometry and length as every other generation in this repo). The bucket
carries **no model label**; only one non-divider run exists, so it is scored
here as `solaris`. The divider run in the same bucket was not evaluated.

Gemini 3 Flash, thinking off, 3 trials, 64 queries:

| | GT | generated |
|---|---|---|
| all queries | 95.3% +/- 1.3 | **69.3% +/- 0.7** |
| no-motion cases excluded (n=32) | 100.0% +/- 0.0 | **96.9% +/- 0.0** |
| no-motion cases only (n=32) | 90.6% +/- 2.6 | **41.7% +/- 1.5** |
| episode-level, all queries | 90.6% +/- 2.6 | **63.5% +/- 1.5** |
| episode-level, no-motion excluded | 100.0% +/- 0.0 | **93.8% +/- 0.0** |

Reporting only the aggregate would be misleading in both directions here: the
model is essentially perfect on the four directional classes and the entire gap
to GT sits in the "no motion" half.

The failure is not spread across the no-motion cases either — it is one
mechanism, visible in the per-combo table:

| alpha + bravo | expected | generated |
|---|---|---|
| forward + back | no motion | 83.3% |
| back + forward | no motion | 83.3% |
| left + right | no motion | **0.0%** |
| right + left | no motion | **0.0%** |

Every one of those 48 queries answers `right`. Front-back co-movement is held;
lateral co-movement is not. The side-by-side comparisons under
`frame_extraction_side_by_side/coMovementEval/solaris/` show it directly: when
both bots strafe together the generated partner drifts across the frame and
grows, while in GT they stay put.

The lateral combos are also the eval's hardest cases for the VLM on GT (87.5%
and 75.0%), so some difficulty is intrinsic — but 0% against 87.5%/75.0% is far
past that.

### The prompt is not what makes the motion-only number high

96.9% on the directional classes is well above the paper's Movement column
(flagship `translationEval`, 67.7% episode-level), which invites the worry that
the screen-relative prompt is doing the work. It is not — re-running the same
32 generated motion queries under the older wordings, 3 trials each:

| Prompt on generated motion-only queries | overall |
|---|---|
| `translation_exact` (translationEval's prompt, byte-for-byte) | **100.0% +/- 0.0** |
| `baseline` (what shipped with this handler) | **100.0% +/- 0.0** |
| `screen_relative` | **100.0% +/- 0.0** |
| `ignore_landmarks` (current) | 95.8% +/- 1.5 |

The current prompt is if anything slightly *worse* here: its "if they look the
same, answer no motion" clause costs a few `farther` cases. Everything the
screen-relative wording bought was in the no-motion half, which is what it was
chosen for. Reproduce with:

```bash
python3 prompt_ab_comovement.py --datasets coMovementEval \
    --generated-subdir generations_comovement/co_movement \
    --exclude-no-motion --trials 3 \
    --variants translation_exact baseline screen_relative ignore_landmarks
```

Nor is it easier geometry. Scoring translationEval's queries through this
handler's projection (which reproduces its labels 64/64) gives a mean
dominant-axis displacement of **7.11 blocks**, against **4.74** for
co-movement's front/back cases and **6.04** for its lateral ones — co-movement's
motion cases carry *less* relative displacement, not more.

So the gap to the Movement column is not the prompt and not the geometry. The
untested difference is the generations themselves: `results_json/generated/
flagship_translationEval` is from the January generation set and eval vintage,
while these clips were rendered this week. Comparing the two as if they were
one experiment is not safe.

### Alignment was checked before trusting the numbers

Generated frame 0 corresponds to GT frame `frame1 + 1` here, as everywhere
else. Two checks: the frame-difference profile of generated frame 0 against the
whole GT clip has its knee at GT frame ~48 (`frame1`), and the generated clips
match the other evals' pipeline exactly in resolution and length. The 96.9% on
directional classes is itself corroboration — a shifted offset would have cost
the closer/farther classes first.

Two bugs found while validating, both fixed:

* `visualization_helper.py` labelled the first comparison frame with
  `meta['frame1']` (episode start) rather than the frame actually extracted.
  Equal for every other handler; off by ~38 frames for co-movement. The images
  were regenerated.
* `run_eval.py` set `using_generated` from `generated_path`, which is `None`
  when `--generated-subdir` is used, so generated runs were stamped as ground
  truth. Fixed, and the flag corrected in the three trial files already written.

## Generated videos: mechanics

One thing is already handled: the generated path in
`extract_query_frames` used to hardcode generated frame 0 as the "before"
image, which is the episode start rather than the pre-test frame this eval
needs. It now honours `query.frame_index` when that is later than the episode
start. For every other two-frame handler the two are equal, so it is a no-op —
verified across all 64 translationEval queries, and by byte-comparing extracted
generated frames before and after the change.

`metadata["frame1"]` still means *episode start* (generated frame 0 is
`frame1 + 1`), so offset arithmetic elsewhere is unaffected.

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

# generated
gsutil -m cp -r gs://solaris-east5/outputs/neurips_eval_coMovement/co_movement \
    generations_comovement/
python3 run_eval.py \
    mc_multiplayer_v2_eval_new_sneak_combined_simplified_naming/coMovementEval \
    --generated-subdir generations_comovement/co_movement --model-name solaris \
    --num-trials 3 --results-dir results_json_comovement
```
