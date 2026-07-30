# Human evaluation — results

Two questions, two tasks:

* **Consistency** — does the human ranking of two models match the VLM judge's
  ranking, and by how much? Yes, for every annotator, and the human margin is
  slightly larger than the judge's. It also survives a threshold-free metric
  (panel-vote AUROC 0.87 human vs 0.74 VLM on flagship).
* **Artifacts** — how often do humans see visible corruption in a clip? Frame
  Concat is unusable (2.2% clean); among our own ablations the clean rate is
  *anti-correlated* with capability, so it must not be read as a quality score
  on its own.

## Provenance

| Item | Value |
|---|---|
| Answers | `responses/{consistency,artifacts}__*.json`, pulled from `fred@69.30.0.74:/nas2/fred/solaris-human-eval/responses/` on 2026-07-27 |
| Annotators | egor, fred, Georgy, Oscar, srivats (both tasks) |
| Judgements | Consistency 1280 of 1280; Artifacts 315 of 315. Both tasks complete. |
| Answer keys | `data/consistency_key.json` (256 items), `data/artifacts_key.json` (63 items), `SHUFFLE_SEED` unchanged |
| VLM numbers | `results_json_late_episode_strict/generated/*/trial_{1,2,3}.json`, via `build_vlm_tables.get_cell` |
| Consistency models | `flagship` (Solaris Default) and `concat_c` (Frame Concat, Table 2) |
| Artifact models | all 7 in the supplementary material |

**The answers committed at `676789d` are stale.** That commit predates Oscar and
srivats, and Georgy was at 128/256 then (they finished 256/256 on 2026-07-27). Run `./deploy.sh fetch` before scoring,
or the numbers below will not reproduce.

# Part 1 — Consistency

## Method

Scoring follows `build_vlm_tables.get_cell(model, "con", ...)` exactly, so the
human and VLM columns are comparable:

* **STRICT toggle** — each episode contributes 2 queries, the original turn-end
  frame and its late-horizon duplicate.
* **AND semantics** — an episode counts correct only if the judge is right at
  *both* timestamps. An incomplete episode is excluded, not counted wrong.
* **Pooling** — `turnToLookEval` and `turnToLookOppositeEval` pool into one
  64-episode number per model.
* **Standard deviation** — population sd, matching `_pop_mean_std`. Across 3
  trials for the VLM, across 5 annotators for the humans. Where inference is
  reported instead (t-test, CI), it uses the sample sd, which is why the CI
  half-width does not match the `+/-` in the table.

Reproduce:

```bash
./deploy.sh fetch
python3 score_human_eval.py          # per-annotator
```

## Headline

| Judge | Solaris Default | Frame Concat | Margin |
|---|---|---|---|
| VLM judge, 3 trials | 56.8 +/- 2.9 | 25.5 +/- 3.2 | **+31.2 +/- 4.6** |
| Humans, 5 annotators | 67.5 +/- 6.2 | 32.5 +/- 7.3 | **+35.0 +/- 8.1** |
| Human majority vote | 82.8 | 34.4 | **+48.4** |

Episode-level accuracy, 64 episodes per model per judge. The VLM row reproduces
the paper's published 56.8 +/- 2.9 and 25.5 +/- 3.2 cell for cell, which
confirms the human scorer and the table builder agree on the pooling.

Relative separation is nearly identical across judges: 2.2x (VLM), 2.1x (human
mean), 2.4x (majority vote).

### Per judge

| Judge | flagship | concat_c | Margin | n episodes |
|---|---|---|---|---|
| VLM trial 1 | 54.7 | 21.9 | +32.8 | 64 / 64 |
| VLM trial 2 | 60.9 | 25.0 | +35.9 | 64 / 64 |
| VLM trial 3 | 54.7 | 29.7 | +25.0 | 64 / 64 |
| egor | 59.4 | 20.3 | +39.1 | 64 / 64 |
| fred | 78.1 | 29.7 | +48.4 | 64 / 64 |
| Georgy | 67.2 | 42.2 | +25.0 | 64 / 64 |
| Oscar | 64.1 | 34.4 | +29.7 | 64 / 64 |
| srivats | 68.8 | 35.9 | +32.8 | 64 / 64 |

**5 of 5 annotators, and 3 of 3 VLM trials, rank flagship above concat_c.**
Paired t-test on the 5 annotator margins: t(4) = 8.6, p = 0.0010, 95% CI
[23.7, 46.3] — which contains the VLM margin of +31.2. Sign test alone gives
p = 0.031 one-sided, without assuming normality.

Humans score *both* models higher than the judge does (+10.7 on flagship, +7.0
on concat_c). The judge is uniformly stricter; it does not favour our model.

### Query level, for reference (128 pairs per model, no AND rule)

| Judge | flagship | concat_c |
|---|---|---|
| egor | 66.4 | 52.3 |
| fred | 80.5 | 49.2 |
| Georgy | 75.0 | 50.8 |
| Oscar | 71.9 | 58.6 |
| srivats | 80.5 | 48.4 |
| VLM, 3 trials | 68.5 +/- 1.9 | 45.1 +/- 4.1 |

The AND rule over 2 timestamps is what separates the models sharply; at the
single-pair level concat_c sits near the 50% base rate for most judges.

## Human–VLM agreement

Raw per-pair answer match, over the 256 items. `maj3` is the VLM's
majority answer across its 3 trials.

| Annotator | vs VLM trial 1 | vs VLM maj3 | kappa (maj3) |
|---|---|---|---|
| egor | 56.2% | 55.5% | +0.17 |
| fred | 64.1% | 68.0% | +0.34 |
| Georgy | 70.7% | 71.5% | +0.37 |
| Oscar | 62.9% | 63.7% | +0.27 |
| srivats | 70.7% | 73.0% | +0.41 |
| majority of 5 | — | **67.6%** | **+0.32** |

Mean pairwise **human–human** agreement is **68.1%** (range 57.4–76.6%, kappa
+0.22 to +0.47). This is the number that makes the row above meaningful: the
judge disagrees with a human no more than two humans disagree with each other.
The item-level task is genuinely hard; the model *ordering* is what is stable.

> **What kappa is, in one line.** How much better than luck the raters did.
> Observed agreement minus chance agreement, divided by the room left above
> chance: `k = (p_o - p_e) / (1 - p_e)`. Chance agreement `p_e` comes from how
> often each rater reaches for each label — two raters who both answer
> "different" 70% of the time collide often without either one knowing anything.
> 0 = no better than luck, 1 = perfect. Cohen's is the 2-rater form, Fleiss' the
> many-rater form. Worked example from Part 2: raters called clips clean 61% of
> the time, so `p_e = 0.61^2 + 0.39^2 = 0.524`, observed was 0.762, giving
> `(0.762 - 0.524)/(1 - 0.524) = 0.50`.
>
> **When to quote it and when not to.** Quote it when there is no comparison
> baseline and one label dominates, which is the artifact study — raw agreement
> there is inflated by most clips being clean. Skip it when the claim is already
> a *comparison* of two agreement rates, as in the 67.6% vs 68.1% above: both
> sides face the same base rates, so the comparison self-corrects and the kappa
> adds nothing except an invitation to argue about Landis–Koch adjectives (0.32
> is "fair", 0.50 is "moderate"). The rebuttal quotes it for artifacts only, and
> states the 52.4% chance level inline so the number explains itself.

### Per-sample alignment, the way reviewer nAFu asked for it

`agreement_extras.py` prints it. Scope is 384 items, the 3 models with a full
annotator panel: flagship, `no_player_attn_sf` and `concat_c`. Every figure
compares two answers **item by item**; no aggregate accuracy enters it, which is
the point of the reviewer's question — the VLM judge and a human annotator can
match on marginal accuracy while disagreeing on nearly every pair.

Two terms only, here and in the response: the **VLM judge** and the **human
annotators**. A human annotator is never called a judge.

A human annotator row is that human annotator compared with each of the other
four, one at a time, then averaged. The VLM judge row is the VLM judge compared
with each of the five the same way. No consensus label is built, so no vote and
no tie-breaking rule can move a row.

**The VLM judge row follows the paper's convention**: score each of the 3 trials
separately, average, and report the population sd across trials. The paper never
merges the trials into one answer, so neither does this. Merging them into a
majority-of-3 answer would give 66.9% and kappa +0.29 — **+1.1 points better** —
which is exactly why it is not quoted: it is a judge configuration the paper does
not evaluate, and it denoises the VLM judge in a way no human annotator row gets.

| Compared with the human annotators | Exact agreement | Cohen's kappa | pairings |
|---|---|---|---|
| human annotator srivats | 71.5% | +0.40 | 4 |
| human annotator Georgy | 71.5% | +0.39 | 4 |
| human annotator Oscar | 70.7% | +0.39 | 4 |
| human annotator fred | 69.7% | +0.36 | 4 |
| human annotator egor | 57.7% | +0.24 | 4 |
| **Five human annotators** | **68.2 +/- 5.3%** | **+0.36 +/- 0.06** | 5 |
| **VLM judge, 3 trials** | **65.7 +/- 0.3%** | **+0.26 +/- 0.01** | 5 |
| VLM judge, t1 / t2 / t3 | 66.2 / 65.4 / 65.6% | +0.28 / +0.26 / +0.25 | 5 |

The "Five human annotators" row is the mean and population sd over the 5 rows
above it. It equals the mean over the 10 distinct annotator pairs, since each
pair is counted once in each of its two members' rows.

**The ceiling: two VLM judge trials on the same image agree 80.7%, kappa +0.56.**
The VLM judge samples its answer, so it does not reproduce itself, and no row
above can be expected to beat that. Against it, a VLM judge at 65.7% sitting
fifth of six — and 2.5 points under the 68.2% two human annotators average
against each other — is not the weak link. Cohen's kappa also disposes of the
reviewer's stated worry directly: a VLM judge reproducing only the marginals
scores 0.00, not +0.26.

### Do not use the majority-vote version of this table

An earlier draft scored every row against the **majority answer** of the human
annotators other than that row. **That was wrong and is not quotable.** A human
annotator's panel is the other 4, which is even, so 2–2 ties occur on 54 to 90 of
the 384 items and get broken by convention. The VLM judge's panel is all 5, which
is odd, so it never ties.

| Row | tie → "no" | tie → "yes" | tied items |
|---|---|---|---|
| Georgy | 82.8% | 68.8% | 80 |
| srivats | 81.5% | 71.1% | 90 |
| fred | 76.8% | 70.3% | 83 |
| Oscar | 73.4% | 74.7% | 65 |
| egor | 52.3% | 60.2% | 54 |
| VLM judge, majority of 3 | 70.6% | n/a | 0 |

Flipping the convention moves a human annotator row by up to 14 points and leaves
the VLM judge row untouched, so the rows were never comparable. `pool_human_eval.majority`
resolves ties to "no" and documents the assumption that "both current panels are
odd" — true of its own callers, false for the leave-one-out panels used here.
The pairwise table above has no such dependency.

**Caveat — do not quote per-model kappa.** Splitting the majority-vs-maj3
agreement by model gives flagship 68.0% (kappa +0.35) but concat_c 67.2%
(kappa **+0.09**). Agreement on concat_c is near-total in raw terms only because
both judges answer "different" to almost everything there, so kappa correctly
discounts it. The ranking conclusion is unaffected; the kappa is not evidence of
judge quality on that model.

## Threshold effects, and the threshold-free version

> An earlier draft of this section claimed that cross-judge absolute accuracy is
> confounded by response bias. **That was wrong** and is corrected below: the item
> set is exactly balanced, and on a balanced set accuracy is already bias-corrected.
> What survives is narrower and more interesting.

Each judge applies its own private threshold for "same scenery". The task is
balanced by construction — 50% of the 256 items have `expected == yes`
(`turnToLookEval`) and 50% `expected == no` (`turnToLookOppositeEval`) — but the
judges are nowhere near calibrated to it.

| Judge | "same" rate | sensitivity (expected yes) | specificity (expected no) | "same" rate: flagship / concat_c |
|---|---|---|---|---|
| egor | 64.8% | 74.2% | 44.5% | 78.9% / 50.8% |
| Oscar | 48.0% | 63.3% | 67.2% | 71.9% / 24.2% |
| fred | 43.8% | 58.6% | 71.1% | 47.7% / 39.8% |
| srivats | 33.2% | 47.7% | 81.2% | 47.7% / 18.8% |
| Georgy | 30.9% | 43.8% | 82.0% | 51.6% / 10.2% |
| VLM trial 1 | 39.8% | 45.3% | 65.6% | 53.9% / 25.8% |
| VLM trial 2 | 37.5% | 43.8% | 68.8% | 53.9% / 21.1% |
| VLM trial 3 | 37.5% | 46.1% | 71.1% | 49.2% / 25.8% |

Humans span **30.9% to 64.8%**; the VLM sits in a tight band near **38%**. egor
and Georgy are close to opposite ends of the sensitivity/specificity trade-off
while landing 6 points apart on flagship accuracy.

The per-model "same" rate columns are signal, not bias: every judge answers
"same" more often on flagship than on concat_c, which is the effect being
measured.

### What response bias does and does not do here

Because the base rate is exactly 50%, accuracy equals **balanced accuracy**, and

```
acc = (TPR + TNR)/2 = (1 + J)/2      where J = TPR - FPR   (Youden's J)
```

so accuracy is an exact affine function of J, which is bias-corrected by
construction. Verified on the data: egor's flagship accuracy 66.4% ↔ J = 32.8.

**Consequence: a skewed threshold cannot inflate accuracy.** A judge that answers
"same" to everything gets TPR = 1, FPR = 1, J = 0, accuracy exactly 50%. Same for
always-"different". So the 31%–65% spread in "same" rate is *not* a confound on
the reported accuracies, and comparing the human mean (67.5) to the VLM (56.8) is
legitimate.

**What does survive:** accuracy is a measurement at *one operating point*. It
under-reports a judge that separates the classes well but places its threshold
badly. `egor` is the clear case — TPR 95.3% / FPR 62.5% on flagship, very
liberal, so 66.4% understates their discrimination:

| Judge | flagship TPR / FPR | J | concat_c TPR / FPR | J |
|---|---|---|---|---|
| egor | 95.3 / 62.5 | 32.8 | 53.1 / 48.4 | 4.7 |
| fred | 78.1 / 17.2 | 60.9 | 39.1 / 40.6 | −1.6 |
| Georgy | 76.6 / 26.6 | 50.0 | 10.9 / 9.5 | 1.6 |
| Oscar | 93.8 / 50.0 | 43.8 | 32.8 / 15.6 | 17.2 |
| srivats | 78.1 / 17.2 | 60.9 | 17.2 / 20.3 | −3.1 |
| VLM t1 | 71.9 / 35.9 | 35.9 | 18.8 / 32.8 | −14.1 |
| VLM t2 | 75.0 / 32.8 | 42.2 | 12.5 / 29.7 | −17.2 |
| VLM t3 | 65.6 / 32.8 | 32.8 | 26.6 / 25.0 | 1.6 |

**Margin in J: humans +45.9 +/- 16.1, VLM +46.9 +/- 11.7** — the two judge types
land on essentially the same effect size once the base rate is factored out. (At
50% base rate the J margin is exactly twice the query-level accuracy margin, so
this is not independent evidence; it is the same number on a bias-free scale. The
*episode*-level margin of +35.0 vs +31.2 is a genuinely different quantity,
because the AND rule over 2 timestamps is not affine in J.)

### A real rank metric, from panel vote counts

A single judge's binary label gives **one point** in ROC space, not a curve — no
confidence score to rank by, so no per-judge AUROC. (This, not "a fixed count of
slots", is the actual obstacle. Sorting into a fixed count of slots is
forced-choice, a different constraint, and AUROC does not require it.)

But a *panel* has a graded decision variable for free: **how many judges voted
"same"**. That is rankable, so AUROC is computable — the normalized
Mann–Whitney U, P(a random `expected==yes` pair outranks a random
`expected==no` pair), with 0.5 credit for ties. Per query, 128 per model, no AND
rule.

| Panel | flagship | concat_c |
|---|---|---|
| 5 human annotators (6 score levels) | **0.898** | 0.541 |
| 3 of 5 humans, mean over all 10 subsets | **0.870 +/- 0.011** | 0.533 +/- 0.023 |
| VLM, 3 trials (4 score levels) | **0.744** | 0.430 |

The 3-of-5 row matters: a 3-rater panel has 4 score levels versus 6, and ties are
scored 0.5, so the 5-rater number is granularity-flattered. Compare **0.870 vs
0.744**, not 0.898 vs 0.744.

Two readings:

1. **The ranking survives a threshold-free metric.** Both panels separate
   flagship far above concat_c. The judge-circularity worry (R2/R3) does not hinge
   on where anyone put their threshold.
2. **Near-chance on concat_c is a statement about the model, not the judge.**
   Both panels sit at chance (human 0.54, VLM 0.43 — below chance, within noise of
   it at n=128). The `expected` label comes from ground-truth geometry; concat_c
   does not render the correct scenery in *either* condition, so there is nothing
   for any judge to discriminate. The human panel being at chance too is what
   rules out a judge defect as the explanation.

The panel AUROC of 0.898 sitting well above every individual accuracy (66–80%) is
the operating-point effect made visible: pooling votes recovers discrimination
that individual thresholds throw away.

All rows use the full n=128 per model: every annotator has now answered all 256
items, so the vote count is defined everywhere.

# Part 2 — Artifacts

63 clips = 7 models × 3 categories {Movement, Grounding, Building} × 3 clips.
5 annotators × 63 = **315 judgements, all collected**. One label per clip:
`none` / `character` / `building` / `other`. Cross-view disagreement and memory
failure are explicitly **excluded** from what counts as an artifact — the
Consistency axis measures those. There is no ground truth, so this is a rate,
not an accuracy.

No guide worked-example clip is in the scored set (`guide_example` is empty), so
every scored clip is blind. `fred`'s run carries a hand-set
`instruction_version: 2` with a note; see `README.md` for why that lock exists.

## Clean rate per model

Pooled over all 5 annotators (n=45 judgements per model). The CI is bootstrapped
over the **9 clips**, not the 45 judgements, since judgements of one clip are not
independent. VLM columns are the published table values, for context.

| Model | none | char | bld | other | Clean % | 95% CI (clip-level) | Majority-clean clips | Mov | Gnd | Bld | Con |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `causvid_dmd` | 39 | 1 | 4 | 1 | **86.7** | [77.8, 95.6] | 9/9 | 35.4 | 3.1 | 2.1 | 45.8 |
| `causvid_regression` | 37 | 0 | 6 | 2 | **82.2** | [64.4, 95.6] | 8/9 | 19.8 | 4.2 | 3.1 | 34.9 |
| `from_scratch` | 34 | 3 | 6 | 2 | **75.6** | [57.8, 91.1] | 7/9 | 69.8 | 25.0 | 0.0 | 32.8 |
| `no_player_attn_sf` | 33 | 6 | 4 | 2 | **73.3** | [60.0, 84.4] | 8/9 | 12.5 | 26.0 | 0.0 | 38.0 |
| `flagship` | 31 | 7 | 6 | 1 | **68.9** | [40.0, 93.3] | 6/9 | 67.7 | 53.1 | 9.4 | 56.8 |
| `no_kv_cache_backprop` | 17 | 0 | 3 | 25 | **37.8** | [20.0, 60.0] | 2/9 | 88.5 | 77.1 | 5.2 | 52.6 |
| `concat_c` | 1 | 2 | 3 | 39 | **2.2** | [0.0, 6.7] | 0/9 | 84.4 | 50.0 | 0.0 | 25.5 |

## The two things this table says

### 1. Frame Concat is visually unusable, and no VLM axis says so

44 of 45 judgements flag an artifact. 0 of 9 clips are clean by majority. **89%**
of the flags are `other`, i.e. generic corruption that is not a character or a
block problem. Yet `concat_c` has the **highest VLM Movement score of all 7
models** (84.4) and a mid-pack Grounding (50.0).

Head to head with flagship, per annotator: +77.8, +66.7, +77.8, +44.4, +66.7 —
mean **+66.7 +/- 12.2**, favouring flagship **5/5**. This is the single largest
and most unanimous effect in either study.

This is the useful rebuttal point for W2/W3: the VLM axes score *semantics*
(did the right thing happen), the human artifact study scores *image quality*,
and the baseline that beats us on Movement fails the second one completely.

**Be precise about what each study can and cannot see.** An earlier draft of the
rebuttal claimed the artifact study "separates action-following from image
quality, which the VLM axes do not do". That is backwards on both halves:

* Annotators see **one generated clip, with no action sequence and no reference
  video**, and are told to ignore cross-view disagreement and memory failure. The
  task therefore measures per-view image quality *only*. It says nothing about
  action-following, and cannot.
* The VLM **Movement** axis *is* an action-following measure. So is Grounding.
  What the VLM axes do not measure is image quality.

The correct claim is about the *pair* of measurements, not about either one:
Frame Concat follows actions (Movement 84.4, the best of the 7) and still renders
corrupted frames (2.2% clean), and you only see that by holding the two
independent studies side by side.

Note also that **Consistency has an action-following component**, which is easy to
miss. `turnToLookEval` and `turnToLookOppositeEval` differ in the *actions* given
— turn to look at the same place versus the opposite direction — so the expected
answer is determined by whether the model followed the camera actions. A model
that ignores camera input fails the eval regardless of image quality. Consistency
is therefore not a pure image-quality axis, and the artifact study is not a
degraded version of it.

Versus FID/FVD: those also target image quality, but each returns one
distributional distance for a whole set, **and neither one is conditioned on the
input actions** — they compare generated frames against real frames as
distributions, so a model that ignores every action it is given can still post a
good FID. That is the reason the human artifact study is not redundant with the
FID columns, and the reason no single number in the paper covers both
action-following and image quality. The human labels also name the failure type
(`character` / `building` / `other`), which is what makes the
flagship-vs-Frame-Concat difference legible rather than just large.

### 2. Clean rate is anti-correlated with capability — do not use it as a score

Spearman rho over the 7 models:

| vs | rho |
|---|---|
| VLM Movement | **-0.64** |
| VLM Grounding | **-0.89** |
| VLM Consistency | +0.04 |

`causvid_dmd` and `causvid_regression` are the *cleanest* models and have
Grounding scores of **3.1** and **4.2**. They generate little motion, so there
is little to corrupt. `no_player_attn_sf` has Movement 12.5, the floor, and 73.3%
clean. A near-static generator is artifact-free and useless.

**So flagship ranking 5th of 7 on clean rate is not the finding it looks like**,
and its clip-level CI [40.0, 93.3] overlaps every ablation above it. Only
`concat_c` and `no_kv_cache_backprop` separate from it. flagship is the only
model that is simultaneously mid-pack-or-better on cleanliness and top on
Grounding, Building and Consistency.

If anyone asks for a single "quality" number, this is not it. Report it jointly
with the capability axes or not at all.

## flagship's own failure profile

| Category | Clean % (n=15) |
|---|---|
| Movement | 100.0 |
| Grounding | 66.7 |
| Building | 40.0 |

Of flagship's 14 flagged judgements, **7 are `character`**, **6 are `building`**,
and **1 is `other`** — so flagship's deficit is *not* generic corruption. That
matters for reading the table above: `concat_c` (89% `other`) and
`no_kv_cache_backprop` (89% `other`) are the generic-corruption models. flagship's
two named failure modes corroborate the paper's own limitation list: character
duplication/removal, and weak block placement next to the 9.4% Building accuracy.

Pooled over all 7 models the category ordering is the same — Building 47.6%,
Grounding 65.7%, Movement 69.5% clean — so Building is the hardest category for
everyone, not only for us.

### Exactly which flagship clips the annotators saw

All 9, worst first. `C` = character, `B` = building, `O` = other, `·` = clean.
Source paths are under `Model Generations on Eval/`, copied verbatim into the
task, and are the same files that ship in the supplementary material.

| id | Category | Clip | egor | fred | Georgy | Oscar | srivats | Flags |
|---|---|---|---|---|---|---|---|---|
| `a0060` | Grounding | `video_2_gen.mp4` | C | C | C | C | C | **5/5** |
| `a0061` | Building | `video_2_gen.mp4` | O | B | C | C | B | **5/5** |
| `a0018` | Building | `video_0_gen.mp4` | · | B | · | B | B | 3/5 |
| `a0054` | Building | `video_1_gen.mp4` | · | · | · | B | · | 1/5 |
| `a0029` | Grounding | `video_0_gen.mp4` | · | · | · | · | · | 0/5 |
| `a0050` | Grounding | `video_1_gen.mp4` | · | · | · | · | · | 0/5 |
| `a0024` | Movement | `video_0_gen.mp4` | · | · | · | · | · | 0/5 |
| `a0025` | Movement | `video_1_gen.mp4` | · | · | · | · | · | 0/5 |
| `a0003` | Movement | `video_2_gen.mp4` | · | · | · | · | · | 0/5 |

**5 of 9 clips are unanimously clean, and 13 of flagship's 14 flags come from
3 clips** (5 + 5 + 3). All three Movement clips and two of three Grounding clips draw not one
flag from any annotator. This is what the wide CI [40.0, 93.3] is made of: the
68.9% is effectively decided by two clips, both of which happen to be the
`video_2` of their category.

**`a0060` — Grounding `video_2`, the worst clip (unanimous `character`).** Verified
by inspection. The clip is alpha's view over bravo's, 640×704, 257 frames. In
**bravo's** view the observed player is present at n=0, gone by n=64, and back as
**2–4 duplicated copies** from n=128 to the end. Alpha's view stays stable. This
is the cleanest single instance of the duplicate/remove failure mode in the whole
study, and it is a good candidate if a reviewer asks for a concrete failure case.

**`a0061` — Building `video_2`, flagged by all 5 but with no label consensus.**
Also verified by inspection, and the split is *not* annotator noise: the clip has
two simultaneous defects. **Alpha loses the other player entirely** from n=64
onward (→ `character`), while **bravo places blocks that do not persist** and
grows thin sliver geometry on the horizon by n=192 (→ `building`). Bravo's
inventory decrements correctly, 64 → 63 → 60 → 58, so the action is registering;
it is the world state that fails to hold.

> **Task-design consequence.** The answer is single-select, so a clip with
> co-occurring defects forces each annotator to pick the one they found most
> salient. The per-label distribution is therefore "which defect each annotator
> judged worst", not "which defects this model has". Do not read the
> `character` / `building` split as a decomposition of the failure modes, and do
> not treat a label disagreement on a unanimously-flagged clip as unreliability.
> The binary clean/not-clean rate is unaffected.

## Reliability

| Measure | Value |
|---|---|
| Pairwise agreement, binary clean/not | 76.2% (range 68.3–90.5) |
| Pairwise agreement, exact 4-way label | 71.9% (range 63.5–87.3) |
| Fleiss kappa, binary | **+0.50** |
| Fleiss kappa, 4-way | **+0.50** |
| Unanimous clean | 21/63 clips |
| Unanimous artifact | 11/63 clips |
| Split | 31/63 clips |

Moderate agreement, and a strictness spread like the one in Consistency: overall
clean rate runs from **fred 47.6%** to **Georgy 73.0%** (egor 66.7, Oscar 57.1,
srivats 60.3). Note that the bias-cancellation argument from Part 1 does **not**
apply here — there is no ground truth and no balanced base rate, so a strict
annotator's clean rate really is lower rather than being corrected by the metric.
That is the whole reason to pool. Georgy recorded no `building` labels at all in 63 clips while
fred recorded 12, so the label mix is annotator-dependent as well as the
strictness. Pool across annotators; do not quote one annotator's rate.

The binary and 4-way kappa being equal is a coincidence of this data, not a
bug — the 4-way disagreements are mostly clean-vs-`other`, which the binary
collapse also counts as disagreement.

## Sample-size caveat

9 clips per model is small; that is `ARTIFACT_VIDEOS_PER_CELL = 3`, chosen to
keep the task at ~63 items per annotator. Raising it to 5 gives 105 clips and
tighter intervals, **but reassigns every artifacts item id and invalidates all
existing artifacts responses** (see `README.md`). Do not raise it without
re-collecting.

# Scripts

`score_human_eval.py` prints the per-annotator tables and the paper numbers
alongside. It does **not** pool across annotators. `pool_human_eval.py` is the
pooled analysis: majority votes, the margin t-test, agreement matrices, Fleiss
kappa, bootstrap CIs, Spearman correlations, panel AUROC and the threshold
tables. `agreement_extras.py` adds the three reference points for per-sample
alignment — judge against itself, annotator against annotator, and each judge
against a panel that does not contain it.

Everything here is derivable from the two keys plus `responses/` plus the strict
results tree. The join keys are:

* Consistency — `(model, eval, query_type, episode, instance)`, stable across
  rebuilds (this is what `migrate_consistency_responses.py` matches on).
* Artifacts — the item id, plus `model` and `category` from the key.

Worth folding into `score_human_eval.py` if this analysis gets repeated: a
pooled-across-annotators mode is the one thing every question above needed.

Action-following, which is a different question about the same generations, lives
in `action_following/`.
