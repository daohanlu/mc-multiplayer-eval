# Action-following — results

One question: **does each player's rendered view do what that player's action
stream told it to do?** Reviewer nAFu asked for it per player, in the form
Matrix-Game reports.

Two answers:

* **Camera** — every model except `causvid_dmd` turns the commanded way on most
  turn events. What separates them is not direction but *restraint*: Solaris and
  the two Causal FT variants hold still on ~99% of no-command frames, while
  `no_player_attn_sf` (92.7), `concat_c` (85.4) and `causvid_regression` (64.1)
  invent camera motion. Solaris's own weakness is magnitude: it turns the right
  way but only 0.68–0.72 of the commanded angle.
* **Per player** — Solaris, `no_player_attn_sf` and `no_kv_cache_backprop` score
  Alpha and Bravo within ~2 points of each other. `concat_c` differs by 18–20
  points, so its concatenation scheme does not give both players the same
  quality. Every model's view responds to its own player's actions and to
  nothing else.

## Provenance

| Item | Value |
| --- | --- |
| Generations | `mc_multiplayer_v2_generations/<model>/step_0001200_*`, the same clips as the paper's Tables 2 and 3 |
| Ground truth | `mc_multiplayer_v2_eval_new_sneak_combined_simplified_naming/<eval>/test/` |
| Models | all 7 in the ablation set |
| Camera evals | rotation, turnToLook, turnToLookOpposite, bothLookAway_long, oneLooksAway_long, structure |
| Key evals | translation, structure |
| Cache | 3,904 `.npz`, 0 failures, `action_following/cache/`, derived and not committed |

**The camera table uses only the 6 eval sets that all 7 models cover.**
`no_player_attn_sf` has no clips for the short `bothLookAway` and `oneLooksAway`
variants. Including them scores it on 339 turn events where every other model
gets 436, which is not a comparison.

## Method in one paragraph

Track points between consecutive frames with pyramidal Lucas-Kanade and a
forward-backward check. Convert them to viewing rays with the known intrinsics. A
camera turn adds the same amount to the azimuth of every ray whatever its depth,
so the median per-point azimuth change is an unbiased estimate of yaw, and
elevation gives pitch the same way. That estimator has no trained parameter, so
it is scored on real Minecraft video of the same episodes first; that row is the
ceiling in every table. Key presses cannot be read that way, so they follow the
literature: a small inverse dynamics model trained on ground-truth video only,
tested on held-out episodes.

# Part 1 — Camera, per player

`event%` is the fraction of commanded turn events rendered in the commanded
direction, with 3 frames of timing slack. `ev ratio` is the median rendered
angle over commanded angle. `frame%` is the same decision judged frame by frame
with no slack. `still%` is the fraction of no-command frames where the view does
not move, so `100 - still%` is the hallucinated-motion rate. `RotErr` is the mean
absolute difference between total rendered and total commanded yaw per episode.

### Alpha — 339 turn events, 3,512 commanded frames

| Model | event% | ev ratio | frame% | still% | gain | RotErr deg |
|---|---|---|---|---|---|---|
| **ground truth (ceiling)** | **97.6** | **1.002** | **81.5** | **99.3** | **0.863** | **12.7** |
| `flagship` | 89.7 | 0.680 | 71.5 | **99.5** | 0.614 | 36.1 |
| `no_player_attn_sf` | **97.1** | 1.042 | **92.7** | 92.7 | **0.937** | 42.6 |
| `concat_c` | 93.2 | 1.007 | 87.3 | 85.4 | 0.817 | 79.1 |
| `from_scratch` | 90.6 | 0.984 | 80.1 | 97.4 | 0.752 | 25.5 |
| `causvid_regression` | 83.2 | 0.534 | 64.0 | 89.2 | 0.534 | 113.9 |
| `causvid_dmd` | 35.1 | 0.052 | 35.8 | 99.5 | 0.275 | 89.2 |
| `no_kv_cache_backprop` | 90.6 | 0.763 | 76.1 | **99.6** | 0.683 | **24.4** |

### Bravo — 302 turn events, 3,219 commanded frames

| Model | event% | ev ratio | frame% | still% | gain | RotErr deg |
|---|---|---|---|---|---|---|
| **ground truth (ceiling)** | **96.7** | **1.004** | **81.2** | **99.4** | **0.872** | **11.0** |
| `flagship` | 88.7 | 0.717 | 73.5 | 99.0 | 0.671 | 33.0 |
| `no_player_attn_sf` | **96.7** | 1.053 | **92.5** | 96.4 | **0.928** | 23.3 |
| `concat_c` | 74.8 | 0.585 | 54.3 | 99.1 | 0.479 | **20.8** |
| `from_scratch` | 92.1 | 0.962 | 77.3 | 98.8 | 0.745 | 18.0 |
| `causvid_regression` | 83.8 | 0.563 | 71.2 | 64.1 | 0.620 | 256.9 |
| `causvid_dmd` | 47.0 | 0.205 | 42.1 | 99.3 | 0.387 | 79.5 |
| `no_kv_cache_backprop` | 90.1 | 0.772 | 75.3 | **99.6** | 0.711 | 24.1 |

## How to read these

**`event%` and `still%` must be read together.** They trade off. A model that
renders more camera motion scores higher on the first and lower on the second.
`no_player_attn_sf` has the best turn accuracy of any model (97.1 / 96.7,
matching ground truth) and the worst stillness on Alpha (92.7). `causvid_dmd` is
the mirror image: near-perfect stillness (99.5 / 99.3) because it barely moves at
all, and 35.1 / 47.0 on turns. Neither is good control. `flagship` and
`no_kv_cache_backprop` are the only models above 88 on turns *and* above 99 on
stillness for both players.

**`RotErr` is the number that combines them,** and it is where the two failure
modes show their cost. `causvid_regression` reaches 83.8% turn accuracy on Bravo
and still accumulates 256.9 degrees of error over a clip, because 36% of its
still frames drift.

**`frame%` has a low ceiling — 81.5, not 100.** Even on real video the strict
per-frame comparison misses about a fifth of commanded frames, because the
renderer and the action log are not perfectly in step and the estimator needs
enough displacement to measure. This is exactly why `event%` is the headline.
Quote `frame%` only alongside its ground-truth row.

**`ev ratio` above 1.0 on the ground-truth row is estimator bias, not signal.**
At the assumed 77-degree vertical fov the estimator sits ~0.2% high pooled over
these eval sets; a sweep of assumed fov puts the best fit near 80 degrees. The
bias is small and identical across models, so ratios compare cleanly with each
other, but a generated ratio should be read against the ground-truth row rather
than against 1.0.

## Per eval set, `flagship`, both players pooled

| Dataset | event% gt → gen | ev ratio gt → gen | still% gt → gen | RotErr gt → gen |
|---|---|---|---|---|
| `rotationEval` | 100.0 → 96.9 | 1.039 → 0.691 | 100.0 → 99.6 | 2.6 → 12.6 |
| `turnToLookEval` | 100.0 → 100.0 | 1.097 → 1.013 | 99.7 → 99.6 | 10.3 → 21.9 |
| `turnToLookOppositeEval` | 100.0 → 100.0 | 1.011 → 0.721 | 99.7 → 99.9 | 2.9 → 24.1 |
| `bothLookAwayEval_long` | 99.2 → 89.9 | 0.879 → 0.578 | 98.3 → 99.3 | 22.0 → 28.1 |
| `oneLooksAwayEval_long` | 100.0 → 92.2 | 0.887 → 0.639 | 99.3 → 98.2 | 7.9 → 23.5 |
| `structureEval` | 94.1 → 82.6 | 0.972 → 0.661 | 99.1 → 98.7 | 25.5 → 97.0 |

`structureEval` is the hardest: it is the only set where the bot turns, pitches
and walks at once, and it is where `flagship`'s RotErr blows out to 97 degrees.

## Pitch

The bots drive pitch in `structureEval` only, on 324 Alpha frames against 3,512
for yaw. The same estimator recovers it, but less well: the ground-truth ceiling
is 96.4% event accuracy with a gain of 0.638, against 0.863 for yaw. Every model
is far below that ceiling — `no_kv_cache_backprop` 68.5, `no_player_attn_sf`
67.6, `flagship` 65.8, `concat_c` 55.9, `from_scratch` and `causvid_regression`
52.3, `causvid_dmd` 47.7.

Run it with `report_camera.py --axis pitch --datasets structureEval`. Do not pool
pitch into the yaw tables; the two axes have different ceilings.

# Part 2 — Cross-player specificity

Each player's view held against the **other** player's camera commands, on the
two eval sets where only one bot turns.

| Model | own gain | cross gain | own r | cross r |
|---|---|---|---|---|
| ground truth | 0.818 | 0.000 | 0.832 | −0.001 |
| `flagship` | 0.650 | 0.000 | 0.800 | −0.003 |
| `no_player_attn_sf` | 0.908 | −0.001 | 0.908 | +0.002 |
| `concat_c` | 0.546 | 0.000 | 0.719 | +0.005 |
| `from_scratch` | 0.745 | 0.002 | 0.804 | −0.000 |
| `causvid_regression` | 0.630 | 0.000 | 0.575 | −0.017 |
| `causvid_dmd` | 0.402 | 0.000 | 0.474 | −0.005 |
| `no_kv_cache_backprop` | 0.687 | 0.000 | 0.845 | −0.002 |

**The restriction to single-actor eval sets is what makes this table mean
anything.** Pooled over all camera evals the two players' own commands correlate
at r = +0.297, because `turnToLook` turns both bots at once. A view that followed
only its own command would then show a spurious cross-correlation of about the
same size — and it does: the pooled table gives ground truth a cross r of +0.175.
On `rotationEval` and `oneLooksAwayEval_long` the command streams correlate at
r = −0.002, and every cross column collapses to zero.

**This is a sanity result, not a discriminator.** Every model passes, including
`no_player_attn_sf`, which renders the two players independently and so cannot
leak between them by construction. It rules out a failure mode rather than
ranking the architectures.

# Part 3 — Keyboard

Multinomial logistic regression over the flow summary of a 5-frame window,
trained on ground-truth video only, episodes split so no episode appears in both
halves. 21,504 training frames; classes `none` 19,030, `forward` 1,408, `back`
402, `left` 329, `right` 335.

**IDM on held-out ground-truth video: 95.9% accuracy, 96.9% balanced.** Per class:
`none` 95.9, `forward` 93.5, `back` 100.0, `left` 96.3, `right` 98.8.

Balanced accuracy is the mean per-class recall, so the `none` class cannot carry
it. Chance is 20.0.

| Model | Alpha bal% | Bravo bal% | Alpha acc% | Bravo acc% | pred none% |
|---|---|---|---|---|---|
| **ground truth (ceiling)** | **97.4** | **96.4** | 96.1 | 95.8 | 85.1 |
| `flagship` | 75.2 | 73.0 | 93.5 | 93.1 | 87.2 |
| `no_player_attn_sf` | 74.5 | 74.0 | 90.8 | 90.7 | 84.0 |
| `concat_c` | 81.7 | 61.8 | 92.5 | 88.6 | 84.7 |
| `from_scratch` | 75.4 | 78.8 | 89.8 | 90.3 | 82.7 |
| `causvid_regression` | 70.6 | 56.2 | 91.5 | 73.0 | 77.0 |
| `causvid_dmd` | 64.8 | 59.8 | 90.7 | 89.4 | 85.6 |
| `no_kv_cache_backprop` | **88.3** | **88.6** | 94.7 | 94.0 | 85.4 |

On `translationEval` alone, where the bot presses one movement key at a time and
never turns, the ceiling rises to 99.3 balanced and every model rises with it
(`flagship` 82.5 / 81.4, `no_kv_cache_backprop` 94.4 / 95.5). The pooled table
above includes `structureEval`, where walking, turning and pitching happen at
once, and that is what costs everyone roughly ten points.

# Part 4 — Per-player gap

Absolute difference between the two players, which is the number a per-player
breakdown exists to produce.

| Model | camera event% | camera still% | keyboard bal% |
|---|---|---|---|
| `no_player_attn_sf` | 0.4 | 3.7 | 0.5 |
| `no_kv_cache_backprop` | 0.5 | 0.0 | 0.3 |
| `causvid_regression` | 0.6 | 25.1 | 14.4 |
| `flagship` | 1.0 | 0.5 | 2.2 |
| `from_scratch` | 1.5 | 1.4 | 3.4 |
| `causvid_dmd` | 11.9 | 0.2 | 5.0 |
| `concat_c` | **18.4** | **13.7** | **19.9** |

`concat_c` is the only model that is large on all three. Note that a small gap is
not by itself a virtue — `no_player_attn_sf` is symmetric because it renders each
player in isolation. The finding is about `concat_c`: joining the players by
frame concatenation makes one of them worse.

# Relation to the paper's VLM axes

Across the 7 models, camera `event%` and the paper's Movement axis rank almost
independently (Spearman rho = +0.04, n = 7). Keyboard balanced accuracy tracks it
a little better (+0.46, p = 0.29, not significant at this n).

That is the expected result, not a problem. Movement asks whether a player ended
up correctly placed *relative to the other player*, which is a multiplayer
semantic question. The metrics here ask whether one view rendered the low-level
control it was given. `no_player_attn_sf` makes the gap concrete: Movement 12.5,
the second worst in the paper, and the best camera turn accuracy of any model.
It follows its own actions precisely and gets the other player wrong.

This is the same lesson as the artifact study in `human-eval/RESULTS.md`, where
`concat_c` posts the best Movement score of all 7 models and is visually
unusable. No single axis covers control, image quality and multiplayer semantics.

# Caveats

1. **The keyboard IDM is trained on real video and applied to generated video.**
   Generated frames are blurrier, and some of the gap below the ground-truth row
   is that domain shift rather than control failure. `pred none%` is the column
   to watch: it is roughly flat across models (77–88 against a true still rate
   shared by all), so the gap is not simply the estimator giving up. Matrix-Game,
   Oasis and WorldMem have the same exposure.
2. **`event%` uses 3 frames of slack.** Generated clips often start a turn a
   frame late. Without slack the strict `frame%` column applies, and its
   ground-truth ceiling is 81.5, not 100.
3. **Turn events are not independent within an episode.** No confidence interval
   is quoted for that reason. The Alpha and Bravo columns are, however,
   independent measurements of the same models, and they agree closely for every
   model except `concat_c` and `causvid_regression`.
4. **Sneak, jump, attack, use and place are not scored.** They occur too rarely
   in the eval sets, or produce no camera-frame motion to measure.
5. **`ev ratio` is a median over events**, so a model that renders one turn at
   triple size and another at zero does not average out to 1.0.
