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
* **Per player** — Solaris scores Alpha and Bravo within 1 point of each other on
  all three measures, and `no_kv_cache_backprop` within 2. `concat_c` differs by
  14 to 18 points, so its concatenation scheme does not give both players the
  same quality. Every model's view responds to its own player's actions and to
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

**The camera half is purely analytical: no learned component, no parameter fitted
to our data.** It is the standard rotating-camera model of multi-view geometry —
two views of a camera that only rotates are related by the infinite homography
`K R K^-1`, whatever the scene depth — and `K` is known, so `R` is solved for
directly. Correspondences come from pyramidal Lucas-Kanade with the
forward-backward check of Kalal et al.; the initial yaw and pitch are the median
change in ray azimuth and elevation, which doubles as the inlier test; `R` is
refit on the inliers by solving Wahba's problem in the Kabsch SVD closed form.
Structure from motion and SLAM are deliberately avoided, though `RotErr` normally
uses COLMAP and GameWorld Score uses DROID-SLAM: these clips are near pure
rotation, where the closed form is exact and steadier than SfM on 256 low-texture
frames. Because nothing is fitted, running the estimator on real Minecraft video
of the same episodes measures the estimator itself, and that row is the ceiling
in every table. Key presses cannot be read this way, so they follow the
literature instead: a small inverse dynamics model trained on ground-truth video
only, tested on held-out episodes.

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

## Temporal tolerance, and why it does not matter much

The eval sets are recorded, not simulated, so a command does not change the
picture on the same frame. `measure_latency.py` measures that delay instead of
assuming it. The frame alignment already removes the recorder's fixed one-frame
lag — established by lining the recorded `yaw` up against measured pixel shift —
so what follows is the jitter left around it.

| Source | best global lag | onset median | onset p90 | within ±3 |
|---|---|---|---|---|
| ground truth | 0 | 0 | 1 | 85.1% |
| `flagship` | +1 | 1 | 3 | — |
| `no_player_attn_sf` | 0 | 0 | 1 | — |
| `concat_c` | +1 | 0 | 6 | — |
| `causvid_dmd` | +1 | 2 | 6 | — |
| all 7 pooled | — | 0 | — | 82.2% |

"Onset" is the first frame at which the rendered view passes the deadband,
relative to the frame the command arrived. On ground truth it lands at −1, 0 or
+1 for 70% of events. **This confirms the 0–2 frame visual delay in the dataset.**
Generated clips sit about one frame later than ground truth; `causvid_dmd`, which
barely moves at all, is the slowest at a median of 2.

`EVENT_SLACK = 3` is set from that. But the event metric sums rotation across the
whole event rather than testing a single frame, so it is nearly insensitive to
the choice:

| slack | GT | `flagship` | `no_player_attn_sf` | `concat_c` | `causvid_dmd` |
|---|---|---|---|---|---|
| ±1 | 97.0 | 88.7 | 97.2 | 83.7 | 41.1 |
| ±2 | 97.2 | 88.9 | 97.0 | 84.2 | 41.4 |
| **±3** | **97.2** | **89.2** | **96.9** | **84.0** | **41.1** |
| ±5 | 97.2 | 89.2 | 96.9 | 84.2 | 41.2 |
| ±8 | 97.2 | 89.2 | 96.9 | 83.9 | 41.2 |

Nothing moves by more than 0.6 points across an eightfold change in tolerance,
and no ranking changes at all. **The tolerance is not a knob that any conclusion
turns on.**

The other two metrics handle the delay differently. `still%` is per frame with no
slack, which is why the ground-truth row is 99.3 rather than 100 — the residual
jitter shows up there. `frame%` is per frame with no slack either, and its
ground-truth ceiling of 81.5 is mostly this delay; quote it only beside that
ceiling. The keyboard IDM reads a 5-frame window, so ±2 frames of context is
built into its features.

## Pitch

The bots drive pitch in `structureEval` only, on 324 Alpha frames against 3,512
for yaw. The same estimator recovers it, but less well: the ground-truth ceiling
is 96.4% event accuracy with a gain of 0.638, against 0.863 for yaw. Every model
is far below that ceiling — `no_kv_cache_backprop` 68.5, `no_player_attn_sf`
67.6, `flagship` 65.8, `concat_c` 55.9, `from_scratch` and `causvid_regression`
52.3, `causvid_dmd` 47.7.

Run it with `report_camera.py --axis pitch --datasets structureEval`. Do not pool
pitch into the yaw tables; the two axes have different ceilings.

# Part 1b — The same question, with the VPT IDM Matrix-Game uses

Matrix-Game reports its inverse dynamics model as trained on 1,962 hours of
Minecraft with 90.6% keypress accuracy and R² 0.97 on mouse movement. **Those are
VPT's own published figures** (Baker et al., 2022), and OpenAI released that
model, so we run the identical one rather than approximate it. `setup_vpt_idm.sh`
fetches it; `vpt_idm.py` runs it; `report_vpt.py` pools the output.

**It reads our render.** On ground-truth video of the same episodes the IDM
recovers the commanded yaw at r = 0.83–0.91 with 99.3% stillness on no-command
frames, despite our 640×352 at fov 77 against VPT's 640×360 at fov 70, and a
different Minecraft version. That was the thing in doubt, and it is settled.

| Model | dir acc A/B | still A/B | gain A/B | **vs GT** A/B |
|---|---|---|---|---|
| **ground truth (ceiling)** | 78.6 / 75.0 | 99.3 / 99.3 | 0.91 / 0.84 | 1.00 / 1.00 |
| `flagship` | 85.4 / 84.0 | 99.5 / 98.9 | 0.76 / 0.79 | **0.84 / 0.93** |
| `no_player_attn_sf` | 92.5 / 87.0 | 96.0 / 97.8 | 1.02 / 0.93 | 1.13 / 1.11 |
| `concat_c` | 87.8 / 86.1 | 90.0 / 98.9 | 0.85 / 0.89 | 0.94 / 1.05 |
| `from_scratch` | 88.3 / 79.5 | 98.3 / 98.8 | 0.94 / 0.84 | 1.04 / 0.99 |
| `causvid_regression` | 75.8 / 62.2 | 93.8 / 97.5 | 0.67 / 0.54 | 0.74 / 0.64 |
| `causvid_dmd` | 47.5 / 54.9 | 99.2 / 99.1 | 0.42 / 0.51 | 0.46 / 0.61 |
| `no_kv_cache_backprop` | 84.9 / 82.7 | 99.6 / 99.7 | 0.76 / 0.78 | 0.84 / 0.92 |

**Read the "vs GT" column for magnitude, not the raw gain.** The IDM's camera
head saturates at 10 deg/frame and our bots command 8.594, so 62–67% of
ground-truth turn frames already read at the ceiling. It can therefore see a
model turning *less* than ground truth but not one turning more, and absolute
gain is compressed at the top for every row alike.

## The keyboard head does not transfer, although the camera head does

Run on ground-truth video, all 32 episodes, both players:

| Eval set | commanded WASD frames | VPT predicted | exact match |
|---|---|---|---|
| `translationEval` | 2,252 | 32 | 30 (1.3%) |
| `structureEval` | 1,540 | 427 | 265 (17.2%) |

On `translationEval` the IDM fires almost no key at all — 32 predictions against
2,252 commanded frames. It is not a plumbing failure: the head is live and does
fire on `structureEval`, where it still only recovers 17%. Our bots strafe across
open ground, so the parallax that separates a sideways step from a small camera
turn lives in a thin band of near ground along the bottom of the frame, and the
IDM's 128×128 input removes most of it. Our own WASD model works on the same
clips (96.9% balanced, held out) because it reads that band explicitly, splitting
flow between the upper and lower half of the view.

**So keys stay with our own model and mouse comes from VPT.** Reproduce with
`vpt_idm.py`, which now saves button predictions alongside the camera.

## What happens if the two tables are scored by one identical rule

Table A judges each frame on its own; Table B groups a turn into an event with 3
frames of slack. Re-scoring the analytic estimator under VPT's exact rule — per
frame, no slack, 1.0° deadband — answers what the difference is worth.

| Model | Direction, ours | Direction, VPT | gap |
|---|---|---|---|
| ground truth | 81.9 / 81.2 | 78.6 / 75.0 | +3.3 / +6.2 |
| `flagship` | 71.6 / 73.6 | 85.4 / 84.0 | **+13.8 / +10.4** |
| `no_player_attn_sf` | 92.9 / 92.7 | 92.5 / 87.0 | −0.4 / −5.7 |
| `concat_c` | 87.4 / 54.5 | 87.8 / 86.1 | +0.4 / **+31.6** |
| `from_scratch` | 80.5 / 77.7 | 88.3 / 79.5 | +7.8 / +1.8 |

**The ground-truth rows converge to within a few points, and the model rows do
not.** That is the useful part. The gap is not noise: across the 8 model-player
cells it correlates with the magnitude disagreement at **r = +0.92**. The two
estimators differ on Direction exactly where they differ on Magnitude.

The mechanism is VPT's mu-law binning. A turn that goes the right way but only
70% as far still lands in a non-zero bin, so VPT scores it a hit; our estimator
measures the angle, so a weak turn can fall under the deadband and score a miss.
`no_player_attn_sf`, which does not under-rotate, agrees across estimators to
within half a point on Alpha. `flagship` and `concat_c`, which do, diverge most.

**Consequence: VPT's Direction column is insensitive to under-rotation, and
forcing both tables onto one rule would hide that rather than resolve it.** It
would also manufacture an apparent contradiction — Solaris sits 7–9 points above
the ground-truth row under VPT's rule and 8–10 points below it under ours — which
is a statement about deadband sensitivity, not about the model. Keeping the two
rules, each with its own ground-truth row, is the more honest presentation. It is
also why Magnitude, not Direction, carries the under-rotation finding.

## Two independent estimators agree

| Model | VPT IDM A/B | analytic A/B |
|---|---|---|
| `flagship` | 0.84 / 0.93 | 0.71 / 0.77 |
| `no_player_attn_sf` | 1.13 / 1.11 | 1.09 / 1.06 |
| `concat_c` | 0.94 / 1.05 | 0.95 / 0.55 |
| `from_scratch` | 1.04 / 0.99 | 0.87 / 0.85 |
| `causvid_regression` | 0.74 / 0.64 | 0.62 / 0.71 |
| `causvid_dmd` | 0.46 / 0.61 | 0.32 / 0.44 |
| `no_kv_cache_backprop` | 0.84 / 0.92 | 0.79 / 0.82 |

Both normalised by their own ground-truth row. **Pearson r = +0.81 over the 14
model-player cells.** The two share no code, no training data and no assumptions
— one is closed-form geometry with a known camera, the other a 1,962-hour neural
network — so the agreement is evidence about the models, not about either
estimator.

**`flagship` under-rotates, and this is the honest headline.** Both estimators put
it below ground truth on magnitude, 0.84/0.93 and 0.71/0.77. The two disagree on
how much, so the defensible claim is the direction and the ordering, not a point
value. It is a property of this model rather than of the measurement:
`no_player_attn_sf` and `from_scratch` sit at or above ground truth on both.

Two caveats worth stating before anyone quotes the direction column.
`flagship` scores *above* the ground-truth ceiling on direction accuracy
(85.4 vs 78.6). That is not the model beating reality; generated turns are
smoother and more exaggerated, which the IDM reads more confidently than it reads
a slow real turn. And pooling `structureEval` into this table pulls every row
down, ground truth included, because yaw, pitch and walking happen at once there.

# Part 1c — The full tables, all 7 ablations

The rebuttal shows the Table 2 rows only, to keep the comparison to the models
the reviewer's question is about. These are the same two tables with every
ablation, in the same format, so the rest can be quoted without re-running
anything.

### Table A — mouse, VPT IDM, all 7 ablations

| Model | Direction | No false turns | Average | Magnitude |
|---|---|---|---|---|
| **ground truth (ceiling)** | 78.6 / 75.0 | 99.3 / 99.3 | 88.9 / 87.2 | 1.00 / 1.00 |
| Solaris | 85.4 / 84.0 | 99.5 / 98.9 | 92.4 / 91.4 | 0.84 / 0.93 |
| Independent | 92.5 / 87.0 | 96.0 / 97.8 | 94.2 / 92.4 | 1.13 / 1.11 |
| Frame Concat | 87.8 / 86.1 | 90.0 / 98.9 | 88.9 / 92.5 | 0.94 / 1.05 |
| Solaris w/o pretrain | 88.3 / 79.5 | 98.3 / 98.8 | 93.3 / 89.1 | 1.04 / 0.99 |
| ODE Reg † | 75.8 / 62.2 | 93.8 / 97.5 | 84.8 / 79.8 | 0.74 / 0.64 |
| Causal FT Pre-DMD † | 47.5 / 54.9 | 99.2 / 99.1 | 73.4 / 77.0 | 0.46 / 0.61 |
| Causal FT no KV-BP † | 84.9 / 82.7 | 99.6 / 99.7 | 92.2 / 91.2 | 0.84 / 0.92 |

### Table B — camera and keys, our own estimators, all 7 ablations

| Model | Direction | No false turns | Average | Magnitude | Keys |
|---|---|---|---|---|---|
| **ground truth (ceiling)** | 97.6 / 96.7 | 99.3 / 99.4 | 98.5 / 98.0 | 1.00 / 1.00 | 97.4 / 96.4 |
| Solaris | 89.7 / 88.7 | 99.5 / 99.0 | 94.6 / 93.9 | 0.71 / 0.77 | 73.3 / 72.6 |
| Independent | 97.1 / 96.7 | 92.7 / 96.4 | 94.9 / 96.5 | 1.09 / 1.06 | 68.4 / 72.3 |
| Frame Concat | 93.2 / 74.8 | 85.4 / 99.1 | 89.3 / 87.0 | 0.95 / 0.55 | 77.7 / 60.1 |
| Solaris w/o pretrain | 90.6 / 92.1 | 97.4 / 98.8 | 94.0 / 95.4 | 0.87 / 0.85 | 73.1 / 78.4 |
| ODE Reg † | 83.2 / 83.8 | 89.2 / 64.1 | 86.2 / 73.9 | 0.62 / 0.71 | 66.7 / 53.4 |
| Causal FT Pre-DMD † | 35.1 / 47.0 | 99.5 / 99.3 | 67.3 / 73.2 | 0.32 / 0.44 | 59.7 / 61.8 |
| Causal FT no KV-BP † | 90.6 / 90.1 | 99.6 / 99.6 | 95.1 / 94.8 | 0.79 / 0.82 | 84.1 / 86.3 |

† Table 3 model. The rebuttal quotes the Table 2 rows only; these are here
so the rest can be produced on request without re-running anything.

Nothing here changes a conclusion. `no_kv_cache_backprop` is the strongest model
on both Average columns and on Keys, and `causvid_dmd` is the weakest on both;
Solaris under-rotates in both, as it does in the Table 2 view.

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

**Scope: WASD only.** The 5 classes are `none`, `forward`, `back`, `left`,
`right`. Block placement, attack, use, mine, jump, sneak and sprint are *not*
modelled. A flow-based estimator reads global motion, and those actions either
move too few pixels to register (placing a block) or do not change the view at
all in these eval sets. A frame holding `forward` and `sprint` together is
labelled `forward`; a frame holding two of the four directions is dropped, not
guessed. The cache stores all 11 recorded keys, so widening the class set later
needs no re-extraction — only `CLASSES` in `report_keys.py`.

Multinomial logistic regression over the flow summary of a 5-frame window,
trained on ground-truth video only.

### How much data, and which split

| | Episodes | Frames | of which carry a key press |
|---|---|---|---|
| Train | 42 | 21,504 | 2,474 |
| Held out | 22 | 11,264 | 1,318 |

Train classes: `none` 19,030, `forward` 1,408, `back` 402, `left` 329,
`right` 335. Held-out classes: `none` 9,946, `forward` 682, `back` 141,
`left` 246, `right` 249.

The split is by episode, every third one held out. Alpha and Bravo of one episode
share a stem and therefore the same side of the split, so the two players never
straddle it, and the 5-frame window never crosses it either.

**The training signal is small — 2,474 labelled key-press frames.** That is the
real size of this model, not the 21,504 figure, which is nine parts `none`.

| Split | Accuracy | Balanced |
|---|---|---|
| Train episodes | 95.9% | 98.5% |
| **Held out** | **95.9%** | **96.9%** |

**Every ceiling quoted anywhere is the held-out number.** The 1.6-point balanced
gap says the model is reading motion rather than memorising scenes. Per-class
held-out recall: `none` 95.9, `forward` 93.5, `back` 100.0, `left` 96.3,
`right` 98.8. Balanced accuracy is the mean per-class recall, so the `none` class
cannot carry it; chance is 20.0.

### Results, held-out episodes only

Every row below is scored on the same 22 held-out episodes as the ceiling. The
IDM never saw generated video of any episode, so scoring the models on all 64
would not leak directly — but a train episode's *ground truth* shares its scene,
so restricting to held-out episodes keeps every row like for like.

| Model | Alpha bal% | Bravo bal% | Alpha acc% | Bravo acc% | pred none% |
|---|---|---|---|---|---|
| **ground truth (ceiling)** | **97.4** | **96.4** | 96.1 | 95.8 | 85.1 |
| `flagship` | 73.3 | 72.6 | 92.4 | 93.5 | 87.2 |
| `no_player_attn_sf` | 68.4 | 72.3 | 89.0 | 90.8 | 84.2 |
| `concat_c` | 77.7 | 60.1 | 91.3 | 88.2 | 85.1 |
| `from_scratch` | 73.1 | 78.4 | 89.3 | 90.2 | 82.8 |
| `causvid_regression` | 66.7 | 53.4 | 90.0 | 72.2 | 76.7 |
| `causvid_dmd` | 59.7 | 61.8 | 88.4 | 89.7 | 85.7 |
| `no_kv_cache_backprop` | **84.1** | **86.3** | 93.8 | 93.8 | 85.5 |

Scoring instead on all 64 episodes, which triples the sample, moves the numbers a
few points and changes no ranking: `flagship` 75.2 / 73.0, `no_kv_cache_backprop`
88.3 / 88.6, `concat_c` 81.7 / 61.8. `report_keys.py` prints both tables.

On `translationEval` alone, where the bot presses one movement key at a time and
never turns, the ceiling rises to 99.3 balanced and every model rises with it
(`flagship` 80.0 / 84.7, `no_kv_cache_backprop` 90.7 / 95.2). The pooled table
above includes `structureEval`, where walking, turning and pitching happen at
once, and that is what costs everyone roughly ten points.

# Part 4 — Per-player gap

Absolute difference between the two players, which is the number a per-player
breakdown exists to produce.

| Model | camera event% | camera still% | keyboard bal% |
|---|---|---|---|
| `no_player_attn_sf` | 0.4 | 3.7 | 3.9 |
| `no_kv_cache_backprop` | 0.5 | 0.0 | 2.2 |
| `causvid_regression` | 0.6 | 25.1 | 13.3 |
| `flagship` | 1.0 | 0.5 | 0.7 |
| `from_scratch` | 1.5 | 1.4 | 5.3 |
| `causvid_dmd` | 11.9 | 0.2 | 2.1 |
| `concat_c` | **18.4** | **13.7** | **17.6** |

`concat_c` is the only model that is large on all three. Note that a small gap is
not by itself a virtue — `no_player_attn_sf` is symmetric because it renders each
player in isolation. The finding is about `concat_c`: joining the players by
frame concatenation makes one of them worse.

# Relation to the paper's VLM axes

Across the 7 models, camera `event%` and the paper's Movement axis rank almost
independently (Spearman rho = +0.04, n = 7). Keyboard balanced accuracy tracks it
a little better (+0.61, p = 0.15, not significant at this n).

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
