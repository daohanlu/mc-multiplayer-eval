# Action-following evaluation

Reviewer nAFu asked why we do not report Matrix-Game's action-controllability
metric, computed per player. This directory is the answer. It measures, for each
player separately, whether the rendered view actually does what that player's
action stream told it to do.

## What is measured, and how

Matrix-Game, Oasis and WorldMem all score action controllability the same way:
run an inverse dynamics model over the generated video to read the action back
out, then compare it with the action that was sent. Matrix-Game's model is
trained on 1,962 hours of Minecraft and reports 90.6% keyboard accuracy and an
R-squared of 0.97 on mouse regression. We do not have that model, so we build
the two halves separately and hold each to its own check.

**Camera, analytically.** A camera turn moves every viewing ray by the same
angle, whatever the depth of what the ray hits. So a rotation can be recovered
from the video with no training at all: track points between consecutive frames,
convert them to viewing rays with the known intrinsics, and take the median
change in azimuth and elevation. A least-squares rotation is then refit on the
points those medians agree with, which also recovers roll.

The commanded rate is exactly 0.15 radians per frame, which is 3 radians per
second at 20 ticks per second, so the estimate can be scored directly. On real
Minecraft video of the same episodes the estimator reaches a gain of about 1.0
and its noise on frames with no camera command is under 0.001 radians per frame,
which is 150 times smaller than the commanded rate. Every table carries that
ground-truth row, so estimator error and model error stay separate.

**Keys, with a small learned model.** A key press cannot be read off analytically.
On an open plain a sideways step and a small camera turn produce nearly the same
flow field, and the parallax that separates them lives in the few metres of
ground at the bottom of the frame. So this half follows the literature: a
multinomial logistic regression over the flow summary of a five-frame window,
trained on ground-truth video only, with the test episodes held out. Its accuracy
on held-out ground-truth video is the ceiling for every generated number.

**Per player, and across players.** Every number is reported for Alpha and Bravo
separately, which is what the reviewer asked for. One further table holds each
player's view against the *other* player's command stream. A model that routes
actions to the right player scores near zero there. A model that has collapsed
the two players into one scene would not.

## Why not the rest of GameWorld Score

The other pillars of GameWorld Score are visual quality (MUSIQ, LAION aesthetic),
temporal quality (CLIP feature similarity, motion smoothness) and physical rule
understanding (DROID-SLAM reprojection, symmetric-motion consistency). The paper
already reports FID and FVD for the first, and the 5-annotator artifact study in
`human-eval/` for visible corruption. The Consistency axis is the
symmetric-motion test: `turnToLookEval` and `bothLookAwayEval` turn a player away
and back, and score whether the scene came back the same. Action controllability
was the gap, and it is the gap this directory fills.

## Files

| File | Content |
| --- | --- |
| `af_data.py` | Joins a generated clip to its ground-truth action stream. Quadrant layout, frame alignment, intrinsics. |
| `af_flow.py` | Point tracking, the rotation and translation estimators, and the flow summary. |
| `extract_motion.py` | Runs the estimator over every clip and caches one `.npz` per (dataset, model, episode, player). |
| `calibrate_fov.py` | Sweeps the assumed field of view on ground-truth video and reports the gain at each one. |
| `report_camera.py` | The per-player camera tables, including cross-player specificity. |
| `report_keys.py` | The learned inverse dynamics model and the per-player keyboard tables. |
| `RESULTS.md` | The numbers, and how to read them. |

## Reproduce

```bash
python3 action_following/extract_motion.py --workers 10   # about 70 minutes
python3 action_following/report_camera.py
python3 action_following/report_camera.py --axis pitch --datasets structureEval
python3 action_following/report_keys.py
```

`cache/` is derived and is not committed.

## Two things to know before changing anything

**Dense flow does not work here.** Farneback returns near-zero motion on this
material. A Minecraft plain is mostly sky and repeating grass, and a camera turn
moves the view about 19 pixels per frame at the width used here, which is outside
what the dense solver recovers on periodic texture. Pyramidal Lucas-Kanade with
six levels and a forward-backward check does recover it. Phase correlation also
works on the strong frames but locks onto the wrong peak when the grass aliases.

**The renderer is one frame behind the action log.** The image at ground-truth
index `k` shows the pose that the action at index `k` produced, so the motion
between images `k` and `k+1` is the action recorded at `k+1`... which, once the
generated clip's own offset is applied, means generated frame pair `i` carries
the action at `frame1 + 1 + i`. This was established by lining the recorded
`yaw` field up against the measured pixel shift, not assumed. Get it wrong by one
frame and every correlation collapses.
