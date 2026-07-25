# Human evaluation

Two blind annotation tasks, served as static pages with a small standard-library
Python server that persists progress to disk.

| Task | What the annotator does | Items |
|---|---|---|
| **Consistency** | Sees two screenshots, answers *same scenery* / *different scenery* | 256 |
| **Artifacts** | Watches a clip, picks one of four artifact labels | 63 |

## Quick start

```bash
python human-eval/build_human_eval.py     # once: extract frames, copy videos
python human-eval/serve.py                # http://localhost:8080
```

To let other machines connect: `python human-eval/serve.py --host 0.0.0.0 --port 8080`.

When answers are in:

```bash
python human-eval/score_human_eval.py     # reads everything in responses/
```

## Task 1 — Consistency

The annotator sees **exactly the screenshot pairs the VLM saw** for the two
Table 3 rows the paper reports at **56.8 ± 2.9** (`flagship`) and
**34.9 ± 1.5** (`causvid_regression`).

Frames are produced by calling `run_eval.extract_query_frames` — the same
"single source of truth" helper the paper run used — with
`LATE_EPISODE_QUERY_STRICT=1`, the same generated videos
(`step_0001200_..._turn_to_look{,_opposite}_max_speed`), and the same GT dataset
(`mc_multiplayer_v2_eval_new_sneak_combined`). The build was verified against
`results_json_late_episode_strict/`: all 256 items match the recorded
`episode`, `instance`, `query_type`, `alpha_frame`, `bravo_frame`, `frame1` and
`expected` — zero mismatches.

```
2 models x 2 evals (same-side + opposite-sides) x 32 episodes x 2 timestamps = 256
```

The two timestamps per episode are the strict toggle's original turn-end frame
and its late-horizon duplicate. Both are included because the paper's
episode-level accuracy requires the model to be right at *both*, and
`score_human_eval.py` applies the same AND-semantics to the human answers.

**Calibration.** Before starting, the annotator is shown two worked examples —
one genuinely same-scenery, one genuinely different — drawn from the
**ground-truth** (real, non-generated) episode videos, so they cannot leak
anything about the scored items. The "different" example explicitly warns that a
shared biome, sky and HUD are *not* evidence of the same scenery.

Alpha is always shown on the left and bravo on the right, matching the order the
frames were handed to Gemini.

## Task 2 — Artifacts

63 clips from `Model Generations on Eval/`, copied verbatim: 7 models ×
3 categories {Movement, Grounding, Building} × the **first 3** of the 5 clips
per cell (`video_0`, `video_1`, `video_2`).

The two `Consistency (…)` folders are **excluded** — the supplementary ships 5
category folders, so including them would give 7 × 5 × 3 = 105 rather than 63.

`ARTIFACT_VIDEOS_PER_CELL` controls the 3. Raising it back to 5 gives 105
clips — but it reassigns every artifacts item id, which invalidates existing
artifacts responses. Check `responses/` before changing it. They are already
generated-only (640×704, alpha view over bravo view), full length, H.264.
Consistency is excluded per the task definition.

Answers are single-select, with no free-text field:

| Key | Option | Guide text |
|---|---|---|
| 1 | No artifacts | Each view looks clean on its own, even if the two views disagree with each other. |
| 2 | Character artifacts | Character disappearing, duplicating, morphing, or unrecognizable. |
| 3 | Building artifacts | Puts down no blocks or multiple blocks when attempting to build. |
| 4 | Other artifacts | Any other clear corruption. |

Annotators are told to judge each view on its own and report only *clear*
visual artifacts. Two failure modes are explicitly **excluded**, because both
are expected in this data and measured elsewhere:

* **Cross-view disagreement** — the same area looking different between the two
  views, or something present in one and missing from the other.
* **Memory failure** — the world changing when a player looks away and back.

The exclusions appear both in the guide and as a standing line above the answer
buttons, since a one-time intro is easy to forget partway through.

Changing the options requires bumping `TASK_VERSION` in `artifacts.html`. It is
part of the localStorage key, so answers recorded under the old options are
ignored rather than being re-POSTed by a returning annotator's browser and
resurrecting values the task no longer offers. Delete the corresponding
`responses/artifacts__*.json` server-side as well.

**Playback speed** is applied in-browser via `HTMLMediaElement.playbackRate`
(1× / 1.5× / 4×, default 1.5×). Nothing is re-encoded, so the bytes the
annotator sees are the same bytes that went into the supplementary material,
and these rates work in every current browser. The choice persists across
clips; a stored rate from an earlier build is discarded rather than selecting a
button that no longer exists.

## Blinding

Items get opaque ids (`c0001`, `a0042`) assigned in a build-time shuffled order,
and stimulus files are named after the id — so neither the filename, the URL,
nor the ordering reveals which model produced a clip.

The model/episode mapping lives in `data/*_key.json`, which **`serve.py` refuses
to serve over HTTP** (403). It is only read locally by `score_human_eval.py`.
Pass `--serve-key` to override, which un-blinds the task.

On top of the build-time shuffle, each annotator gets their own deterministic
permutation seeded by their name, so order effects do not correlate across
people — and because it is a pure function of the name, resuming always
reproduces the same sequence.

## Progress, saving and resuming

The annotator enters a name on the landing page. After **every answer**:

1. the full answer set is written to `localStorage` (instant, survives a closed
   tab or a crash), and
2. it is `POST`ed to `serve.py`, which writes
   `responses/<task>__<name>.json` atomically (survives a different browser,
   a different machine, or a cleared cache).

On load, both copies are merged and the larger one wins, so neither path can
silently lose work. Re-entering the same name resumes at the first unanswered
item. A "saved" indicator appears in the header on each write; if the server is
unreachable it reads **saved in browser only**.

Progress bars are on the landing page (per task) and pinned to the header of
both annotation pages.

### Reviewing a finished task

A completed task is not locked. The landing page shows a **Review answers**
button once a task is done (and alongside **Resume** while it is in progress),
and the completion screen has the same button. Both open the normal annotation
page at the first item with every previous answer pre-selected, so answers can
be paged through with the arrow keys and changed in place; the guide or the
calibration examples can be re-opened from there too.

Review mode is just `?review=1` on the task URL. It suppresses the completion
screen so a finished annotator is not bounced back to it on every load —
pressing **Finish** on the last item brings the summary back.

### Instruction versions

Two independent counters, both in the task page:

* `TASK_VERSION` — bump when the *options* change. It is part of the
  localStorage key, so old answers are ignored rather than resurrected.
  Existing server-side files must be deleted by hand.
* `INSTRUCTION_VERSION` — bump when the *wording* changes but the options do
  not. This invalidates nothing; it is recorded on each saved run as
  `instruction_version` so a run collected under earlier guidance can be told
  apart later. `score_human_eval.py` prints a note when a run predates the
  current wording, and echoes an optional `instruction_version_note` if the
  file carries one.

To correct a run's version by hand — say the annotator confirms their answers
already followed guidance the page had not yet been updated with — edit the
response file and set:

```json
"instruction_version": 2,
"instruction_version_locked": true,
"instruction_version_note": "why this was set by hand"
```

The lock matters. A browser rewrites the *entire* response file on every save,
so without it the next page load would silently drop both the corrected version
and the note. `serve.py` carries `STICKY_FIELDS` forward from the file on disk
into each incoming payload, and pins `instruction_version` whenever the lock is
set, so stale page code cannot downgrade a deliberate correction.

If you serve the folder with plain `python -m http.server` instead, everything
still works, but saving is localStorage-only — annotators must use the
**Download my answers** button, and you then pass those files to
`score_human_eval.py` directly.

## Changing the model pair

`CONSISTENCY_MODELS` in `build_human_eval.py` selects which two models the
Consistency task compares. Changing it renumbers **every** item id, because ids
are assigned by position in a shuffled list — so `c0007` means a different
screenshot pair before and after. Answers keyed by id would silently point at
the wrong pair.

`migrate_consistency_responses.py` handles this. It matches answers on the
query itself — `(model, eval, query_type, episode, instance)`, which is stable
across rebuilds — rewrites answers for retained models under their new ids, and
drops answers for models no longer in the task. Annotators keep everything they
have already judged for a retained model.

```bash
./deploy.sh stop
cp data/consistency_key.json backups/<stamp>/consistency_key.OLD.json   # BEFORE rebuilding
# edit CONSISTENCY_MODELS, then:
python3 build_human_eval.py --only consistency
python3 migrate_consistency_responses.py --old-key backups/<stamp>/consistency_key.OLD.json --dry-run
python3 migrate_consistency_responses.py --old-key backups/<stamp>/consistency_key.OLD.json
# bump TASK_VERSION in consistency.html and VERSIONS in index.html
./deploy.sh && ./deploy.sh restart
```

Back up `responses/` first — the old key is needed to migrate, and it is
overwritten by the rebuild. Bumping `TASK_VERSION` is not optional: a returning
browser would otherwise re-POST its localStorage answers under ids that now
mean something else. Each migrated file records what happened under a
`migrations` list.

## Deployment

`deploy.sh` syncs the bundle to the annotation host and keeps the server alive
in a tmux session.

```bash
./deploy.sh            # sync, then start the server if it is not already up
./deploy.sh status     # session alive? port answering from inside and outside?
./deploy.sh fetch      # pull collected answers into ./responses/
./deploy.sh logs       # tail the remote server log
./deploy.sh restart    # bounce the server, leaving files alone
./deploy.sh stop       # kill the session (answers untouched)
```

Defaults — override with environment variables:

| Variable | Default |
|---|---|
| `REMOTE` | `fred@69.30.0.74` |
| `REMOTE_DIR` | `/nas2/fred/solaris-human-eval` |
| `PORT` | `9001` |
| `SESSION` | `solaris-human-eval` |

Currently live at **http://69.30.0.74:9001/**.

Two guarantees the script is built around:

* **`responses/` on the remote is never deleted.** `rsync --delete` keeps the
  remote tidy, but `responses/` is excluded, and rsync protects excluded paths
  from deletion. Collected answers are the one thing here that cannot be
  regenerated. The script also never deletes by wildcard.
* **`data/*_key.json` is never uploaded.** `serve.py` would refuse to serve it
  (403), but keeping it off a public host entirely is the stronger guarantee.
  Scoring is local: `./deploy.sh fetch && python3 score_human_eval.py`.

`deploy` is idempotent — re-running syncs changed files and leaves a running
server alone. Use `restart` if you changed `serve.py` itself.

### Security note

The server binds `0.0.0.0` and is **unauthenticated**: anyone who can reach port
9001 can read the stimuli and POST answers under any name. That is fine for
recruiting annotators by sharing a link, and the write path is constrained —
names are validated against `^[A-Za-z0-9 _.\-]{1,64}$` and slugified, bodies are
capped at 8 MB, and path traversal returns 404 (all verified against the live
host). But there is nothing stopping a stranger from submitting junk under a
plausible name. Since answers are keyed by annotator, treat unfamiliar names in
`responses/` as suspect, and take the server down with `./deploy.sh stop` once
collection is finished.

## Scoring

```bash
python human-eval/score_human_eval.py                          # all of responses/
python human-eval/score_human_eval.py path/to/downloaded.json  # a specific file
```

Consistency prints query-level and episode-level accuracy per model with the
paper's VLM numbers alongside. Artifacts prints the label distribution per model
and per category, plus any free-text notes. Incomplete runs are labelled and
partial episodes are excluded rather than counted as wrong.

Sanity check: replaying the VLM's own trial-1 responses through the scorer
reproduces its recorded episode-level accuracy exactly (`flagship` 54.7% =
35/64, `causvid_regression` 32.8% = 21/64).

## Files

| Path | Purpose | In git |
|---|---|---|
| `build_human_eval.py` | Extracts frames, copies videos, writes manifests | yes |
| `serve.py` | Static server + progress API | yes |
| `deploy.sh` | Sync to the annotation host, run it in tmux | yes |
| `score_human_eval.py` | Scores collected answers | yes |
| `index.html`, `consistency.html`, `artifacts.html` | Annotation UI | yes |
| `static/` | Shared CSS + JS | yes |
| `data/*_items.json` | Client manifests (no model info) | no — generated |
| `data/*_key.json` | Answer key (model, episode, expected) | no — generated |
| `frames/`, `videos/` | Stimuli (136 MB / 17 MB) | no — generated |
| `Model Generations on Eval/` | Supplementary source clips | no — large |
| `responses/` | Collected answers | no — data |

Everything in the "no" rows is reproduced by `build_human_eval.py`, except
`Model Generations on Eval/` (the paper supplementary) and `responses/` (the
collected answers) — **back those up separately.**

## Rebuilding

`build_human_eval.py` is deterministic: fixed shuffle seed, fixed calibration
episodes. Re-running reproduces identical ids and identical stimuli.

```bash
python human-eval/build_human_eval.py --only artifacts   # one task
python human-eval/build_human_eval.py --skip-frames      # re-shuffle ids only
```

`--skip-frames` reuses the PNGs already on disk and is only safe while the
shuffle seed is unchanged. It also skips the calibration images, so re-run
without it if you change which calibration episodes are used.

Changing `SHUFFLE_SEED` **invalidates every existing response file**, because
answers are keyed by item id.
