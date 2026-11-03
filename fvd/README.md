# FVD for NeurIPS 2026 (Table 8)

This directory holds the code and provenance for the test-split FVD in Table 8 of the
NeurIPS 2026 submission (submitted 2026-05-06). Regenerate the table from the saved features:

```bash
bash fvd/fvd_table8.sh
```

## Where the numbers come from

Table 8 is the **Mixed** column of the "Random-window 16-frame FVD (seed=42)" table in
[`RESULTS.md`](RESULTS.md). Fred Lu computed it on the lab desktop on 2026-05-06, in
`~/solaris-additional-metrics/solaris_fvd`. That directory was never committed. This
directory and the archive below replace it.

`fvd_table8.sh` recomputes every Table 8 value exactly from the archived features. The
shaded Self Forcing row is the final Solaris configuration, so its value is **81.3**. The
88.7 printed in that row of the submission is Independent's value, copied by mistake.

## Method

| Step | What was done |
|---|---|
| Videos | the 2,048 test-split segments of `eval_ids_multiplayer_v2.json` (57 test episodes, 257 frames each), one 1280×704 side-by-side mp4 per segment |
| Quadrants | 0 = GT bot 1, 1 = Gen bot 1, 2 = GT bot 2, 3 = Gen bot 2 (640×352 each) |
| Clip | one contiguous 16-frame window per video, start from `np.random.default_rng(42)` offsets; the same windows for every model |
| Resize | each quadrant squashed to 224×224 (bilinear), scaled to [-1, 1] |
| Features | StyleGAN-V I3D TorchScript (`flateon/FVD-I3D-torchscript`, md5 `8d45542982dc92282a73cb852e3c53d4`), 400-d logits |
| FVD | Fréchet distance with both bots pooled: q0+q2 (4,096 GT clips) vs q1+q3 (4,096 generated clips) |

The paper text must match this. The segments are from the test split, not "held-out
training videos". The two players are pooled, not "averaged across two players".

## Files

| File | Purpose |
|---|---|
| `compute_features.py` | I3D feature extraction for one quadrant; identical to the 2026-05-06 file |
| `compute_fvd.py` | Fréchet distance between feature files, with pooling over several files |
| `run_rw16.sh` | the 2026-05-06 run, with paths changed to the archive (arguments unchanged) |
| `fvd_table8.sh` | prints Table 8 from a features directory |
| `RESULTS.md` | the results table written on 2026-05-06, unchanged |
| `features_rw16.md5` | checksums of the 28 archived feature files |
| `logs/` | the 2026-05-06 extraction logs |
| `original_2026-05-06/` | the unchanged original shell scripts and README, and yl11330's earlier all-frames `compute_features.py` |

## Data

Everything except the videos is under `/mnt/data-hdd/dl3957/neurips2026_eval/fvd/` on the lab
desktop and is mirrored, checksum-verified, at `gs://solaris-central1/solaris/neurips2026_eval/fvd/`:

| Data | Subdirectory |
|---|---|
| Features (2026-05-06, used for Table 8) | `features_rw16/` |
| All-frames features (not in the paper) | `features_allframes/` |
| I3D weights | `models/i3d_torchscript.pt` |
| Snapshots of both `solaris_fvd` dirs | `script_snapshots/` (with `MD5SUMS`) |
| Input videos (HDD only; the GCS copies are listed below) | `test_set_generations/` |

To regenerate the table on another machine, download the features and point the script at them:

```bash
gsutil -m cp -r gs://solaris-central1/solaris/neurips2026_eval/fvd/features_rw16 .
FEATS=$PWD/features_rw16 PY=python bash fvd/fvd_table8.sh
```

The input videos are also on GCS, identical by md5 to the copies used on 2026-05-06
(checked 2026-11-02 for all 14,336 files):

| Short name | Table 8 row | GCS directory under `gs://solaris-central1/output/` |
|---|---|---|
| `no_player_attn_sf` | Independent | `no_player_attn_sf/step_0001200_duet_2048` |
| `v2_blk` | Frame concat | `v2_concat_c_fix_bidirectional_fix_blocksize_sf/step_0001200_duet` |
| `v2_scratch` | Solaris w/o pretrain | `v2_flagship_from_scratch_fix_bidirectional_sf/step_0001200_duet` |
| `v2_flag` | Solaris (and the shaded Self Forcing row) | `v2_flagship_fix_bidirectional_sf/step_0001200_duet` |
| `v2_reg` | ODE Reg, Pre-DMD ×, KV-BP ✓ | `v2_flagship_fix_bidirectional_sf_regression_lr_1e-4_continue/step_0001200_duet` |
| `dmd_cont` | Causal FT, Pre-DMD ✓, KV-BP ✓ | `flagship_fix_bidirectional_sf_dmd_continue/step_0001200_duet` |
| `v2_nokv` | Causal FT, Pre-DMD ×, KV-BP × | `v2_flagship_fix_bidirectional_sf_no_kv_backprop/step_0001200_duet` |

## Rerunning from the videos

```bash
GPU=0 QUADS='0 1' bash fvd/run_rw16.sh
GPU=1 QUADS='2 3' bash fvd/run_rw16.sh
FEATS=fvd/outputs_rw16 bash fvd/fvd_table8.sh
```

This takes about an hour on two RTX 4090s. Each job decodes whole 257-frame videos in 8
worker processes, so run at most two jobs at a time on the 122 GB desktop. Four parallel
jobs ran out of memory there.
