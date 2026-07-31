# How `mc_multiplayer_v2_eval_new_sneak_combined/` was built

This documents the sequence of local file operations that assembled the combined eval directory, and the eval_ids fix applied at the end.

## Step 1: Raw GPU data collection

Episodes were collected on a GPU machine into raw directories:
- `/mnt/data/dl3957/mc_multiplayer_v2_eval_gpu_new_sneak_max_speed/` (all eval types)
- `/mnt/data/dl3957/mc_multiplayer_v2_eval_gpu_new_sneak_max_speed_more_freeze/` (updated bothLookAway + oneLooksAway with longer freeze)

## Step 2: Packaging raw data into eval format

Raw data was aligned and packaged into the standard eval format (`000000_Alpha_instance_000.json` / `_camera.mp4`):

```
python prepare_episodes_for_eval.py \
  --episodes-dir /mnt/data/dl3957/mc_multiplayer_v2_eval_gpu_new_sneak_max_speed \
  --destination_dir /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed
```

A separate non-max-speed collection was also packaged:
```
-> /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak
```

## Step 3: Filtering to 32 episodes

Both packaged directories were filtered down to 32 episodes:

```
python filter_32_episodes.py /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak
-> /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_32_episodes

python filter_32_episodes.py /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed
-> /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes
```

## Step 4: Upload to gcloud

The 32-episode datasets were uploaded to gcloud:

```
gcloud storage cp -r /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_32_episodes \
  gs://solaris-central1/solaris/data/mc_multiplayer_v2_eval_new_sneak

gcloud storage cp -r /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes/ \
  gs://solaris-central1/solaris/data/mc_multiplayer_v2_eval_new_sneak_max_speed
```

## Step 5: Fixing structure data

The original structure data had issues. A fixed version was created by re-running the packaging pipeline on the raw GPU data:

```
python realign_all_datasets.py /mnt/data/dl3957/mc_multiplayer_v2_eval_gpu_new_sneak_max_speed/

python prepare_episodes_for_eval.py \
  --episodes-dir /mnt/data/dl3957/mc_multiplayer_v2_eval_gpu_new_sneak_max_speed \
  --destination_dir /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_structure_fixed
```

The fixed structure data was uploaded to gcloud, replacing the originals:

```
gcloud storage rm -r gs://solaris-central1/.../mc_multiplayer_v2_eval_new_sneak_max_speed/structureEval
gcloud storage rm -r gs://solaris-central1/.../mc_multiplayer_v2_eval_new_sneak_max_speed/structureNoPlaceEval

gcloud storage cp -r .../structure_fixed/structureEval \
  gs://solaris-central1/.../mc_multiplayer_v2_eval_new_sneak_max_speed/structureEval
gcloud storage cp -r .../structure_fixed/structureNoPlaceEval \
  gs://solaris-central1/.../mc_multiplayer_v2_eval_new_sneak_max_speed/structureNoPlaceEval
```

The fixed data was also moved into the local 32-episode directory, replacing the originals:

```
mv /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_structure_fixed/structure* \
   /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes
```

## Step 6: Adding long look-away episodes

A separate collection with longer freeze frames was packaged:

```
python prepare_episodes_for_eval.py \
  --episodes-dir /mnt/data/dl3957/mc_multiplayer_v2_eval_gpu_new_sneak_max_speed_more_freeze \
  --destination_dir /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_look_away_updated
```

This was renamed and uploaded:

```
mv /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_look_away_updated \
   /mnt/data/dl3957/mc_multiplayer_v2_eval_new_sneak_max_speed_long/

gcloud storage cp -r /mnt/data/dl3957/mc_multiplayer_v2_eval_new_sneak_max_speed_long \
  gs://solaris-central1/solaris/data/mc_multiplayer_v2_eval_new_sneak_max_speed_long
```

## Step 7: Assembling the combined directory

The combined directory was assembled locally from the 32-episode directories:

```
# Copy the max-speed 32-episode data (which now has fixed structure)
cp -r /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes .

# Copy the fixed structure data into the local working copy
cp -r /mnt/data/dl3957/mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes/structure* \
  mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes
```

The directory was then synced to a remote machine and renamed to `mc_multiplayer_v2_eval_new_sneak_combined`, with `translationEval` coming from the non-max-speed source and the `_long` datasets added separately.

Final data sources in `mc_multiplayer_v2_eval_new_sneak_combined/`:
- Most evals: from `mc_multiplayer_v2_eval_packaged_new_sneak_max_speed_32_episodes` (max-speed)
- `structureEval`, `structureNoPlaceEval`: from `structure_fixed` (moved into the 32-ep dir)
- `translationEval`: from `mc_multiplayer_v2_eval_packaged_new_sneak_32_episodes` (non-max-speed)
- `bothLookAwayEval_long`, `oneLooksAwayEval_long`: from `mc_multiplayer_v2_eval_new_sneak_max_speed_long`

## Step 8: Fixing structure eval_ids (2026-03-27)

The eval_ids for structure had been generated on a TPU from a different 64-episode dataset (`mc_multiplayer_v2_eval_packaged_new_sneak_max_speed`), which contained entirely different episode data from the fixed 32-episode structure data in the combined directory. This was fixed by regenerating the eval_ids locally from the correct data:

```
# Temporarily changed config to point to the local 32-episode data, then ran:
conda run -n oasis python dataset/prepare_eval_sample_ids_mp_final_eval.py \
  dataset=multiplayer_v2_eval_structure_max_speed device=gpu num_frames_eval=257

conda run -n oasis python dataset/prepare_eval_sample_ids_mp_final_eval.py \
  dataset=multiplayer_v2_eval_structure_no_place_max_speed device=gpu num_frames_eval=257
```

After this, all 10 eval subdirs in the combined directory have verified-matching eval_ids.
