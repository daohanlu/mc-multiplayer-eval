# FVD Results — Solaris (step_0001200_duet, n=2048)

I3D features (400-dim, Kinetics-400 InceptionV1) extracted per quadrant from
1280×704 side-by-side mp4s and compared with Frechet Video Distance.

- **Bot1**  = `q0` (GT Bot1) vs `q1` (Gen Bot1)
- **Bot2**  = `q2` (GT Bot2) vs `q3` (Gen Bot2)
- **Mixed** = `q0+q2`        vs `q1+q3`

Lower is better.

## Random-window 16-frame FVD (seed=42)

Per-video deterministic random 16-frame contiguous window — same set of
windows used across all 7 models so the comparison is matched.

| Model               | Bot1     | Bot2     | Mixed    |
|---------------------|---------:|---------:|---------:|
| dmd_cont            | 143.336  | 160.306  | 131.840  |
| v2_blk              | 101.533  | 225.039  | 113.206  |
| v2_flag             |  93.500  | 106.647  |  81.329  |
| v2_nokv             |  90.277  | 103.047  |  76.897  |
| v2_reg              | 152.279  | 201.731  | 151.873  |
| v2_scratch          | 116.484  | 130.687  | 101.987  |
| **no_player_attn_sf** | **100.368** | **123.548** | **88.705** |

### Ranking by Mixed (random-window)

| Rank | Model               | Mixed   |
|-----:|---------------------|--------:|
|    1 | v2_nokv             |  76.897 |
|    2 | v2_flag             |  81.329 |
|    3 | **no_player_attn_sf** | **88.705** |
|    4 | v2_scratch          | 101.987 |
|    5 | v2_blk              | 113.206 |
|    6 | dmd_cont            | 131.840 |
|    7 | v2_reg              | 151.873 |

## All-frames FVD (full 257-frame clip per video)

Original setting: feed the entire 257-frame clip to I3D once per video. Kept
here for reference; values are not directly comparable to the random-window
table since I3D was trained on 16-frame contiguous clips.

| Model               | Bot1     | Bot2     | Mixed    |
|---------------------|---------:|---------:|---------:|
| dmd_cont            | 279.120  | 304.304  | 258.015  |
| v2_blk              | 164.559  | 417.271  | 211.867  |
| v2_flag             | 159.308  | 157.762  | 132.459  |
| v2_nokv             | 153.868  | 140.504  | 120.483  |
| v2_reg              | 310.828  | 352.955  | 295.750  |
| v2_scratch          | 232.834  | 223.505  | 195.502  |
| **no_player_attn_sf** | **149.248** | **183.529** | **135.530** |

### Ranking by Mixed (all-frames)

| Rank | Model               | Mixed   |
|-----:|---------------------|--------:|
|    1 | v2_nokv             | 120.483 |
|    2 | v2_flag             | 132.459 |
|    3 | **no_player_attn_sf** | **135.530** |
|    4 | v2_scratch          | 195.502 |
|    5 | v2_blk              | 211.867 |
|    6 | dmd_cont            | 258.015 |
|    7 | v2_reg              | 295.750 |

The Mixed-FVD ranking is identical between the two protocols. Magnitudes
shrink in the random-window table because 16-frame I3D logits sit closer in
feature space than full-clip logits do.

## Short-name → source dir

| Short             | Source generation dir |
|-------------------|-----------------------|
| dmd_cont          | `flagship_fix_bidirectional_sf_dmd_continue/step_0001200_duet` |
| v2_blk            | `v2_concat_c_fix_bidirectional_fix_blocksize_sf/step_0001200_duet` |
| v2_flag           | `v2_flagship_fix_bidirectional_sf/step_0001200_duet` |
| v2_nokv           | `v2_flagship_fix_bidirectional_sf_no_kv_backprop/step_0001200_duet` |
| v2_reg            | `v2_flagship_fix_bidirectional_sf_regression_lr_1e-4_continue/step_0001200_duet` |
| v2_scratch        | `v2_flagship_from_scratch_fix_bidirectional_sf/step_0001200_duet` |
| no_player_attn_sf | `test_set_generations/no_player_attn_sf/step_0001200_duet_2048` |
