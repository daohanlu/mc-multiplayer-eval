# I3D feature extraction for FVD: 7 experiment dirs x 4 quadrants = 28 jobs
# Quadrants: 0=GT Bot1 (TL), 1=Gen Bot1 (TR), 2=GT Bot2 (BL), 3=Gen Bot2 (BR)
# makesbatch --time 1 --ngpu 1 --l40s --bash_file bash_cmds/fvd_features.sh

# dmd_cont_q0
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/flagship_fix_bidirectional_sf_dmd_continue/step_0001200_duet \
    --quadrant 0 --output outputs/dmd_cont/q0.npz

# dmd_cont_q1
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/flagship_fix_bidirectional_sf_dmd_continue/step_0001200_duet \
    --quadrant 1 --output outputs/dmd_cont/q1.npz

# dmd_cont_q2
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/flagship_fix_bidirectional_sf_dmd_continue/step_0001200_duet \
    --quadrant 2 --output outputs/dmd_cont/q2.npz

# dmd_cont_q3
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/flagship_fix_bidirectional_sf_dmd_continue/step_0001200_duet \
    --quadrant 3 --output outputs/dmd_cont/q3.npz

# v2_blk_q0
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_concat_c_fix_bidirectional_fix_blocksize_sf/step_0001200_duet \
    --quadrant 0 --output outputs/v2_blk/q0.npz

# v2_blk_q1
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_concat_c_fix_bidirectional_fix_blocksize_sf/step_0001200_duet \
    --quadrant 1 --output outputs/v2_blk/q1.npz

# v2_blk_q2
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_concat_c_fix_bidirectional_fix_blocksize_sf/step_0001200_duet \
    --quadrant 2 --output outputs/v2_blk/q2.npz

# v2_blk_q3
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_concat_c_fix_bidirectional_fix_blocksize_sf/step_0001200_duet \
    --quadrant 3 --output outputs/v2_blk/q3.npz

# v2_flag_q0
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf/step_0001200_duet \
    --quadrant 0 --output outputs/v2_flag/q0.npz

# v2_flag_q1
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf/step_0001200_duet \
    --quadrant 1 --output outputs/v2_flag/q1.npz

# v2_flag_q2
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf/step_0001200_duet \
    --quadrant 2 --output outputs/v2_flag/q2.npz

# v2_flag_q3
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf/step_0001200_duet \
    --quadrant 3 --output outputs/v2_flag/q3.npz

# v2_nokv_q0
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf_no_kv_backprop/step_0001200_duet \
    --quadrant 0 --output outputs/v2_nokv/q0.npz

# v2_nokv_q1
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf_no_kv_backprop/step_0001200_duet \
    --quadrant 1 --output outputs/v2_nokv/q1.npz

# v2_nokv_q2
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf_no_kv_backprop/step_0001200_duet \
    --quadrant 2 --output outputs/v2_nokv/q2.npz

# v2_nokv_q3
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf_no_kv_backprop/step_0001200_duet \
    --quadrant 3 --output outputs/v2_nokv/q3.npz

# v2_reg_q0
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf_regression_lr_1e-4_continue/step_0001200_duet \
    --quadrant 0 --output outputs/v2_reg/q0.npz

# v2_reg_q1
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf_regression_lr_1e-4_continue/step_0001200_duet \
    --quadrant 1 --output outputs/v2_reg/q1.npz

# v2_reg_q2
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf_regression_lr_1e-4_continue/step_0001200_duet \
    --quadrant 2 --output outputs/v2_reg/q2.npz

# v2_reg_q3
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_fix_bidirectional_sf_regression_lr_1e-4_continue/step_0001200_duet \
    --quadrant 3 --output outputs/v2_reg/q3.npz

# v2_scratch_q0
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_from_scratch_fix_bidirectional_sf/step_0001200_duet \
    --quadrant 0 --output outputs/v2_scratch/q0.npz

# v2_scratch_q1
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_from_scratch_fix_bidirectional_sf/step_0001200_duet \
    --quadrant 1 --output outputs/v2_scratch/q1.npz

# v2_scratch_q2
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_from_scratch_fix_bidirectional_sf/step_0001200_duet \
    --quadrant 2 --output outputs/v2_scratch/q2.npz

# v2_scratch_q3
.venv/bin/python compute_features.py \
    --data_dir /scratch/dl3957/rebuttal_generations/v2_flagship_from_scratch_fix_bidirectional_sf/step_0001200_duet \
    --quadrant 3 --output outputs/v2_scratch/q3.npz

# no_player_attn_sf_q0
.venv/bin/python compute_features.py \
    --data_dir /home/dl3957/solaris-additional-metrics/test_set_generations/no_player_attn_sf/step_0001200_duet_2048 \
    --quadrant 0 --output outputs/no_player_attn_sf/q0.npz

# no_player_attn_sf_q1
.venv/bin/python compute_features.py \
    --data_dir /home/dl3957/solaris-additional-metrics/test_set_generations/no_player_attn_sf/step_0001200_duet_2048 \
    --quadrant 1 --output outputs/no_player_attn_sf/q1.npz

# no_player_attn_sf_q2
.venv/bin/python compute_features.py \
    --data_dir /home/dl3957/solaris-additional-metrics/test_set_generations/no_player_attn_sf/step_0001200_duet_2048 \
    --quadrant 2 --output outputs/no_player_attn_sf/q2.npz

# no_player_attn_sf_q3
.venv/bin/python compute_features.py \
    --data_dir /home/dl3957/solaris-additional-metrics/test_set_generations/no_player_attn_sf/step_0001200_duet_2048 \
    --quadrant 3 --output outputs/no_player_attn_sf/q3.npz

#! Submitted batch job 5036320 -> 36_mren -- dmd_cont_q0
#! Submitted batch job 5036321 -> 36_general -- dmd_cont_q1
#! Submitted batch job 5036322 -> 36_cds -- dmd_cont_q2
#! Submitted batch job 5036323 -> 36_mren -- dmd_cont_q3
#! Submitted batch job 5036324 -> 36_general -- v2_blk_q0
#! Submitted batch job 5036329 -> 36_cds -- v2_blk_q1
#! Submitted batch job 5036330 -> 36_mren -- v2_blk_q2
#! Submitted batch job 5036331 -> 36_general -- v2_blk_q3
#! Submitted batch job 5036332 -> 36_cds -- v2_flag_q0
#! Submitted batch job 5036333 -> 36_mren -- v2_flag_q1
#! Submitted batch job 5036334 -> 36_general -- v2_flag_q2
#! Submitted batch job 5036335 -> 36_cds -- v2_flag_q3
#! Submitted batch job 5036336 -> 36_mren -- v2_nokv_q0
#! Submitted batch job 5036337 -> 36_mren -- v2_nokv_q1
#! Submitted batch job 5036338 -> 36_mren -- v2_nokv_q2
#! Submitted batch job 5036339 -> 36_mren -- v2_nokv_q3
#! Submitted batch job 5036340 -> 36_mren -- v2_reg_q0
#! Submitted batch job 5036341 -> 36_mren -- v2_reg_q1
#! Submitted batch job 5036342 -> 36_mren -- v2_reg_q2
#! Submitted batch job 5036343 -> 36_mren -- v2_reg_q3
#! Submitted batch job 5036344 -> 36_mren -- v2_scratch_q0
#! Submitted batch job 5036345 -> 36_mren -- v2_scratch_q1
#! Submitted batch job 5036346 -> 36_mren -- v2_scratch_q2
#! Submitted batch job 5036347 -> 36_mren -- v2_scratch_q3
