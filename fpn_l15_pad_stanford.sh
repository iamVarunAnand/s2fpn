#!/bin/bash

source /home/varun/environments/mlp/bin/activate
cd /home/varun/mlp-cw3

python train.py --model fpn --upsample zero-pad --batch-size 16 --test-batch-size 16 --epochs 100 --data_folder ../2d3ds_sphere --max_level 5 --min_level 1 --feat 32 --fold 1 --log_dir logs/stanford/ugscnn_fpn_l15_pad_fold1 --decay --in_ch rgbd

python train.py --model fpn --upsample zero-pad --batch-size 16 --test-batch-size 16 --epochs 100 --data_folder ../2d3ds_sphere --max_level 5 --min_level 1 --feat 32 --fold 2 --log_dir logs/stanford/ugscnn_fpn_l15_pad_fold2 --decay --in_ch rgbd

python train.py --model fpn --upsample zero-pad --batch-size 16 --test-batch-size 16 --epochs 100 --data_folder ../2d3ds_sphere --max_level 5 --min_level 1 --feat 32 --fold 3 --log_dir logs/stanford/ugscnn_fpn_l15_pad_fold3 --decay --in_ch rgbd