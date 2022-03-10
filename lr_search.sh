python train.py --batch-size 8 --test-batch-size 8 --epochs 10 --data_folder "../2d3ds_sphere" --max_level 5 --min_level 0 --model FPN --feat 32 --fold 1 --log_dir logs/log_f32_fold1 --decay --in_ch rgbd --optim sgd --lr 0.0001

python train.py --batch-size 8 --test-batch-size 8 --epochs 10 --data_folder "../2d3ds_sphere" --max_level 5 --min_level 0 --model FPN --feat 32 --fold 1 --log_dir logs/log_f32_fold2 --decay --in_ch rgbd --optim sgd --lr 0.1

python train.py --batch-size 8 --test-batch-size 8 --epochs 10 --data_folder "../2d3ds_sphere" --max_level 5 --min_level 0 --model FPN --feat 32 --fold 1 --log_dir logs/log_f32_fold3 --decay --in_ch rgbd --optim sgd --lr 0.005