source ~/.bashrc

####################### Testing #####################
log_dir=path_to_directory_of_model
gpu=0
D_arch='DCGAN'
stage=2

model_dir=${log_dir}'/PG2_model'
start_step=0
pretrained_path=${model_dir}'/model.ckpt-'${start_step}
python main.py --dataset=market1501_train_attr_pose_onlyPosPair_128x64Pose_Mask_test_sparse \
             --use_gpu=False --input_scale_size=128 \
             --batch_size=2 --max_step=40000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --is_train=False \
             --model=1 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=${start_step} --pretrained_path=${pretrained_path} \

## Score
python score.py ${stage} ${gpu} ${model_dir} 'test_result'
python score_mask.py ${stage} ${gpu} ${model_dir} 'test_result'
