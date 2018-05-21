source ~/.bashrc

####################### Testing #####################
gpu=0
D_arch='DCGAN'
stage=2

model_dir=path_to_directory_of_model
start_step=0
pretrained_path=${model_dir}'/model.ckpt-'${start_step}

## Make sure dataset name appear in  --dataset  (i.e. 'Market' or 'DeepFashion')
python main.py --dataset=data4tf_DeepFashion_256x256Pose_Mask_test_sparse \
             --img_H=256  --img_W=256 \
             --batch_size=1 --max_step=40000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --is_train=False \
             --model=11 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=${start_step} --pretrained_path=${pretrained_path} \
             # --test_one_by_one=True

## Score
python score.py ${stage} ${gpu} ${model_dir} 'test_result'
python score_mask.py ${stage} ${gpu} ${model_dir} 'test_result'
