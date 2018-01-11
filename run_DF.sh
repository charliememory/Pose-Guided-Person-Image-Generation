source ~/.bashrc_liqianma

####################### Test demo test data #####################
log_dir='/esat/diamond/liqianma/exp_logs/GAN/PG2_demo'
gpu=0
D_arch='DCGAN'
model_dir=${log_dir}'/PG2_model_DF' # model=13
############################
start_step=0
pretrained_path=${model_dir}'/model.ckpt-'${start_step}
python main.py --dataset=deepfashion_pose_onlyPosPair_128x64Pose_Mask_test_sparse \
             --use_gpu=True --input_scale_size=128 \
             --batch_size=1 --max_step=40000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --is_train=False  --test_one_by_one=True \
             --model=13 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=${start_step} --pretrained_path=${pretrained_path} \