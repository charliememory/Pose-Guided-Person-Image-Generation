# source ~/.bashrc_liqianma
source ~/.bashrc_liqianma
# source /BS/sun_project2/work/mlq_project/tensorflow/bin/activate

####################### Testing #####################
log_dir='/esat/diamond/liqianma/exp_logs/GAN/PG2_demo'
gpu=0
D_arch='DCGAN'
stage=2

model_dir=${log_dir}'/PG2_model' # model=3
start_step=0
pretrained_path=${model_dir}'/model.ckpt-'${start_step}
python main.py --dataset=market1501_train_attr_pose_onlyPosPair_128x64Pose_Mask_test_sparse \
             --use_gpu=False --input_scale_size=128 \
             --batch_size=2 --max_step=40000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --is_train=False \
             --model=3 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=${start_step} --pretrained_path=${pretrained_path} \

## Score
python score.py ${stage} ${gpu} ${model_dir} 'test_result'
python score_mask.py ${stage} ${gpu} ${model_dir} 'test_result'
