source ~/.bashrc

if [ ! -d ./data/DF_test_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_test_data.zip
    unzip DF_test_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_256x256PoseRCV_Mask_test_sparse_partBbox37_maskR4R8_roi10Complete DF_test_data
    rm -f DF_test_data.zip
    cd -
fi

#######################################################################
################################ Testing ##############################
gpu=0
D_arch='DCGAN'

model_dir=path_to_directory_of_model
start_step=0
pretrained_path=${model_dir}'/model.ckpt-'${start_step}

## Make sure dataset name appear in  --dataset  (i.e. 'Market' or 'DF')
python main.py --dataset=DF_test_data \
             --img_H=256  --img_W=256 \
             --batch_size=1 \
             --is_train=False \
             --model=11 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=${start_step} --pretrained_path=${pretrained_path} \

## Score
stage_num=1
python score.py ${stage_num} ${gpu} ${model_dir} 'test_result'
python score_mask.py ${stage_num} ${gpu} ${model_dir} 'test_result'
