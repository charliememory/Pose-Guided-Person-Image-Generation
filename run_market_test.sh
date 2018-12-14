source ~/.bashrc

if [ ! -d ./data/Market_test_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_test_data.zip
    unzip Market_test_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_128x64PoseRCV_Mask_test_sparse_Attr_partBbox7_maskR4R6 Market_test_data
    rm -f Market_test_data.zip
    cd -
fi

######################################################################
############################### Testing ##############################
gpu=0
D_arch='DCGAN'

model_dir=path_to_directory_of_model
start_step=0
pretrained_path=${model_dir}'/model.ckpt-'${start_step}

## Make sure dataset name appear in  --dataset  (i.e. 'Market' or 'DF')
python main.py --dataset=Market_test_data \
             --img_H=128  --img_W=64 \
             --batch_size=32 \
             --is_train=False \
             --model=1 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
             --start_step=${start_step} --pretrained_path=${pretrained_path} \

## Score
stage_num=1
python score.py ${stage_num} ${gpu} ${model_dir} 'test_result'
python score_mask.py ${stage_num} ${gpu} ${model_dir} 'test_result'
