source ~/.bashrc

if [ ! -d ./data/DF_train_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_train_data.zip
    unzip DF_train_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_256x256PoseRCV_Mask_sparse_partBbox37_maskR4R8_roi10Complete DF_train_data
    rm -f DF_train_data.zip
    cd -
fi


#######################################################################
################################ Training #############################
gpu=0
D_arch='DCGAN'

model_dir=path_to_directory_of_model

## Make sure dataset name appear in  --dataset  (i.e. 'Market' or 'DF')
python main.py --dataset=DF_train_data \
             --img_H=256  --img_W=256 \
             --batch_size=1 --max_step=80000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --is_train=True \
             --model=11 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
