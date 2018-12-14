source ~/.bashrc

if [ ! -d ./data/Market_train_data ]; then
    cd data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_train_data.zip
    unzip Market_train_data.zip
    mv data4tf_GAN_attr_pose_onlyPosPair_128x64PoseRCV_Mask_sparse_Attr_partBbox7_maskR4R6 Market_train_data
    rm -f Market_train_data.zip
    cd -
fi

#######################################################################
################################ Training #############################
gpu=0
D_arch='DCGAN'

model_dir=path_to_directory_of_model

## Make sure dataset name appear in  --dataset  (i.e. 'Market' or 'DF')
python main.py --dataset=Market_train_data \
             --img_H=128  --img_W=64 \
             --batch_size=16 --max_step=60000 \
             --d_lr=0.00002  --g_lr=0.00002 \
             --lr_update_step=50000 \
             --is_train=True \
             --model=1 \
             --D_arch=${D_arch} \
             --gpu=${gpu} \
             --z_num=64 \
             --model_dir=${model_dir} \
