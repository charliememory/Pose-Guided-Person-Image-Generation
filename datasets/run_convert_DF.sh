source ~/.bashrc

if [ ! -d ../data/DF_img_pose ]; then
    cd ../data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_img_pose.zip
    unzip DF_img_pose.zip
    rm -f DF_img_pose.zip
    cd -
fi

python convert_DF.py '../data/DF_img_pose' 'train'
python convert_DF.py '../data/DF_img_pose' 'test'
python convert_DF.py '../data/DF_img_pose' 'test_samples'
python convert_DF.py '../data/DF_img_pose' 'test_seq'
