source ~/.bashrc

if [ ! -d ../data/Market1501_img_pose_attr_seg ]; then
    cd ../data
    wget homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market1501_img_pose_attr_seg.zip
    unzip Market1501_img_pose_attr_seg.zip
    rm -f Market1501_img_pose_attr_seg.zip
    cd -
fi

python convert_market.py '../data/Market1501_img_pose_attr_seg' 'train'
python convert_market.py '../data/Market1501_img_pose_attr_seg' 'test'
python convert_market.py '../data/Market1501_img_pose_attr_seg' 'test_samples'
