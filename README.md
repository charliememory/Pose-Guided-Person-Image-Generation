# Pose-Guided-Person-Image-Generation
Tensorflow implementation of NIPS 2017 paper [Pose Guided Person Image Generation](https://papers.nips.cc/paper/6644-pose-guided-person-image-generation.pdf)

![alt text](https://github.com/charliememory/Pose-Guided-Person-Image-Generation/blob/master/imgs/Poster_task.svg)

## Network architecture
![alt text](https://github.com/charliememory/Pose-Guided-Person-Image-Generation/blob/master/imgs/Paper-framework.svg)

## Dependencies
- python 2.7
- tensorflow-gpu (1.4.1)
- numpy (1.14.0)
- Pillow (5.0.0)
- scikit-image (0.13.0)
- scipy (1.0.1)
- matplotlib (2.0.0)

## Resources
 - Pretrained models: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/models/Market1501_model.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/models/DF_model.zip).
 - Training data in tf-record format: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_train_data.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_train_data.zip).
 - Testing data in tf-record format: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market_test_data.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_test_data.zip).
 - Raw data: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/Market1501_img_pose_attr_seg.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_img_pose.zip) 
 - Testing results: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/results/Market_test_result_12800.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/results/DF_test_result_12800.zip) 

## TF-record data preparation steps
 You can skip this data preparation procedure if directly using the tf-record data files.
 1. `cd datasets`
 2. `./run_convert_market.sh` to download and convert the original images, poses, attributes, segmentations
 3. `./run_convert_DF.sh` to download and convert the original images, poses

 Note: we also provide the convert code for [Market-1501 Attribute](https://github.com/vana77/Market-1501_Attribute) and Market-1501 Segmentation results from [PSPNet](https://github.com/hszhao/PSPNet). These extra info. are provided for further research. In our experiments, pose mask are ontained from pose key-points (see `_getPoseMask` function in convert .py files).

## Training steps
 1. Download the tf-record training data.
 2. Modify the `model_dir` in the run_market_train.sh/run_DF_train.sh scripts.
 3. run run_market_train.sh/run_DF_train.sh 
 
 Note: we use a triplet instead of pair real/fake for adversarial training to keep training more stable.

## Testing steps
 1. Download the pretrained models and tf-record testing data.
 2. Modify the `model_dir` in the run_market_test.sh/run_DF_test.sh scripts.
 3. run run_market_test.sh/run_DF_test.sh 

## Citation
```
@inproceedings{ma2017pose,
  title={Pose guided person image generation},
  author={Ma, Liqian and Jia, Xu and Sun, Qianru and Schiele, Bernt and Tuytelaars, Tinne and Van Gool, Luc},
  booktitle={Advances in Neural Information Processing Systems},
  pages={405--415},
  year={2017}
}
```

## Related projects
- [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow)
- [improved_wgan_training](https://github.com/igul222/improved_wgan_training)

