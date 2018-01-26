# Pose-Guided-Person-Image-Generation
Tensorflow implementation of NIPS 2017 paper [Pose Guided Person Image Generation](https://papers.nips.cc/paper/6644-pose-guided-person-image-generation.pdf)

![alt text](https://github.com/charliememory/Pose-Guided-Person-Image-Generation/blob/master/imgs/Poster_task.svg)

## Network architecture
![alt text](https://github.com/charliememory/Pose-Guided-Person-Image-Generation/blob/master/imgs/Paper-framework.svg)

## Resources
 - Pretrained models: [Market-1501](https://drive.google.com/drive/folders/1KLz9SBxOl2Djsqf3NytScPWJIf8K4Qec?usp=sharing), [DeepFashion](https://drive.google.com/drive/folders/19STFGHvwcLFasLXqqLWd-ONiTgARTukN?usp=sharing).
 - Testing data in tf-record format: [Market-1501](https://drive.google.com/drive/folders/1XHYyAAlvn1M73-TNo59uqA8r2YNuM4kg?usp=sharing), [DeepFashion](https://drive.google.com/drive/folders/1f3skQQtsN3mj3lFeYbe8b88hsMl1cyD7?usp=sharing).
 - Filtered training/testing images: [DeepFashion](https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?usp=sharing)

## Testing steps
 1. Download the pretrained models and tf-record data.
 2. Move tf-record data in to ./data directory
 3. Modify the data, pretrained model path in the run_market.sh/run_DF.sh scripts.
 4. run run_market.sh or run_DF.sh 

## TODO list
- [ ] Training and tf-record-data-preparation code

## Related projects
- [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow)
- [improved_wgan_training](https://github.com/igul222/improved_wgan_training)
