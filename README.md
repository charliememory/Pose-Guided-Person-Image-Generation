# Pose-Guided-Person-Image-Generation
Tensorflow implementation of NIPS 2017 paper [Pose Guided Person Image Generation](https://papers.nips.cc/paper/6644-pose-guided-person-image-generation.pdf)

![alt text](https://github.com/charliememory/Pose-Guided-Person-Image-Generation/blob/master/imgs/Poster_task.svg)

## Network architecture
![alt text](https://github.com/charliememory/Pose-Guided-Person-Image-Generation/blob/master/imgs/Paper-framework.svg)

## Resources
 - Pretrained models: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/models/Market1501.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/models/DF.zip).
 - Testing data in tf-record format: [Market-1501](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/test_data_Market.zip), [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/test_data_DF.zip).
 - Filtered training/testing images: [DeepFashion](http://homes.esat.kuleuven.be/~liqianma/NIPS17_PG2/data/DF_filted_up_train_test_data.zip) 

## Testing steps
 1. Download the pretrained models and tf-record data.
 2. Move tf-record data in to ./data directory
 3. Modify the data, pretrained model path in the run_market.sh/run_DF.sh scripts.
 4. run run_market.sh or run_DF.sh 

## TODO list
- [ ] Training and tf-record-data-preparation code

## Citation
```
@inproceedings{DBLP:conf/nips/MaJSSTG17,
  title={Pose Guided Person Image Generation},
  author={Ma, Liqian and Jia, Xu and Sun, Qianru and Schiele, Bernt and Tuytelaars, Tinne and Van Gool, Luc},
  booktitle = {Advances in Neural Information Processing Systems 30: Annual Conference
               on Neural Information Processing Systems 2017, 4-9 December 2017,
               Long Beach, CA, {USA}},
  pages     = {405--415},
  year      = {2017}
}
```

## Related projects
- [BEGAN-tensorflow](https://github.com/carpedm20/BEGAN-tensorflow)
- [improved_wgan_training](https://github.com/igul222/improved_wgan_training)
