
AugFPN: Improving Multi-scale Feature Learning for Object Detection
---------------------
By Chaoxu Guo, Bin Fan, Qian Zhang, Shiming Xiang, Chunhong Pan

arxiv paper, [pdf](https://arxiv.org/abs/1912.05384)

This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection)


Introduction
----------------
Current state-of-the-art detectors typically exploit feature pyramid to detect objects at different scales. Among them, FPN is one of the representative works that build a feature pyramid by multi-scale features summation. However, the design defects behind prevent the multi-scale features from being fully exploited. In this paper, we begin by first analyzing the design defects of feature pyramid in FPN, and then introduce a new feature pyramid architecture named AugFPN to address these problems. Specifically, AugFPN consists of three components: Consistent Supervision, Residual Feature Augmentation, and Soft RoI Selection. AugFPN narrows the semantic gaps between features of different scales before feature fusion through Consistent Supervision. In feature fusion, ratio-invariant context information is extracted by Residual Feature Augmentation to reduce the information loss of feature map at the highest pyramid level. Finally, Soft RoI Selection is employed to learn a better RoI feature adaptively after feature fusion.

Install
-------------
Please refer to [INSTALL.md](INSTALL.md) for installation.

Prepare data
----------
```
  mkdir -p data/coco
  ln -s /path_to_coco_dataset/annotations data/coco/annotations
  ln -s /path_to_coco_dataset/train2017 data/coco/train2017
  ln -s /path_to_coco_dataset/test2017 data/coco/test2017
  ln -s /path_to_coco_dataset/val2017 data/coco/val2017
```


Pretrained Models
---------------

Pretrained models will be available.

Training
--------------
```shell
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --validate --work_dir <WORK_DIR>
```
For example,
```shell
./tools/dist_train.sh configs/faster_rcnn_r50_augfpn_1x.py 8 --validate --work_dir faster_rcnn_r50_augfpn_1x
```

see more details at [mmdetection](https://github.com/open-mmlab/mmdetection)


Testing
-----------
```shell
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --gpus <GPU_NUM> --out <OUT_FILE> --eval <EVAL_TYPE>
```
When test results of detection, use `--eval bbox`. When test results of instance segmentation, use `--eval bbox segm`. See more details at [mmdetection](https://github.com/open-mmlab/mmdetection).

For example,
```shell
python tools/test.py configs/mask_rcnn_r50_augfpn_1x.py <CHECKPOINT_FILE> --gpus 8 --out results.pkl --eval bbox segm
```

Results on testdev
---------
| Backbone | detector | mAP(mask) | mAP(det)  |
|----------|--------|-----------|-----------|
| ResNet-50 FPN | Faster R-CNN | - | 36.5 |
| ResNet-50 AugFPN | Faster R-CNN | - | 38.8 |
| ResNet-50 FPN | Mask R-CNN | 34.4 | 37.5 |
| ResNet-50 AugFPN | Mask R-CNN | 36.3 | 39.5 |
| ResNet-50 FPN | RetinaNet |  -  | 35.9    |
| ResNet-50 AugFPN| RetinaNet | -  | 37.5  |
| ResNet-50 FPN | FCOS  |   -  | 37.0   |
| ResNet-50 AugFPN| FCOS |  -  | 37.9   |



Citations
------------

If you find AugFPN useful in your research, please consider citing:
```
@misc{guo2019augfpn,
    title={AugFPN: Improving Multi-scale Feature Learning for Object Detection},
    author={Chaoxu Guo and Bin Fan and Qian Zhang and Shiming Xiang and Chunhong Pan},
    year={2019},
    eprint={1912.05384},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

License
--------
This project is released under the [Apache 2.0 license](LICENSE)
