# RTM3D

## Installation

### 1. Install Conda Env
### 2. Install PyTorch and Torchvision
### 3. Install Requirements

### 4. Compile Deformable Convolution (DCNv2)

Clone DCNv2 for PyTorch > 1.8 from [ruhyadi/DCNv2_18](https://github.com/ruhyadi/DCNv2_18)
```
cd src/lib/models/networks
rm -rf DCNv2
git clone https://github.com/ruhyadi/DCNv2_18 ./DCNv2
```
Build DCNv2 from scrach
```
cd DCNv2
python setup.py build develop
```

### 5. Compile IoU3D
Clone IoU3D from [ruhyadi/iou3d](https://github.com/ruhyadi/iou3d)
```
cd src/lib/utils
rm -rf iou3d
git clone https://github.com/ruhyadi/iou3d ./iou3d
```
Build IoU3D from scract
```
cd iou3d
python setup.py install
```
Failed on PyTorch 1.9.0, but can inference model.

## Inference

```
python ./src/faster.py \
    --demo ./demo_kitti_format/data/kitti/image \
    --calib_dir ./demo_kitti_format/data/kitti/calib/ \
    --load_model ./weights/model_res18_1.pth \
    --gpus 0 \
    --vis \
    --arch res_18
```