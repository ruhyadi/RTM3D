# RTM3D

## Installation

### Conda Environment Development
1. Crete conda environment
```bash
conda create -n rtm3d python=3.8 numpy
conda activate rtm3d
```
2. Install PyTorch and Torchvision
```bash
pip install torch==1.4.0 torchvision==0.5.0
```
3. Install depedencies from `requirements.txt`
```bash
cd rtm3d
pip install -r requirements.txt
```
4. Compile Deformable Convolution (DCNv2)
```bash
cd src/lib/models/networks
./make.sh
```
5. Compile IoU3D
```bash
cd src/lib/utils/iou3d
python setup.py install
```

### Conda Environment Inference
1. Crete conda environment
```bash
conda create -n rtm3d python=3.8 numpy
conda activate rtm3d
```
2. Install PyTorch and Torchvision
```bash
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
3. Install depedencies from `requirements.txt`
```bash
cd rtm3d
pip install -r requirements.txt
```
4. Compile Deformable Convolution (DCNv2). 
Clone DCNv2 for PyTorch > 1.8 from [ruhyadi/DCNv2_18](https://github.com/ruhyadi/DCNv2_18)
```bash
cd src/lib/models/networks
rm -rf DCNv2
git clone https://github.com/ruhyadi/DCNv2_18 ./DCNv2

cd DCNv2
python setup.py build develop
```

### Docker
Docker image can be build with:
```bash
docker build -t username/rtm3d:latest .
```

Or you can pull docker image (**recommend**) with:
```bash
docker pull -t ruhyadi/rtm3d:latest
```

Then you can run docker container in interactive mode with:
```bash
./runDocker.sh
```

## Dataset Preparation
**Works under progress**

## Inference
Download pretrained model. Refer to [README-OLD.md](README-OLD.md)
```bash
cd /weights
./download_pretrained.sh
```
Run demo inference with:
```bash
python ./src/faster.py \
    --demo ./demo_kitti_format/data/kitti/image \
    --calib_dir ./demo_kitti_format/data/kitti/calib/ \
    --load_model ./weights/model_res18_1.pth \
    --gpus 0 \
    --arch res_18
```

## Todos
- [x] Install development
- [x] Run inference (on inference env)
- [ ] Train KITTI dataset

## Note
- Development using PyTorch 1.4.0. Inference can use PyTorch > 1.4.0 (1.8.1)
- 