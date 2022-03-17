## Dataset Preparation

### Convert Lyft to KITTI
```bash
python src/tools/export_kitti.py \
    --lyft_dir data/lyft \
    --json_dir data/lyft/train_data \
    --get_all_detections False \
    --num_workers 2 \
    --store_dir data/lyft_kitti
```

### Generate Image List
```bash
python src/tools/create_sets_lyft.py \
    --data_path data/lyft_mini/label_2 \
    --val_size 0.1 \
    --output_path data
```

### Convert Lyft to COCO
```bash
python src/tools/lyft_to_coco.py \
    --data_path data/Lyft_KITTI/Store/ \
    --output_path data
```