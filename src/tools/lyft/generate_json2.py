"""
Script for convert lyft to coco keypoint format (json file)
"""

from __future__ import absolute_import, annotations
from __future__ import division
from __future__ import print_function

import os
from random import sample
from tqdm import tqdm
import numpy as np
import math

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.kitti import KittiDB
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, transform_matrix
from pyquaternion import Quaternion

from utils.ddd_utils import compute_box_3d, project_to_image

"""
Goals to generate
annotation = {
    'segmentation': [[0,0,0,0,0,0]],
    'num_keypoints':num_keypoints,
    'area':1,
    'iscrowd': 0,
    'keypoints': box_2d_as_point,
    'image_id': self.image_id,
    'bbox': self._bbox_to_coco_bbox(bbox),
    'category_id': cat_id,
    'id': int(len(self.ret['annotations']) + 1),
    'dim': dim,
    'rotation_y': rotation_y,
    'alpha': alpha,
    'location':location,
    'calib':calib_list
    }
"""

class LyftConverter:
    def __init__(
        self,
        lyft_path,
        output_path,
        cam_uses = ['CAM_FRONT']
        ) -> None:

        # directory
        self.LYFT_PATH = lyft_path
        self.JSON_PATH = os.path.join(self.LYFT_PATH, 'train_data')
        if not os.path.isdir(self.JSON_PATH):
            self.JSON_PATH = os.path.join(self.LYFT_PATH, 'data')
        self.OUTPUT_PATH = output_path
        if not os.path.isdir(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)
        self.CAM_USES = cam_uses # ['CAM_FONT', etc]

        # category (9 items)
        self.CATEGORIES = [
            'car', 'pedestrian', 
            'animal', 'other_vehicle', 
            'bus', 'motorcycle', 'truck', 
            'emergency_vehicle', 'bicycle']

        self.DET_CATS = [
            'car', 'pedestrian']

        self.cat_ids = {cat: i + 1 for i, cat in enumerate(self.categories)}

        self.cat_info = []
        for i, cat in enumerate(self.DET_CATS):
            self.cat_info.append({'name': cat, 'id': i+1})

    def generate(self):
        # lyft dataset
        self.lyft = LyftDataset(data_path=self.LYFT_PATH, json_path=self.JSON_PATH)

        samples = self.lyft.sample



def main():
    converter = LyftConverter('/content/lyft', '/content/lyft')
    converter.generate()
