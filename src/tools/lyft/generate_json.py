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

        # SAMPLES
        samples = self.lyft.sample

        # IMAGES TOKEN
        # TODO: change image cam front to variable
        image_tokens = [sample['data']['CAM_FRONT'] for sample in samples]
        filenames = [self.lyft.get('sample_data', image_token)['filename'].split('/')[1] \
                    for image_token in image_tokens]
        calib_tokens = [self.lyft.get('sample_data', image_token)['calibrated_sensor_token'] \
                    for image_token in image_tokens]
        calibs = [self.lyft.get('calibrated_sensor', calib_token) for calib_token in calib_tokens]
        calib_kitti = np.zeros((3, 4))
        cam_width = self.lyft.get('sample_data', image_tokens[0])['width']
        cam_height = self.lyft.get('sample_data', image_tokens[0])['height']
        self.imsize = (cam_width, cam_height)
        img_shape = [cam_height, cam_width, 3]

        # LIDAR PROCESSING
        lidar_tokens, velo_to_cam_rot, velo_to_cam_trans, r0_rect = self.lidar_processing(samples, image_tokens, calibs)

        # ANNOTATIONS TOKEN
        self.annotation_tokens = samples['ann']
        annotations = self.annotation_processing(
            lidar_tokens, 
            velo_to_cam_rot, 
            velo_to_cam_trans, 
            r0_rect, 
            calibs)
        CATS = annotations[0]
        BBOX = annotations[1]
        HWL = annotations[2]
        XYZ = annotations[3]
        YAW = annotations[4]

        # create result json
        result = {
            'images': [],
            'annotations': [],
            'categories': self.cat_info
        }

        # loop into filenames
        for id, (filename, calib, cat, bbox, hwl, xyz, yaw) in tqdm(enumerate(zip(filenames, calibs, CATS, BBOX, HWL, XYZ, YAW))):
            # change calib to kitti format 3 x 4 matrix
            calib_kitti[:3, :3] = calib

            # image info
            image_info = {
                'file_name': f'{filename}',
                'id': id,
                'calib': calib_kitti.tolist()
                }
            result['images'].append(image_info)

            # annotations
            cat_id = self.cat_ids[cat]
            num_keypoints = 0
            box_2d_as_point = [0] * 27
            calib_list = np.reshape(calib_kitti, (12)).tolist()

            if cat in self.DET_CATS:
                box_3d = compute_box_3d(hwl, xyz, yaw)
                box_2d_as_point, vis_num, pts_center = project_to_image(box_3d, calib_kitti, img_shape)
                box_2d_as_point = np.reshape(box_2d_as_point, (1, 27))
                box_2d_as_point = box_2d_as_point.tolist()[0]
                num_keypoints = vis_num
                alpha = yaw - math.atan2(pts_center[0, 0] - calib_kitti[0, 2], calib_kitti[0, 0])

            annotation = {
                'segmentation': [[0,0,0,0,0,0]],
                'num_keypoints':num_keypoints,
                'area':1,
                'iscrowd': 0,
                'keypoints': box_2d_as_point,
                'image_id': id,
                'bbox': self._bbox_to_coco_bbox(bbox),
                'category_id': cat_id,
                'id': int(len(result['annotations']) + 1),
                'dim': hwl,
                'rotation_y': yaw,
                'alpha': alpha,
                'location':xyz,
                'calib':calib_list
                }
            result['annotations'].append(annotation)

    def lidar_processing(self, samples, image_tokens, calibs):

        # lidar data
        lidar_tokens = [sample['data']['LIDAR_TOP'] for sample in samples]
        sd_records_lid = [self.lyft.get('sample_data', lidar_token) for lidar_token in lidar_tokens]
        cs_records_lid = [self.lyft.get('calibrated_sensor', x['calibrated_sensor_token']) for x in sd_records_lid]
        ego_records_lid = [self.lyft.get('ego_pose', x['ego_pose_token']) for x in sd_records_lid]

        # cam data
        sd_records_cam = [self.lyft.get('sample_data', token) for token in image_tokens]
        cs_records_cam = calibs
        ego_records_cam = [self.lyft.get("ego_pose", x["ego_pose_token"]) for x in sd_records_cam]
        
        lid_to_ego = [transform_matrix(cs_record_lid["translation"], Quaternion(cs_record_lid["rotation"]), inverse=False) \
            for cs_record_lid in cs_records_lid]
        lid_ego_to_world = [transform_matrix(ego_record_lid["translation"], Quaternion(ego_record_lid["rotation"]), inverse=False) \
            for ego_record_lid in ego_records_lid]
        world_to_cam_ego = [transform_matrix(ego_record_cam["translation"], Quaternion(ego_record_cam["rotation"]), inverse=True) \
            for ego_record_cam in ego_records_cam]
        ego_to_cam = [transform_matrix(cs_record_cam["translation"], Quaternion(cs_record_cam["rotation"]), inverse=True) \
            for cs_record_cam in cs_records_cam]
        velo_to_cam = [np.dot(A, np.dot(B, np.dot(C, D))) \
            for A, B, C, D in zip(ego_to_cam, world_to_cam_ego, lid_ego_to_world, lid_to_ego)]

        # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi)
        velo_to_cam_kitti = [np.dot(_velo_to_cam, kitti_to_nu_lidar.transformation_matrix) \
            for _velo_to_cam in velo_to_cam]

        # Currently not used.
        imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
        r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

        # Create KITTI style transforms.
        velo_to_cam_rot = [_velo_to_cam_kitti[:3, :3] for _velo_to_cam_kitti in velo_to_cam_kitti]
        velo_to_cam_trans = [_velo_to_cam_kitti[:3, 3] for _velo_to_cam_kitti in velo_to_cam_kitti]

        return lidar_tokens, velo_to_cam_rot, velo_to_cam_trans, r0_rect

    def annotation_processing(
        self, 
        lidar_tokens, 
        rot_matrixs, 
        trans_matrixs,
        r0_rect,
        calibs
        ):
        CATS = []
        BBOX = []
        HWL = []
        XYZ = []
        YAW = []

        for sample_annot_token, lidar_token, rot_mtx, trans_mtx, rect, calib in zip(self.annotation_tokens, lidar_tokens, rot_matrixs, trans_matrixs, r0_rect, calibs):
            sample_annot = self.lyft.get('sample_annotation', sample_annot_token)
            _, box_lidar_nusc, _ = self.lyft_ds.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE, selected_anntokens=[sample_annot_token])
            box_lidar_nusc = box_lidar_nusc[0]

            truncated = 0.0
            occluded = 0.0
            detection_name = sample_annot['category_name']
            CATS.append(detection_name)

            # Convert from nuScenes to KITTI box format.
            box_cam_kitti = KittiDB.box_nuscenes_to_kitti(box_lidar_nusc, Quaternion(matrix=rot_mtx), trans_mtx, rect)
            box_cam_kitti.score = 0

            # hwl
            hwl = [box_cam_kitti.wlh[2], box_cam_kitti.wlh[0], box_cam_kitti.wlh[1]]
            HWL.append(hwl)

            # xyz
            xyz = [box_cam_kitti.center[0], box_cam_kitti.center[1], box_cam_kitti.center[2]]
            XYZ.append(xyz)

            # Convert quaternion to yaw angle.
            v = np.dot(box_cam_kitti.rotation_matrix, np.array([1, 0, 0]))
            yaw = -np.arctan2(v[2], v[0])
            YAW.append(yaw)

            # Project 3d box to 2d box in image, ignore box if it does not fall inside.
            calib_kitti = np.zeros((3, 4))
            calib_kitti[:3, :3] = calib
            bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, calib_kitti, imsize=self.imsize)
            if bbox_2d is None:
                continue
            BBOX.append(bbox_2d)

        return [CATS, BBOX, HWL, XYZ, YAW]

    def _bbox_to_coco_bbox(self, bbox):
        return [(bbox[0]), (bbox[1]),
                (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def main():
    converter = LyftConverter('/content/lyft', '/content/lyft')
    converter.generate()
