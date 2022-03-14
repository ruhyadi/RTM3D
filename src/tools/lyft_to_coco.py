"""
Script for generating json file from Lyft Dataset
Created by: Didi Ruhyadi - ruhyadi.dr@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.eval.detection.utils import category_to_detection_name

from nuscenes.nuscenes import NuScenes


from pyquaternion import Quaternion
from sklearn.linear_model import QuantileRegressor

import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y

def generate(data_path, ):
    # PATH
    DATA_PATH = data_path
    OUT_PATH = DATA_PATH + 'annotations/'
    # TODO: get exact split
    SPLITS = {'train': 'train_data', 'val': 'train_data', 'test': 'test_data'}
    DEBUG = False

    # lyft categories 
    CATS = ['car', 'pedestrian', 'animal', 'other_vehicle', 'bus',
            'motorcycle', 'truck', 'emergency_vehicle', 'bicycle']

    # SENSOR ID = SENSOR + 1, see sensor.json from dataset
    SENSOR_ID = {
    'CAMERA_FRONT_LEFT': 1,
    'LIDAR_FRONT_LEFT': 2,
    'CAMERA_FRONT': 3,
    'LIDAR_TOP': 4,
    'CAMERA_BACK_LEFT': 5,
    'LIDAR_FRONT_RIGHT': 6,
    'CAMERA_BACK': 7,
    'CAMERA_BACK_RIGHT': 8,
    'CAMERA_FRONT_ZOOMED': 9,
    'CAMERA_FRONT_RIGHT': 10
    }

    USED_SENSOR = [
        'CAMERA_FRONT_LEFT',
        'CAMERA_FRONT_RIGHT',
        'CAMERA_FRONT',
        'CAMERA_BACK_LEFT',
        'CAMERA_BACK_RIGHT'
        ]
    
    CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}

    # ATTRIBUTE + 1 from ATTRIBUTE in json
    ATTRIBUTE_TO_ID = {
    '': 0,
    'object_action_lane_change_right': 1,
    'object_action_running': 2,
    'object_action_lane_change_left': 3,
    'object_action_parked': 4,
    'object_action_standing': 5,
    'object_action_right_turn': 6,
    'object_action_gliding_on_wheels': 7,
    'object_action_loss_of_control': 8,
    'object_action_u_turn': 9,
    'object_action_sitting': 10,
    'object_action_walking': 11,
    'object_action_stopped': 12,
    'object_action_left_turn': 13,
    'object_action_reversing': 14,
    'is_stationary': 15,
    'object_action_driving_straight_forward': 16,
    'object_action_abnormal_or_traffic_violation': 17,
    'object_action_other_motion': 18
    }
    
    # create output dir
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        JSON_PATH = DATA_PATH + f'{SPLITS[split]}'
        lyft = LyftDataset(data_path=DATA_PATH, json_path=JSON_PATH, verbose=True)
        out_path = OUT_PATH + f'{split}.json'
        categories_info = [{'name': CATS[i], 'id': i + 1} for i in range(len(CATS))]
        ret = {'images': [], 'annotations': [], 'categories': categories_info,
                'attributes': ATTRIBUTE_TO_ID}
        # set initial value for ret
        num_images = 0
        num_anns = 0
        num_videos = 0

        for sample in lyft.sample:
            scene_name = lyft.get('scene', sample['scene_token'])['name']
            # TODO: maybe add scene list to check scene name
            if sample['prev'] == '':
                print('scene_name:', scene_name)
                # frame ids for every sensor
                frame_ids = {k: 0 for k in sample['data']}
                # track ids for tracking box
                track_ids = {}

            for sensor_name in sample['data']:
                if sensor_name in USED_SENSOR:
                    image_token = sample['data'][sensor_name]
                    image_data = lyft.get('sample_data', image_token)
                    num_images += 1

                    # coordinate transform
                    sd_record = lyft.get('sample_data', image_token)
                    # calibrated sensor 
                    cs_record = lyft.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                    pose_record = lyft.get('ego_pose', sd_record['ego_pose_token'])
                    global_from_car = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']), inverse=False)
                    car_from_sensor = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']), inverse=False)
                    # transformation matrix
                    trans_matrix = np.dot(global_from_car, car_from_sensor)
                    # get boxes and camera instrinsic
                    _, boxes, camera_intrinsic = lyft.get_sample_data(image_token, box_vis_level=BoxVisibility.ANY)
                    # make calibration space
                    calib = np.eye(4, dtype=np.float32)
                    calib[:3, :3] = camera_intrinsic
                    # convert calib from 4x4 to 3x3
                    calib = calib[:3]
                    # add up frame ids from every sensor
                    frame_ids[sensor_name] += 1

                    # generate image information in COCO format
                    image_info = {
                        'id': num_images,
                        'file_name': image_data['filename'],
                        'calib': calib.tolist(),
                        'frame_id': frame_ids[sensor_name],
                        'sensor_id': SENSOR_ID[sensor_name],
                        'trans_matrix': trans_matrix.tolist(),
                        'width': sd_record['width'],
                        'height': sd_record['height'],
                        'pose_record_trans': pose_record['translation'],
                        'pose_record_rot': pose_record['rotation'],
                        'cs_record_trans': cs_record['translation'],
                        'cs_record_rot': cs_record['rotation']
                    }
                    ret['images'].append(image_info)
                    # annotation
                    anns = []
                    for box in boxes:
                        det_name = box.names
                        if det_name is None:
                            continue
                        # add up num annotations
                        num_anns += 1
                        v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
                        yaw = -np.arctan2(v[2], v[0])
                        box.translate(np.array([0, box.wlh[2]/2, 0]))
                        category_id = CAT_IDS[det_name]

                        amodel_center = project_to_image(np.array([box.center[0], box.center[1] - box.wlh[2] / 2, box.center[2]], np.float32).reshape(1, 3), calib)[0].tolist()
                        sample_ann = lyft.get('sample_annotation', box.token)
                        instance_token = sample_ann['instance_token']

                        if not (instance_token in track_ids):
                            track_ids[instance_token] = len(track_ids) + 1
                        attribute_tokens = sample_ann['attribute_tokens']
                        # get box attribute
                        attributes = [lyft.get('attribute', att_token)['name'] for att_token in attribute_tokens]
                        att = '' if len(attributes) == 0 else attributes[0]
                        if len(attributes) > 1:
                            print(attributes)
                            import pdb; pdb.set_trace()
                        track_id = track_ids[instance_token]
                        vel = lyft.box_velocity(box.token) # in global frame
                        vel = np.dot(np.linalg.inv(trans_matrix), np.array([vel[0], vel[1], vel[2], 0], np.float32)).tolist()

                        # instance information in COCO format
                        ann = {
                            'id': num_anns,
                            'image_id': num_images,
                            'category_id': category_id,
                            'dim': [box.wlh[2], box.wlh[0], box.wlh[1]],
                            'location': [box.center[0], box.center[1], box.center[2]],
                            'depth': box.center[2],
                            'occluded': 0,
                            'truncated': 0,
                            'rotation_y': yaw,
                            'amodel_center': amodel_center,
                            'iscrowd': 0,
                            'track_id': track_id,
                            'attributes': ATTRIBUTE_TO_ID[att],
                            'velocity': vel
                        }




def opt():
    parser = argparse.ArgumentParser(description='Generate JSON from Lyft Dataset')
    parser.add_argument('--data_path', type=str, default='data/lyft/', help='dataset path (lyft dir)')

def main():
    print('hello')

if __name__ == '__main__':
    main(opt)