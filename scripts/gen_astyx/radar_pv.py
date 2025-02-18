import os
import pickle
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

# Paths (Modify according to your dataset structure)
DATA_PATH = 'data/Astyx'
RADAR_SPLIT = 'radar_data'
OUT_PATH = 'radar_pv_astyx'
info_path = 'data/Astyx/astyx_infos.pkl'

MIN_DISTANCE = 0.1
MAX_DISTANCE = 100.

IMG_SHAPE = (1080, 1920) 

# Camera list for Astyx dataset
cam_keys = ['CAM_FRONT']


def load_radar_data(file_path):
    with open(file_path, 'rb') as f:
        radar_data = pickle.load(f)
    return radar_data


def map_pointcloud_to_image(pc, features, img_shape, cam_calib, cam_pose):
    """ Projects point cloud onto the image plane """
    pc = np.array(pc)  # Convert to NumPy array if needed

    # Convert from sensor frame → ego frame
    pc = np.dot(cam_pose['rotation'], pc.T).T + cam_pose['translation']

    # Convert from ego frame → camera frame
    pc = np.dot(cam_calib['rotation'], pc.T).T + cam_calib['translation']

    # Extract depth (z-coordinate in camera frame)
    depths = pc[:, 2]
    features = np.concatenate((depths[:, None], features), axis=1)

    # Project to image plane using camera intrinsics
    points = view_points(pc.T[:3, :], np.array(cam_calib['camera_intrinsic']), normalize=True)

    # Apply filtering (ensure points are within image bounds)
    mask = (depths > MIN_DISTANCE) & (depths < MAX_DISTANCE)
    mask &= (points[0, :] > 0) & (points[0, :] < img_shape[1])
    mask &= (points[1, :] > 0) & (points[1, :] < img_shape[0])
    
    return points[:, mask], features[mask]


def worker(info):
    """ Processes a single frame from the Astyx dataset """
    radar_file_name = os.path.split(info['radar_info']['filename'])[-1]
    radar_data = load_radar_data(os.path.join(DATA_PATH, RADAR_SPLIT, radar_file_name))

    points = radar_data['points']  # Extract radar point cloud
    features = radar_data['features']  # Extract radar features (e.g., intensity, velocity)

    radar_calib = info['radar_info']['calibrated_sensor']
    radar_pose = info['radar_info']['ego_pose']

    # Transform radar points to global frame
    points = np.dot(radar_pose['rotation'], points.T).T + radar_pose['translation']

    for cam_key in cam_keys:
        cam_calib = info['cam_info'][cam_key]['calibrated_sensor']
        cam_pose = info['cam_info'][cam_key]['ego_pose']

        # Project radar points to camera image
        pts_img, features_img = map_pointcloud_to_image(points, features, IMG_SHAPE, cam_calib, cam_pose)

        file_name = os.path.split(info['cam_info'][cam_key]['filename'])[-1]
        np.concatenate([pts_img[:2, :].T, features_img], axis=1).astype(np.float32).flatten().tofile(
            os.path.join(DATA_PATH, OUT_PATH, f'{file_name}.bin')
        )


if __name__ == '__main__':
    os.makedirs(os.path.join(DATA_PATH, OUT_PATH), exist_ok=True)
    infos = pickle.load(open(info_path, 'rb'))  # Load dataset metadata

    for info in tqdm(infos):
        worker(info)
