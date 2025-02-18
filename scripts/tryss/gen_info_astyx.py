import os
import json
import mmcv
import numpy as np
from tqdm import tqdm
from glob import glob

def generate_astyx_info_for_prediction(data_root):
    """
    Generates dataset info in NuScenes format for Astyx dataset (Prediction only).
    """
    infos = []

    # Get all available frames (assuming groundtruth files exist but are ignored)
    image_files = sorted(glob(os.path.join(data_root, 'camera_bfly/*.jpg')))
    
    for img_path in tqdm(image_files, desc="Processing frames for inference"):
        info = {}
        frame_id = os.path.basename(img_path).split('.')[0]

        # Load calibration info
        calib_file = os.path.join(data_root, 'calibration', f'{frame_id}.json')
        if not os.path.exists(calib_file):
            continue  # Skip if calibration file is missing
        
        with open(calib_file, 'r') as f:
            calib = json.load(f)

        # Basic info
        info['frame_id'] = frame_id

        # Camera info (assuming single front camera)
        cam_info = {
            'filename': f'camera_bfly/{frame_id}.jpg',
            'calibrated_sensor': calib['sensors'][1]['calib_data'],
        }
        info['camera'] = cam_info

        # Radar info
        radar_info = {
            'filename': f'radar_6455/{frame_id}.txt',
            'calibrated_sensor': calib['sensors'][0]['calib_data'],
        }
        info['radar'] = radar_info

        # Append to final list
        infos.append(info)

    return infos

def main():
    data_root = 'data/radar_dataset_astyx-main/dataset_astyx_demo'
    
    # Generate info only for prediction (test set)
    infos = generate_astyx_info_for_prediction(data_root)
    
    # Save as a pickle file
    output_path = os.path.join(data_root, 'astyx_infos_test.pkl')
    mmcv.dump(infos, output_path)
    print(f"✅ Astyx prediction info saved to: {output_path}")

if __name__ == '__main__':
    main()
