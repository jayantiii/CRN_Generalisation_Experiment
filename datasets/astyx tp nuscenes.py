import os
import json
import numpy as np
from glob import glob

astyx_data_path = "data/radar_dataset_astyx-main/dataset_astyx_demo"

def convert_astyx_to_nuscenes_format(astyx_data_path):
    """Convert Astyx dataset to NuScenes format"""
    
    # Initialize NuScenes format structure
    nusc_format = {
        "info": [],
        "samples": [],
        "sweeps": [],
        "annotations": []
    }
    
    radar_path = os.path.join(astyx_data_path, 'radar_hires1')
    camera_path = os.path.join(astyx_data_path, 'camera_bfly')
    label_path = os.path.join(astyx_data_path, 'groundtruth_obj3d')
    
    print(f"Checking paths exist:")
    print(f"Radar: {os.path.exists(radar_path)}")
    print(f"Camera: {os.path.exists(camera_path)}")
    print(f"Labels: {os.path.exists(label_path)}")
    
    radar_files = sorted(glob(os.path.join(radar_path, '*.txt')))
    camera_files = sorted(glob(os.path.join(camera_path, '*.png')))
    label_files = sorted(glob(os.path.join(label_path, '*.json')))
    
    print(f"File counts:")
    print(f"Radar files: {len(radar_files)}")
    print(f"Camera files: {len(camera_files)}")
    print(f"Label files: {len(label_files)}")
    
    if not (radar_files and camera_files and label_files):
        raise ValueError("No files found in one or more directories")
    
    
    for idx, (radar_file, camera_file, label_file) in enumerate(zip(radar_files, camera_files, label_files)):
        # Create sample token
        print("in for loop")
        token = f"astyx_{idx:06d}"
        
        # Read label file
        with open(label_file, 'r') as f:
            label_data = json.load(f)
        
        # Create sample
        sample = {
            "token": token,
            "timestamp": idx,
            "scene_token": "astyx_scene",
            "data": {
                "RADAR_FRONT": radar_file,
                "CAM_FRONT": camera_file
            },
            "anns": []
        }

        
        # Convert annotations
        for obj in label_data['objects']:
            x, y, z = obj['center3d']
            # Extract dimensions
            l, w, h = obj['dimension3d']
            # Extract quaternion
            qw, qx, qy, qz = obj['orientation_quat']
            annotation = {
                "token": f"{token}_ann_{len(sample['anns'])}",
                "sample_token": token,
                "category_name": obj['classname'],
                "translation": [x, y, z],
                "size": [l, w, h],
                'visibility': 4 - obj['occlusion'],  # Convert occlusion to visibility
                "rotation": [qw, qx, qy, qz],
                "velocity": obj.get('velocity', [0, 0]),
                "num_pts": -1
            }
            sample['anns'].append(annotation['token'])
            nusc_format['annotations'].append(annotation)
        
        nusc_format['samples'].append(sample)
    
    # Save converted format
    output_path = os.path.join(astyx_data_path, 'nuscenes_format')
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'astyx_infos_test.json'), 'w') as f:
        json.dump(nusc_format, f, indent=2)
        print(f"Converted data saved to {os.path.join(output_path, 'astyx_infos_test.json')}")
    
    return nusc_format

convert_astyx_to_nuscenes_format(astyx_data_path)
