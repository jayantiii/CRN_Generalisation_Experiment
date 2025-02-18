# Plan: Analyze Image Configuration Requirements
# Check if normalization needed for Astyx
# Compare image formats
# Update configuration if needed
# Analysis Steps:
# Load Astyx image
# Calculate statistics
# Update config

import numpy as np
from PIL import Image
import glob
import os

def calculate_image_stats():
    """Calculate mean and std for Astyx dataset"""
    img_dir = 'data/astyx/camera_frontcenter/'
    imgs = []
    
    # Load all images
    for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
        img = np.array(Image.open(img_path))
        imgs.append(img)
    
    # Calculate stats
    imgs = np.stack(imgs)
    mean = imgs.mean(axis=(0,1,2))/255.0
    std = imgs.std(axis=(0,1,2))/255.0
    
    print(f"Mean: {mean*255}")
    print(f"Std: {std*255}")
    
if __name__ == '__main__':
    calculate_image_stats()