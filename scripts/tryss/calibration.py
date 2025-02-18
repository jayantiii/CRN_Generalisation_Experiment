# K = [fx  0  cx]  - camera intrinsics
#     [0   fy cy]
#     [0   0   1]


#Astyx format:
# [R₃ₓ₃ | t₃ₓ₁]  # Rotation & Translation
# [  0  |  1  ]  # Homogeneous coordinates
# Identity matrix means:
# - No rotation (R = I)       -->
# - No translation (t = 0)
# T = [
#     [1, 0, 0, 0],  # First row: X-axis transform
#     [0, 1, 0, 0],  # Second row: Y-axis transform
#     [0, 0, 1, 0],  # Third row: Z-axis transform
#     [0, 0, 0, 1]   # Homogeneous coordinates
# ]

# class AstyxObject:  -- ground truth
#     """
#     Fields:
#     - center3d: [x,y,z] object center in meters
#     - classname: object type (Car, Bus, Person etc)
#     - dimension3d: [length, width, height] in meters
#     - label_certainty: annotation confidence (0-2)
#     - measured_by: sensor flags {camera,lidar,radar} 
#     - object_id: instance ID (-1 = not tracked)
#     - occlusion: visibility level (0=visible, 1=partial, 2=heavy)
#     - orientation_quat: [w,x,y,z] rotation quaternion
#     - score: detection confidence (-1 for ground truth)
#     """

# # Example object format:
# {
#     "center3d": [22.7, 3.1, 0.4],      # Position
#     "classname": "Car",                 # Category
#     "dimension3d": [3.5, 1.8, 1.5],     # Size
#     "label_certainty": 0,               # Annotation quality
#     "measured_by": {                    # Sensor visibility
#         "camera": 1,
#         "lidar": 1, 
#         "radar": 1
#     },
#     "object_id": -1,                    # No tracking
#     "occlusion": 0,                     # Fully visible
#     "orientation_quat": [0.98,0,0,0.19], # Orientation
#     "score": -1.0                        # Ground truth
# }

# The orientation_quat field represents the orientation of an object in quaternion form, which is a 
# mathematical representation used to describe 3D rotations.
# A quaternion is a four-element vector 
# [w,x,y,z] used to represent rotations in 3D space.
#Compared to rotation matrices (3×3), quaternions (4 values) are more efficient in memory and processing.

# Plan: Analyze Calibration vs Orientation Quaternions
# Purpose Differences:
# Calibration: Sensor-to-vehicle transform
# Orientation: Object rotation in space

# Usage Locations:
# Calibration: Point cloud preprocessing
# Orientation: Object detection/tracking

# import numpy as np
# from scipy.spatial.transform import Rotation

# def explain_quaternion(q):
#     """
#     q: [w,x,y,z] quaternion
#     returns: roll, pitch, yaw in degrees
#     """
#     # Create rotation object
#     r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # [x,y,z,w] format
    
#     # Get Euler angles
#     euler = r.as_euler('xyz', degrees=True)
    
#     return {
#         'quaternion': {
#             'w': q[0],  # Real component (cos(theta/2))
#             'x': q[1],  # X rotation axis component
#             'y': q[2],  # Y rotation axis component
#             'z': q[3]   # Z rotation axis component
#         },
#         'euler_angles': {
#             'roll': euler[0],   # Rotation around X
#             'pitch': euler[1],  # Rotation around Y
#             'yaw': euler[2]     # Rotation around Z
#         }
#     }

# # Example usage:
# q = [0.98, 0, 0, 0.19]  # [w,x,y,z]
# angles = explain_quaternion(q)
# # Results in approximately 22° yaw rotation
# # (0.98 = cos(11°), 0.19 = sin(11°))


# graph TD
#     A[Radar Points] --> B[Apply Calibration Matrix]
#     B --> C[Vehicle Reference Frame]
#     C --> D[Apply Object Orientation]
#     D --> E[Object Local Frame]

# def transform_pipeline(radar_points, calib_matrix, obj_quaternion):
#     """Complete transformation pipeline"""
#     # 1. Transform radar points to vehicle frame
#     points_homogeneous = np.hstack((radar_points[:, :3], np.ones((len(radar_points), 1))))
#     points_vehicle = (calib_matrix @ points_homogeneous.T).T[:, :3]
    
#     # 2. Transform points to object frame
#     obj_rotation = Quaternion(obj_quaternion).rotation_matrix
#     points_obj = (obj_rotation.T @ (points_vehicle - obj_center).T).T
    
#     return points_obj

# Dependencies:
# 1. Calibration must be applied first
# 2. Object orientation only valid in vehicle frame
# 3. Both needed for accurate 3D detection