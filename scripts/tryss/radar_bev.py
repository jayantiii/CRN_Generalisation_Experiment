import numpy as np

file_path = "data/nuScenes/radar_bev_filter/sample_file.pcd.bin"  # Update the path

# Load the binary file
points = np.fromfile(file_path, dtype=np.float32)

# Reshape it to match the number of saved fields
num_fields = 7  # x, y, z, rcs, vx_comp, vy_comp, sweep_id
points = points.reshape(-1, num_fields)

print("Loaded points shape:", points.shape)  # (N, 7)
print("First 5 points:\n", points[:5])


import open3d as o3d

# Convert to Open3D format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use x, y, z

# Visualize
o3d.visualization.draw_geometries([pcd])
