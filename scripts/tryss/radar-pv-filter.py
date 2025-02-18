

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Define file paths
bin_file = "./data/radar_pv_filter/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg.bin"
image_file = "./data/nuScenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg"

# Load the binary file and reshape it correctly
data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 7)  # (x, y) + 5 additional features

print(f"Loaded Data Shape: {data.shape}")
print("First 5 Points:\n", data[:5])

# Extract pixel coordinates
x, y = data[:, 0], data[:, 1]

# Load the corresponding image
image = cv2.imread(image_file)
if image is None:
    raise FileNotFoundError(f"Image file not found: {image_file}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB (for matplotlib)

# Plot the image with overlaid radar points
plt.figure(figsize=(10, 5))
plt.imshow(image)
plt.scatter(x, y, s=5, c='red', alpha=0.7)  # Overlay radar points in red
plt.title("Projected Radar Points on Image")
plt.axis("off")  # Hide axes for a cleaner view
plt.show()
