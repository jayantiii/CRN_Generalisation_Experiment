import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_bev(bev_map, predictions, bev_shape, resolution, offset):
    """
    Visualize predictions on the BEV map.

    Args:
        bev_map (numpy.ndarray): A blank or prefilled BEV map (e.g., radar points).
        predictions (list): List of predictions in BEV coordinates [bbox, score, class].
        bev_shape (tuple): Shape of the BEV grid (height, width).
        resolution (float): Resolution of the BEV grid (meters per pixel).
        offset (tuple): Offset to align BEV coordinates (e.g., min_x, min_y).

    Returns:
        None: Displays the BEV map with predictions.
    """
    bev_map = np.zeros(bev_shape, dtype=np.uint8)  # Blank BEV map

    # Overlay radar points or other features onto the BEV map
    for bbox, score, cls in predictions:
        x_min, y_min, x_max, y_max = bbox

        # Convert BEV coordinates to grid indices
        x_min_idx = int((x_min - offset[0]) / resolution)
        y_min_idx = int((y_min - offset[1]) / resolution)
        x_max_idx = int((x_max - offset[0]) / resolution)
        y_max_idx = int((y_max - offset[1]) / resolution)

        # Draw the bounding box
        cv2.rectangle(bev_map, (x_min_idx, y_min_idx), (x_max_idx, y_max_idx), (255, 255, 255), 1)

        # Annotate with class and score
        label = f"{cls} ({score:.2f})"
        cv2.putText(bev_map, label, (x_min_idx, y_min_idx - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the BEV map
    plt.figure(figsize=(10, 10))
    plt.imshow(bev_map, cmap="gray")
    plt.title("BEV Visualization with Predictions")
    plt.axis("off")
    plt.show()


    def bev_to_image_projection(bev_bboxes, projection_matrix):
    """
    Project BEV bounding boxes back to image space.

    Args:
        bev_bboxes (list): List of bounding boxes in BEV space [(x_min, y_min, x_max, y_max)].
        projection_matrix (numpy.ndarray): Camera projection matrix (3x4).

    Returns:
        list: List of bounding boxes in image coordinates [(x_min, y_min, x_max, y_max)].
    """
    image_bboxes = []

    for bbox in bev_bboxes:
        x_min, y_min, x_max, y_max = bbox

        # Convert BEV coordinates to 3D world coordinates (add z=0 for ground level)
        world_coords = np.array([
            [x_min, y_min, 0, 1],
            [x_max, y_max, 0, 1]
        ]).T  # Shape: (4, 2)

        # Project to image coordinates
        image_coords = projection_matrix @ world_coords  # Shape: (3, 2)
        image_coords /= image_coords[2]  # Normalize by z

        # Extract pixel coordinates
        x_min_img, y_min_img = image_coords[0, 0], image_coords[1, 0]
        x_max_img, y_max_img = image_coords[0, 1], image_coords[1, 1]

        image_bboxes.append((x_min_img, y_min_img, x_max_img, y_max_img))

    return image_bboxes

