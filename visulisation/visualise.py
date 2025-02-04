import cv2
import matplotlib.pyplot as plt

def visualize_predictions(image, predictions, img_meta):
    """
    Overlay predictions onto the image.

    Args:
        image (numpy.ndarray): The original image as a NumPy array.
        predictions (list): List of predictions in the format [bbox, score, class].
        img_meta (dict): Metadata about the image (e.g., dimensions, scaling factors).

    Returns:
        None: Displays the image with overlays.
    """
    for bbox, score, cls in predictions:
        # Unpack bounding box
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Annotate with class and score
        label = f"{cls} ({score:.2f})"
        cv2.putText(image, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
