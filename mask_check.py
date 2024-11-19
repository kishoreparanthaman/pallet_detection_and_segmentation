import cv2
import numpy as np

# Load the grayscale mask
mask_path = "/home/kishore/peer_ros2/src/segformer/Balloons-22/train/1_jpg.rf.9ad759c5d8cc6fbeb7cff152a436e2b4.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Define color mapping for each class
color_map = {
    0: (0, 0, 0),       # Background (black)
    1: (0, 255, 0),     # Class 1 (green)
    2: (0, 0, 255)      # Class 2 (red)
}

# Create a color image with the same height and width as the mask
color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

# Apply the color mapping
for class_id, color in color_map.items():
    color_mask[mask == class_id] = color

# Display the color-coded mask
cv2.imshow("Color Mask", color_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
