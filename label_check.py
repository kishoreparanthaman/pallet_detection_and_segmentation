import cv2
import os
import numpy as np

# Paths to your dataset
image_dir = '/home/kishore/peer_ros2/src/yolo/data/valid/images'
label_dir = '/home/kishore/peer_ros2/src/yolo/data/valid/images_auto_annotate_labels'
output_dir = '/home/kishore/peer_ros2/src/yolo/data/valid/segmented_data'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define class names and colors (adjust according to your dataset)
class_names = ['Cement Floor', 'Pallet']
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]

def visualize_labels(image_path, label_path):
    """Visualize polygon labels on the image with different colors."""
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Read the label file
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Iterate over each line to parse polygons
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        coordinates = list(map(float, parts[1:]))

        # Extract polygon points from the coordinates
        points = []
        for i in range(0, len(coordinates), 2):
            x = int(coordinates[i] * width)
            y = int(coordinates[i + 1] * height)
            points.append([x, y])

        # Convert the points to a NumPy array and reshape for polylines
        if len(points) > 2:
            points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

            # Choose a color based on the class ID
            color = colors[class_id % len(colors)]
            
            # Draw the polygon with the chosen color
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

            # Annotate with class label
            label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            cv2.putText(image, label, (points[0][0][0], points[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image

# Loop through all images in the directory and visualize their labels
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

        # Check if the label file exists
        if os.path.exists(label_path):
            annotated_image = visualize_labels(image_path, label_path)

            # Save the annotated image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, annotated_image)

            # Display the image
            cv2.imshow("Annotated Image", annotated_image)
            
            # Wait for user input to move to the next image
            print(f"Press any key to view the next image or 'q' to quit.")
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
