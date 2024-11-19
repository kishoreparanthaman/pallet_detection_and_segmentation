import os
import cv2
import numpy as np
from PIL import Image

def create_mask_from_annotation(image_shape, annotation_file):
    """
    Create a segmentation mask from an annotation file.
    :param image_shape: Tuple (height, width) of the mask.
    :param annotation_file: Path to the annotation file.
    :return: A numpy array representing the segmentation mask.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    class_ids = set()  # To store encountered class IDs for debugging

    with open(annotation_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split()
            class_id = int(values[0])  # Extract class ID
            class_ids.add(class_id)
            coords = [float(v) for v in values[1:]]  # Extract coordinates
            
            # Convert normalized coordinates to pixel coordinates
            points = np.array(coords).reshape(-1, 2)
            points[:, 0] *= image_shape[1]  # Convert x (width)
            points[:, 1] *= image_shape[0]  # Convert y (height)
            points = points.astype(np.int32)
            
            # Increment class_id by 1 so that background is 0 and classes start from 1
            cv2.fillPoly(mask, [points], class_id + 1)
    
    print(f"Encountered class IDs in {annotation_file}: {class_ids}")
    return mask


def process_annotations(input_dir, output_dir, image_shape):
    """
    Process all annotation files in the input directory to create segmentation masks.
    :param input_dir: Directory containing annotation files.
    :param output_dir: Directory to save the generated masks.
    :param image_shape: Tuple (height, width) of the masks.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            annotation_file = os.path.join(input_dir, filename)
            mask = create_mask_from_annotation(image_shape, annotation_file)
            
            # Save the mask as a PNG file
            mask_filename = os.path.splitext(filename)[0] + ".png"
            mask_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(mask_path, mask)
            print(f"Saved mask: {mask_path}")


def check_mask_classes(mask_dir):
    """
    Check the unique class IDs in the generated masks.
    :param mask_dir: Directory containing the generated masks.
    """
    unique_values = set()
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):
            mask = np.array(Image.open(os.path.join(mask_dir, mask_file)))
            unique_values.update(np.unique(mask))
    print(f"Unique classes in masks: {unique_values}")


# Example Usage
input_dir = "/home/kishore/peer_ros2/src/yolo/data/train/images_auto_annotate_labels"  # Directory with annotation files
output_dir = "/home/kishore/peer_ros2/src/yolo/data/train/masks2"  # Directory to save masks
image_shape = (640, 640)  # Set the desired resolution for the masks

# Process the annotations to generate masks
process_annotations(input_dir, output_dir, image_shape)

# Verify the generated masks
check_mask_classes(output_dir)
