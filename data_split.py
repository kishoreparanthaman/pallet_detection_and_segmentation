import os
import shutil
import random

# Paths
input_dir = "/home/kishore/peer_ros2/src/segformer/Balloons-22/train/"  # Replace with your dataset directory
output_dir = "/home/kishore/peer_ros2/src/segformer/data2"  # Replace with your desired output directory

# Create output directories for train, validation, and test splits
split_dirs = {
    "train": os.path.join(output_dir, "train"),
    "valid": os.path.join(output_dir, "valid"),
    "test": os.path.join(output_dir, "test"),
}

for split, path in split_dirs.items():
    os.makedirs(os.path.join(path, "images"), exist_ok=True)
    os.makedirs(os.path.join(path, "masks"), exist_ok=True)

# Get all image-mask pairs
image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
mask_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

# Ensure corresponding masks exist for all images
image_files = sorted(image_files)
mask_files = sorted(mask_files)
paired_files = [(img, img.replace(".jpg", ".png")) for img in image_files if img.replace(".jpg", ".png") in mask_files]

# Shuffle and split
random.seed(42)  # For reproducibility
random.shuffle(paired_files)

n_total = len(paired_files)
n_train = int(0.85 * n_total)
n_valid = int(0.10 * n_total)
n_test = n_total - n_train - n_valid

train_files = paired_files[:n_train]
valid_files = paired_files[n_train:n_train + n_valid]
test_files = paired_files[n_train + n_valid:]

# Function to copy files
def copy_files(file_list, split):
    for img_file, mask_file in file_list:
        shutil.copy(os.path.join(input_dir, img_file), os.path.join(split_dirs[split], "images", img_file))
        shutil.copy(os.path.join(input_dir, mask_file), os.path.join(split_dirs[split], "masks", mask_file))

# Copy files to respective directories
copy_files(train_files, "train")
copy_files(valid_files, "valid")
copy_files(test_files, "test")

print(f"Dataset split completed!")
print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
