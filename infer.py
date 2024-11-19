import cv2
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Define the color map for visualization
color_map = {
    0: (0, 0, 0),        # Background - Black
    1: (0, 255, 0),      # Pallet - Green
    2: (255, 0, 0)       # Cement Floor - Red
}

def prediction_to_vis(prediction):
    """
    Convert predicted mask to a visual format.
    """
    vis_shape = prediction.shape + (3,)
    vis = np.zeros(vis_shape, dtype=np.uint8)
    for class_id, color in color_map.items():
        vis[prediction == class_id] = color
    return vis

def preprocess_frame(frame, feature_extractor):
    """
    Preprocess a single video frame for the model.
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    encoded_inputs = feature_extractor(pil_image, return_tensors="pt")
    pixel_values = encoded_inputs['pixel_values']
    return pixel_values

import matplotlib.pyplot as plt

def infer_webcam(model, feature_extractor):
    """
    Perform real-time segmentation on webcam feed and display results using matplotlib.
    """
    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    model.eval()
    with torch.no_grad():
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(10, 10))
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Preprocess the frame
            pixel_values = preprocess_frame(frame, feature_extractor)
            pixel_values = pixel_values.to(next(model.parameters()).device)

            # Perform inference
            outputs = model(pixel_values)
            logits = outputs.logits

            # Resize the logits to match the frame size
            upsampled_logits = F.interpolate(
                logits,
                size=frame.shape[:2],  # (height, width)
                mode="bilinear",
                align_corners=False
            )
            predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

            # Convert the predicted mask to a visual format
            mask_vis = prediction_to_vis(predicted_mask)

            # Resize mask to match the frame
            mask_vis = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]))

            # Blend the original frame and the mask
            overlay = cv2.addWeighted(frame, 0.5, mask_vis, 0.5, 0)

            # Update the matplotlib plot
            ax.clear()
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            plt.draw()
            plt.pause(0.001)  # Pause to allow plot updates

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        plt.ioff()  # Disable interactive mode
        plt.show()

    cap.release()


# Initialize the feature extractor
feature_extractor = SegformerFeatureExtractor()

# Load your custom model checkpoint
checkpoint_path = "/home/kishore/peer_ros2/src/segformer/lightning_logs/version_7/checkpoints/epoch=9-step=1150.ckpt"
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    num_labels=3,  # Number of classes (background, pallet, cement floor)
    ignore_mismatched_sizes=True
)

# Load the checkpoint and extract state_dict
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint["state_dict"]
model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

print("Model loaded successfully from checkpoint!")

# Perform inference on webcam feed
infer_webcam(model, feature_extractor)
