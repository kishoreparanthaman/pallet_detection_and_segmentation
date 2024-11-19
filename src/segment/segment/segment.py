import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

class SegformerSubscriber(Node):
    def __init__(self):
        super().__init__('segformer_subscriber')
        
        # Subscribe to the 'webcam_image' topic
        self.subscription = self.create_subscription(
            Image,
            'webcam_image',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        # Initialize CvBridge
        self.bridge = CvBridge()

        # Load the trained SegFormer model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            num_labels=3,  # Adjust according to your classes
            ignore_mismatched_sizes=True
        )

        # Load your custom checkpoint
        checkpoint_path = "/home/kishore/peer_ros2/src/segformer/lightning_logs/version_7/checkpoints/epoch=9-step=1150.ckpt"
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})

        # Move model to GPU if available
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        # Initialize the feature extractor
        self.feature_extractor = SegformerFeatureExtractor()

        # Define the color map for visualization
        self.color_map = {
            0: (0, 0, 0),        # Background - Black
            1: (0, 255, 0),      # Pallet - Green
            2: (255, 0, 0)       # Cement Floor - Red
        }

        self.get_logger().info("SegFormerSubscriber initialized successfully!")

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocess the frame for the model
        pixel_values = self.preprocess_frame(frame)

        # Perform inference
        with torch.no_grad():
            pixel_values = pixel_values.to(next(self.model.parameters()).device)
            outputs = self.model(pixel_values)
            logits = outputs.logits

            # Resize logits to match the input frame size
            upsampled_logits = F.interpolate(
                logits,
                size=frame.shape[:2],  # (height, width)
                mode="bilinear",
                align_corners=False
            )
            predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

        # Convert the predicted mask to a visual format
        mask_vis = self.prediction_to_vis(predicted_mask)

        # Resize mask to match the original frame size
        mask_vis = cv2.resize(mask_vis, (frame.shape[1], frame.shape[0]))

        # Blend the original frame with the segmentation mask
        overlay = cv2.addWeighted(frame, 0.5, mask_vis, 0.5, 0)

        # Display the annotated frame
        cv2.imshow('SegFormer Segmentation', overlay)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for the SegFormer model.
        """
        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        encoded_inputs = self.feature_extractor(pil_image, return_tensors="pt")
        return encoded_inputs['pixel_values']

    def prediction_to_vis(self, prediction):
        """
        Convert the predicted mask to a color visualization.
        """
        vis_shape = prediction.shape + (3,)
        vis = np.zeros(vis_shape, dtype=np.uint8)
        for class_id, color in self.color_map.items():
            vis[prediction == class_id] = color
        return vis


def main(args=None):
    rclpy.init(args=args)
    node = SegformerSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
