import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

class YOLOSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber')

        # Define QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Subscribe to the '/robot1/zed2i/left/image_rect_color' topic
        self.subscription = self.create_subscription(
            Image,
            '/robot1/zed2i/left/image_rect_color',
            self.listener_callback,
            qos_profile)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Load YOLO model
        self.model = YOLO('/home/kishore/peer_ros2/src/yolo/runs/detect/train20/weights/best.pt')

        # Initialize annotators
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Define a confidence threshold
        self.CONFIDENCE_THRESHOLD = 0.6

    def listener_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize the frame to the model's expected input size (640x640)
            resized_frame = cv2.resize(frame, (640, 640))

            # Perform YOLO inference
            results = self.model(resized_frame)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Filter detections based on confidence
            detections = self.filter_detections(detections)

            # Annotate frame with detections
            annotated_image = self.bounding_box_annotator.annotate(scene=resized_frame, detections=detections)
            annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections)

            # Display the annotated frame
            cv2.imshow('YOLO Detection', annotated_image)
            if cv2.waitKey(1) & 0xFF == 27:
                self.get_logger().info("Shutting down...")
                cv2.destroyAllWindows()
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

    def filter_detections(self, detections):
        filtered_boxes = []
        filtered_labels = []
        filtered_confidences = []

        for box, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            if confidence >= self.CONFIDENCE_THRESHOLD:
                filtered_boxes.append(box)
                filtered_labels.append(f"Class {class_id}")
                filtered_confidences.append(confidence)

        if filtered_boxes:
            detections.xyxy = np.array(filtered_boxes)
            detections.labels = filtered_labels
            detections.confidence = np.array(filtered_confidences)
        else:
            detections.xyxy = np.empty((0, 4))
            detections.labels = []
            detections.confidence = np.empty(0)

        return detections


def main(args=None):
    rclpy.init(args=args)
    node = YOLOSubscriber()
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
