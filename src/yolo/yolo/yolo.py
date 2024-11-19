import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

class YOLOSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber')
        
        # Subscribe to the 'webcam_image' topic
        self.subscription = self.create_subscription(
            Image,
            'webcam_image',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        # Initialize CvBridge
        self.bridge = CvBridge()

        # Load the trained YOLO model
        self.model = YOLO('/home/kishore/peer_ros2/src/yolo/runs/detect/train19/weights/best.pt')

        # Initialize annotators
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Define a confidence threshold
        self.CONFIDENCE_THRESHOLD = 0.5

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Perform inference
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Initialize lists to hold filtered results
        filtered_boxes = []
        filtered_labels = []
        filtered_confidences = []

        # Filter detections based on confidence score
        for box, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            if confidence >= self.CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box
                class_name = f"Class {class_id}"
                print(f"Accepted: Class: {class_name}, Confidence: {confidence:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")
                
                # Append filtered data
                filtered_boxes.append(box)
                filtered_labels.append(class_name)
                filtered_confidences.append(confidence)

        # Convert filtered boxes back to a NumPy array
        if filtered_boxes:
            detections.xyxy = np.array(filtered_boxes)
            detections.labels = filtered_labels
            detections.confidence = np.array(filtered_confidences)
        else:
            # If no detections, set empty arrays
            detections.xyxy = np.empty((0, 4))
            detections.labels = []
            detections.confidence = np.empty(0)

        # Annotate the frame with bounding boxes and labels
        annotated_image = self.bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections)

        # Display the annotated frame
        cv2.imshow('YOLO Detection', annotated_image)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            rclpy.shutdown()

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
