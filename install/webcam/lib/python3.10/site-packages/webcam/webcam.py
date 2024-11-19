import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam')
        self.publisher_ = self.create_publisher(Image, 'webcam_image', 10)
        self.timer = self.create_timer(0.1, self.publish_image)  # Publish at 10Hz
        self.cap = cv2.VideoCapture(0)  # Use the default camera
        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().error("Could not open webcam")
            rclpy.shutdown()

    def publish_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image")
            return

        # Convert the frame to a ROS2 Image message
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(ros_image)
        self.get_logger().info('Publishing webcam image')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
