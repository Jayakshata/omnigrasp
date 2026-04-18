"""
mock_camera_node.py — Publishes synthetic camera data for testing.

WHY THIS EXISTS:

Isaac Sim requires an NVIDIA GPU we don't have locally. But we still
need to test that our perception node correctly receives and processes
camera data. This node publishes fake RGB and depth images so we can
verify the entire ROS2 communication pipeline works.

This is a standard practice in robotics called "mock testing" or
"hardware-in-the-loop simulation." You replace the real hardware
(or simulator) with a software stand-in that produces data in the
exact same format.

WHAT IT PUBLISHES:
  /camera/rgb_image    — A solid-colored image with a rectangle drawn on it
                         (simulating an object the VLM should detect)
  /camera/depth_image  — A synthetic depth map with a known depth value
                         at the rectangle's location
  /operator_command    — A hardcoded text command for testing
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String


class MockCameraNode(Node):
    """
    Publishes synthetic camera frames at a fixed rate.

    TIMER CALLBACKS:
    Instead of subscribing to something, this node uses a TIMER.
    A timer calls a function at a fixed interval — in our case,
    every 0.1 seconds (10 Hz). This simulates a camera running
    at 10 frames per second.

    'self.create_timer(period, callback)' is the ROS2 way to do this.
    """

    def __init__(self):
        super().__init__("mock_camera_node")

        # Publishers for camera data
        self.rgb_pub = self.create_publisher(Image, "/camera/rgb_image", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth_image", 10)
        self.command_pub = self.create_publisher(String, "/operator_command", 10)

        # Image dimensions (standard VGA resolution)
        self.width = 640
        self.height = 480

        # Simulated object location (a rectangle representing "the object")
        # These coordinates define where our fake object sits in the image
        self.obj_x_min = 250
        self.obj_y_min = 180
        self.obj_x_max = 390
        self.obj_y_max = 300

        # Simulated depth of the object (0.5 meters from camera)
        self.obj_depth = 0.5

        # Frame counter
        self.frame_count = 0

        # Timer: calls publish_frames() every 0.1 seconds (10 Hz)
        # 10 Hz is a typical camera rate for robotic perception
        self.timer = self.create_timer(0.1, self.publish_frames)

        # Publish the operator command once after 1 second
        self.command_timer = self.create_timer(1.0, self.publish_command)
        self.command_sent = False

        self.get_logger().info(
            f"MockCameraNode started — publishing {self.width}x{self.height} " f"frames at 10 Hz"
        )

    def publish_frames(self):
        """Generate and publish synthetic RGB and depth images."""
        self.frame_count += 1
        now = self.get_clock().now().to_msg()

        # Create RGB image
        rgb_msg = self.create_rgb_image(now)
        self.rgb_pub.publish(rgb_msg)

        # Create depth image
        depth_msg = self.create_depth_image(now)
        self.depth_pub.publish(depth_msg)

        if self.frame_count % 50 == 0:
            self.get_logger().info(f"Published frame #{self.frame_count}")

    def create_rgb_image(self, timestamp) -> Image:
        """
        Create a synthetic RGB image with a colored rectangle.

        The image is a dark gray background with a bright red rectangle
        where our simulated object is. This gives the VLM something to
        detect when we integrate it.

        IMAGE FORMAT IN ROS2:
        ROS2 Image messages store pixels as a flat byte array.
        For an RGB image: [R0, G0, B0, R1, G1, B1, ...] — 3 bytes per pixel.
        'encoding' tells subscribers how to interpret the bytes.
        'step' is the number of bytes per row (width * 3 for RGB).
        """
        # Start with a dark gray background
        img = np.full((self.height, self.width, 3), 40, dtype=np.uint8)

        # Draw a red rectangle (our simulated object)
        img[self.obj_y_min : self.obj_y_max, self.obj_x_min : self.obj_x_max] = [200, 50, 50]

        # Build the ROS2 Image message
        msg = Image()
        msg.header = Header()
        msg.header.stamp = timestamp
        msg.header.frame_id = "camera_frame"
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "rgb8"  # 8-bit RGB, 3 channels
        msg.is_bigendian = False
        msg.step = self.width * 3  # bytes per row
        msg.data = img.tobytes()  # flatten numpy array to bytes

        return msg

    def create_depth_image(self, timestamp) -> Image:
        """
        Create a synthetic depth image.

        Depth images store distance-from-camera in meters as 32-bit floats.
        Background is at 2.0 meters, the object is at 0.5 meters.

        This is exactly how Isaac Sim's depth camera works — each pixel
        contains the distance to the nearest surface at that pixel.
        """
        # Background at 2.0 meters
        depth = np.full((self.height, self.width), 2.0, dtype=np.float32)

        # Object area at 0.5 meters (closer to camera)
        depth[self.obj_y_min : self.obj_y_max, self.obj_x_min : self.obj_x_max] = self.obj_depth

        msg = Image()
        msg.header = Header()
        msg.header.stamp = timestamp
        msg.header.frame_id = "camera_frame"
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "32FC1"  # 32-bit float, 1 channel
        msg.is_bigendian = False
        msg.step = self.width * 4  # 4 bytes per float32 pixel
        msg.data = depth.tobytes()

        return msg

    def publish_command(self):
        """Publish the operator command once."""
        if not self.command_sent:
            msg = String()
            msg.data = "pick up the red block"
            self.command_pub.publish(msg)
            self.get_logger().info(f"Published command: '{msg.data}'")
            self.command_sent = True


def main(args=None):
    rclpy.init(args=args)
    node = MockCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down mock camera...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
