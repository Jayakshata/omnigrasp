"""
perception_node.py — The main orchestrator for OmniGrasp's perception stack.

THIS IS YOUR FIRST ROS2 NODE. Here's what's happening:

A ROS2 node is a single process that participates in the ROS2 communication
network. It can subscribe to topics (receive data), publish to topics (send data),
and run callbacks (functions triggered when data arrives).

This node:
  - Subscribes to /camera/rgb_image (camera frames from Isaac Sim)
  - Subscribes to /camera/depth_image (depth data from Isaac Sim)
  - Subscribes to /operator_command (text from the web dashboard)
  - Publishes to /target_pose (3D grasp target for the RL controller)
  - Publishes to /perception/diagnostics (pipeline health)

For now, it's a skeleton that logs when messages arrive.
We'll add the actual perception pipeline (GDINO, OWL-ViT, SAM2, etc.) next.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped


class PerceptionNode(Node):
    """
    Main perception node that orchestrates the full pipeline.

    WHAT IS Node?
    rclpy.node.Node is the base class for all ROS2 Python nodes.
    By inheriting from it, our class gets all the ROS2 superpowers:
    creating publishers, subscribers, timers, services, etc.

    The string 'perception_node' passed to super().__init__ is the
    node's name — it shows up when you run 'ros2 node list'.
    """

    def __init__(self):
        super().__init__("perception_node")

        # ============================================================
        # SUBSCRIBERS
        # A subscriber listens on a topic and calls a function (callback)
        # every time a new message arrives on that topic.
        #
        # Parameters:
        #   - Message type (e.g., Image, String)
        #   - Topic name (e.g., '/camera/rgb_image')
        #   - Callback function (runs when message arrives)
        #   - Queue size (how many messages to buffer if we're slow)
        #     10 is standard — if 10+ messages pile up, oldest get dropped
        # ============================================================

        # Camera RGB image from Isaac Sim
        self.rgb_sub = self.create_subscription(Image, "/camera/rgb_image", self.rgb_callback, 10)

        # Camera depth image from Isaac Sim
        self.depth_sub = self.create_subscription(
            Image, "/camera/depth_image", self.depth_callback, 10
        )

        # Operator text command from web dashboard
        self.command_sub = self.create_subscription(
            String, "/operator_command", self.command_callback, 10
        )

        # ============================================================
        # PUBLISHERS
        # A publisher sends messages to a topic. Any node subscribed
        # to that topic will receive the message.
        #
        # Parameters:
        #   - Message type
        #   - Topic name
        #   - Queue size
        # ============================================================

        # 6-DOF target pose for the RL controller
        self.pose_pub = self.create_publisher(PoseStamped, "/target_pose", 10)

        # ============================================================
        # STATE VARIABLES
        # These track the current state of the perception pipeline.
        # ============================================================
        self.current_command = ""
        self.frame_count = 0

        self.get_logger().info("PerceptionNode initialized — waiting for data...")

    # ================================================================
    # CALLBACKS
    # These functions are called automatically by ROS2 when messages
    # arrive on the subscribed topics. Think of them as event handlers.
    # ================================================================

    def rgb_callback(self, msg: Image):
        """Called every time a new RGB image arrives from the camera."""
        self.frame_count += 1
        if self.frame_count % 30 == 1:  # Log every 30th frame to avoid spam
            self.get_logger().info(
                f"RGB frame #{self.frame_count}: "
                f"{msg.width}x{msg.height}, encoding={msg.encoding}"
            )

    def depth_callback(self, msg: Image):
        """Called every time a new depth image arrives from the camera."""
        pass  # Will process depth in the next iteration

    def command_callback(self, msg: String):
        """Called when the operator sends a new text command."""
        self.current_command = msg.data
        self.get_logger().info(f"New command received: '{self.current_command}'")


def main(args=None):
    """
    Entry point for the perception node.

    rclpy.init() — Initializes the ROS2 Python client library.
    rclpy.spin() — Keeps the node alive and processing callbacks.
                   Without spin(), the node would start and immediately exit.
    rclpy.shutdown() — Cleans up when the node is stopped (Ctrl+C).

    This init/spin/shutdown pattern is the same for EVERY ROS2 Python node.
    You'll see it in every node we write.
    """
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down perception node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
