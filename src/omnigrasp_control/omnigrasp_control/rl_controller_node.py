"""
rl_controller_node.py — Receives target poses and outputs joint commands.

HOW THIS NODE FITS IN THE PIPELINE:

    Perception Stack                    This Node                     Isaac Sim
    ┌─────────────┐              ┌──────────────────┐              ┌──────────┐
    │ VLM + Depth │──/target_pose──>│ RL Controller  │──/joint_commands──>│ Motors   │
    └─────────────┘              │                  │              └──────────┘
                                 │  joint_states ◄──────────────────── Encoders │
                                 └──────────────────┘

In Week 3, we'll load a real PPO-trained policy here.
For now, this node demonstrates the ROS2 communication pattern
and uses a simple proportional controller as a placeholder.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class RLControllerNode(Node):
    """
    Control node that converts target poses into joint commands.

    WHAT THIS NODE DOES:
    1. Subscribes to /target_pose (from perception) — WHERE to go
    2. Subscribes to /joint_states (from Isaac Sim) — WHERE we are now
    3. Queries the RL policy (placeholder for now)
    4. Publishes /joint_commands — HOW to move
    """

    def __init__(self):
        super().__init__("rl_controller_node")

        # Subscribe to the target pose from perception
        self.target_sub = self.create_subscription(
            PoseStamped, "/target_pose", self.target_callback, 10
        )

        # Subscribe to current joint states from the robot
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

        # Publish joint commands to the robot
        self.command_pub = self.create_publisher(JointTrajectory, "/joint_commands", 10)

        # State tracking
        self.current_target = None
        self.current_joint_state = None

        # Franka Panda has 7 joints
        self.joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        self.get_logger().info("RLControllerNode initialized — waiting for target pose...")

    def target_callback(self, msg: PoseStamped):
        """Store the latest target pose from the perception stack."""
        self.current_target = msg
        self.get_logger().info(
            f"Target received: ({msg.pose.position.x:.3f}, "
            f"{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
        )

    def joint_state_callback(self, msg: JointState):
        """
        Called when new joint states arrive from the robot.
        This is where the control loop runs — every time we know
        where the robot IS, we compute where it should GO.
        """
        self.current_joint_state = msg

        if self.current_target is None:
            return  # No target yet, nothing to do

        # In Week 3, this becomes: action = policy.predict(observation)
        # For now, publish a zero-velocity command as a placeholder
        command = self.compute_command()
        self.command_pub.publish(command)

    def compute_command(self) -> JointTrajectory:
        """
        Placeholder for the RL policy.

        In Week 3, this method will:
        1. Construct an observation vector from joint_state + target_pose
        2. Pass it through the trained PPO policy network
        3. Return the policy's action as joint velocities

        For now, it returns a zero command (robot stays still).
        This lets us test the full communication pipeline without
        needing a trained policy.
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        # 7 zeros — one for each joint velocity
        point.velocities = [0.0] * 7
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 0.1 seconds

        trajectory.points = [point]
        return trajectory


def main(args=None):
    """Entry point — same init/spin/shutdown pattern as every ROS2 node."""
    rclpy.init(args=args)
    node = RLControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down controller node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
