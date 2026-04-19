"""
perception_node.py — The main orchestrator for OmniGrasp's perception stack.

THIS NODE ORCHESTRATES THE FULL PIPELINE:

  Frame arrives → GDINO + OWL-ViT → Fusion → SAM2 → Depth → Kalman → Publish

Each stage is a separate module. This node wires them together,
manages the data flow, and handles the ROS2 communication.

TOPICS:
  Subscribes to:
    /camera/rgb_image      — RGB camera frames
    /camera/depth_image    — Depth maps
    /operator_command      — Text commands from dashboard

  Publishes to:
    /target_pose           — 6-DOF grasp target for RL controller
    /detection_viz         — Annotated image for dashboard
    /perception/diagnostics — Pipeline health status
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

from omnigrasp_perception.detectors.grounding_dino import GroundingDINODetector
from omnigrasp_perception.detectors.owl_vit import OWLViTDetector
from omnigrasp_perception.detectors.detection_fusion import DetectionFusion
from omnigrasp_perception.segmentation.sam2_segmentor import SAM2Segmentor
from omnigrasp_perception.geometry.camera_model import PinholeCamera, CameraIntrinsics
from omnigrasp_perception.geometry.frame_transforms import FrameTransformer
from omnigrasp_perception.geometry.grasp_pose_estimator import GraspPoseEstimator
from omnigrasp_perception.tracking.temporal_filter import TemporalFilter
from omnigrasp_perception.diagnostics import PerceptionDiagnostics


class PerceptionNode(Node):
    """
    Main perception node that orchestrates the full pipeline.

    The pipeline runs every time an RGB frame arrives:
    1. Detect objects with GDINO + OWL-ViT
    2. Fuse detections (IoU-based agreement)
    3. Segment with SAM 2
    4. Deproject to 3D using depth + camera model
    5. Smooth with Kalman filter
    6. Publish target pose + diagnostics
    """

    def __init__(self):
        super().__init__("perception_node")

        # ============================================================
        # SUBSCRIBERS
        # ============================================================
        self.rgb_sub = self.create_subscription(Image, "/camera/rgb_image", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera/depth_image", self.depth_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, "/operator_command", self.command_callback, 10
        )

        # ============================================================
        # PUBLISHERS
        # ============================================================
        self.pose_pub = self.create_publisher(PoseStamped, "/target_pose", 10)
        self.diag_pub = self.create_publisher(String, "/perception/diagnostics", 10)

        # ============================================================
        # PERCEPTION PIPELINE MODULES
        # Each module is a separate class — modular, testable, swappable
        # ============================================================

        # Stage 1: Detection
        self.gdino = GroundingDINODetector(use_mock=True)
        self.owlvit = OWLViTDetector(use_mock=True)
        self.fusion = DetectionFusion(iou_threshold=0.5)

        # Stage 2: Segmentation
        self.segmentor = SAM2Segmentor(use_mock=True)

        # Stage 3: 3D Geometry
        # Default camera intrinsics (will be updated from /camera/camera_info)
        intrinsics = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)
        self.camera = PinholeCamera(intrinsics)
        self.transformer = FrameTransformer()
        self.grasp_estimator = GraspPoseEstimator()

        # Stage 4: Temporal filtering
        self.tracker = TemporalFilter(dt=0.1)

        # Stage 5: Diagnostics
        self.diagnostics = PerceptionDiagnostics()

        # ============================================================
        # STATE
        # ============================================================
        self.current_command = ""
        self.latest_depth = None
        self.frame_count = 0

        self.get_logger().info("PerceptionNode initialized with full pipeline")

    # ================================================================
    # CALLBACKS
    # ================================================================

    def depth_callback(self, msg: Image):
        """Store the latest depth frame for use when RGB arrives."""
        # Convert ROS2 Image to numpy array
        depth = np.frombuffer(msg.data, dtype=np.float32)
        self.latest_depth = depth.reshape((msg.height, msg.width))

    def command_callback(self, msg: String):
        """Update the text command for detection."""
        self.current_command = msg.data
        self.get_logger().info(f"Command updated: '{self.current_command}'")

    def rgb_callback(self, msg: Image):
        """
        Main processing callback — runs the full pipeline on each frame.

        This is where everything comes together. Each step uses the
        output of the previous step, forming a pipeline.
        """
        self.frame_count += 1
        self.diagnostics.start_frame()

        # Convert ROS2 Image to numpy array
        image = np.frombuffer(msg.data, dtype=np.uint8)
        image = image.reshape((msg.height, msg.width, 3))

        # Skip if no command yet
        if not self.current_command:
            if self.frame_count % 50 == 1:
                self.get_logger().info("Waiting for operator command...")
            return

        # ──────────────────────────────────────────────────────────
        # STAGE 1: Multi-model detection
        # ──────────────────────────────────────────────────────────
        gdino_result = self.gdino.detect(image, self.current_command)
        owlvit_result = self.owlvit.detect(image, self.current_command)
        fused = self.fusion.fuse(gdino_result, owlvit_result)

        self.diagnostics.update_detection(
            detected=fused.detected,
            gdino_confidence=fused.gdino_confidence,
            owlvit_confidence=fused.owlvit_confidence,
            fusion_iou=fused.fusion_iou,
            fusion_status=fused.fusion_status,
        )

        if not fused.detected:
            self._publish_diagnostics()
            return

        # ──────────────────────────────────────────────────────────
        # STAGE 2: Instance segmentation
        # ──────────────────────────────────────────────────────────
        seg_result = self.segmentor.segment(image, fused.box)

        if not seg_result.valid:
            self._publish_diagnostics()
            return

        # ──────────────────────────────────────────────────────────
        # STAGE 3: Depth deprojection + grasp pose
        # ──────────────────────────────────────────────────────────
        if self.latest_depth is None:
            self._publish_diagnostics()
            return

        # Get depth at the mask centroid
        cx, cy = seg_result.centroid
        cx_int, cy_int = int(round(cx)), int(round(cy))

        # Bounds check
        if (
            cy_int < 0
            or cy_int >= self.latest_depth.shape[0]
            or cx_int < 0
            or cx_int >= self.latest_depth.shape[1]
        ):
            self._publish_diagnostics()
            return

        depth_value = float(self.latest_depth[cy_int, cx_int])
        self.diagnostics.update_depth(depth_value)

        if not self.diagnostics.current_state.depth_valid:
            self._publish_diagnostics()
            return

        # Deproject centroid to 3D (camera frame)
        point_camera = self.camera.deproject(cx, cy, depth_value)
        if point_camera is None:
            self._publish_diagnostics()
            return

        # Transform to robot frame
        point_robot = self.transformer.camera_to_robot_frame(point_camera)

        # Estimate grasp pose from point cloud
        points_3d = self.camera.deproject_mask_to_points(seg_result.mask, self.latest_depth)
        grasp_info = self.grasp_estimator.estimate(points_3d, seg_result.mask)

        # ──────────────────────────────────────────────────────────
        # STAGE 4: Temporal filtering (Kalman)
        # ──────────────────────────────────────────────────────────
        self.tracker.predict()
        filtered_position = self.tracker.update(point_robot, confidence=fused.confidence)

        innovation = self.tracker.get_innovation_magnitude()
        self.diagnostics.update_tracking(innovation)

        # ──────────────────────────────────────────────────────────
        # STAGE 5: Publish results
        # ──────────────────────────────────────────────────────────
        self._publish_pose(filtered_position, grasp_info)
        self._publish_diagnostics()

        # Log periodically
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count}: "
                f"pos=({filtered_position[0]:.3f}, {filtered_position[1]:.3f}, "
                f"{filtered_position[2]:.3f}) "
                f"conf={fused.confidence:.2f} "
                f"status={fused.fusion_status}"
            )

    # ================================================================
    # PUBLISHING HELPERS
    # ================================================================

    def _publish_pose(self, position: np.ndarray, grasp_info: dict = None):
        """Publish the filtered 6-DOF target pose."""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "robot_base"

        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])

        # Set orientation from grasp info if available
        if grasp_info and grasp_info.get("approach_direction") is not None:
            # TODO: Convert approach_direction to quaternion using scipy Rotation
            # For now, use identity quaternion (no rotation)
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0
        else:
            pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)

    def _publish_diagnostics(self):
        """Publish current diagnostics state."""
        state = self.diagnostics.end_frame()
        diag_msg = String()
        diag_msg.data = (
            f"status={state.status} "
            f"gdino={state.gdino_confidence:.2f} "
            f"owlvit={state.owlvit_confidence:.2f} "
            f"iou={state.fusion_iou:.2f} "
            f"depth_ok={state.depth_valid} "
            f"latency={state.inference_latency_ms:.1f}ms"
        )
        self.diag_pub.publish(diag_msg)


def main(args=None):
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
