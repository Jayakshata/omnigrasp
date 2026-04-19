"""
evaluate_perception.py — Quantitative evaluation of the perception pipeline.

WHAT THIS DOES:

Runs the perception pipeline on synthetic test data with KNOWN
ground truth positions, and measures:
1. Detection IoU (predicted box vs ground truth box)
2. 3D Localisation Error (predicted position vs actual position in mm)
3. Fusion Agreement Rate (% of frames where both models agree)
4. Detection Recall (% of frames where object is detected)
5. Inference Latency (ms per frame)

WHY THIS MATTERS:

Most student projects show a demo video and say "it works."
You show NUMBERS. A table with IoU, 3D error, and latency
is the perception engineer's equivalent of a green CI badge.
It's the first thing a hiring manager looks for.

USAGE:
    python3 -m omnigrasp_perception.eval.evaluate_perception
"""

import time

import numpy as np


def create_test_scene(
    width: int = 640,
    height: int = 480,
    obj_x_min: int = 250,
    obj_y_min: int = 180,
    obj_x_max: int = 390,
    obj_y_max: int = 300,
    obj_depth: float = 0.5,
):
    """
    Create a synthetic test scene with known ground truth.

    Returns:
        image: RGB image with a colored rectangle (the object)
        depth: Depth map with known depth at the object location
        ground_truth: Dictionary with all known values
    """
    # RGB image: dark background with colored rectangle
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    image[obj_y_min:obj_y_max, obj_x_min:obj_x_max] = [200, 50, 50]

    # Depth image: background at 2.0m, object at obj_depth
    depth = np.full((height, width), 2.0, dtype=np.float32)
    depth[obj_y_min:obj_y_max, obj_x_min:obj_x_max] = obj_depth

    ground_truth = {
        "box": np.array([float(obj_x_min), float(obj_y_min), float(obj_x_max), float(obj_y_max)]),
        "centroid_px": (
            (obj_x_min + obj_x_max) / 2.0,
            (obj_y_min + obj_y_max) / 2.0,
        ),
        "depth": obj_depth,
        # 3D position calculated using our camera model
        # fx=525, fy=525, cx=320, cy=240
        "position_3d": np.array(
            [
                ((obj_x_min + obj_x_max) / 2.0 - 320) * obj_depth / 525.0,
                ((obj_y_min + obj_y_max) / 2.0 - 240) * obj_depth / 525.0,
                obj_depth,
            ]
        ),
    }

    return image, depth, ground_truth


def run_evaluation(num_frames: int = 100):
    """
    Run the full perception pipeline evaluation.

    Creates synthetic test data, runs each pipeline stage,
    and collects metrics.
    """
    from omnigrasp_perception.detectors.grounding_dino import GroundingDINODetector
    from omnigrasp_perception.detectors.owl_vit import OWLViTDetector
    from omnigrasp_perception.detectors.detection_fusion import DetectionFusion
    from omnigrasp_perception.segmentation.sam2_segmentor import SAM2Segmentor
    from omnigrasp_perception.geometry.camera_model import PinholeCamera, CameraIntrinsics
    from omnigrasp_perception.geometry.frame_transforms import FrameTransformer
    from omnigrasp_perception.tracking.temporal_filter import TemporalFilter

    # Initialize pipeline components
    gdino = GroundingDINODetector(use_mock=True)
    owlvit = OWLViTDetector(use_mock=True)
    fusion = DetectionFusion(iou_threshold=0.5)
    segmentor = SAM2Segmentor(use_mock=True)
    intrinsics = CameraIntrinsics(fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480)
    camera = PinholeCamera(intrinsics)
    transformer = FrameTransformer()
    tracker = TemporalFilter(dt=0.1)

    # Metrics storage
    detection_ious = []
    localization_errors_mm = []
    latencies_ms = []
    fusion_statuses = []
    detection_count = 0

    # Create test scene
    image, depth, gt = create_test_scene()
    prompt = "red block"

    print(f"Running evaluation on {num_frames} frames...")
    print(f"Ground truth box: {gt['box']}")
    print(f"Ground truth 3D position: {gt['position_3d']}")
    print()

    for frame_idx in range(num_frames):
        start_time = time.time()

        # Stage 1: Detection
        gdino_result = gdino.detect(image, prompt)
        owlvit_result = owlvit.detect(image, prompt)
        fused = fusion.fuse(gdino_result, owlvit_result)

        fusion_statuses.append(fused.fusion_status)

        if not fused.detected:
            latencies_ms.append((time.time() - start_time) * 1000)
            continue

        detection_count += 1

        # Detection IoU vs ground truth
        iou = DetectionFusion.calculate_iou(fused.box, gt["box"])
        detection_ious.append(iou)

        # Stage 2: Segmentation
        seg_result = segmentor.segment(image, fused.box)
        if not seg_result.valid:
            latencies_ms.append((time.time() - start_time) * 1000)
            continue

        # Stage 3: Deprojection
        cx, cy = seg_result.centroid
        depth_value = depth[int(round(cy)), int(round(cx))]
        point_camera = camera.deproject(cx, cy, float(depth_value))

        if point_camera is not None:
            point_robot = transformer.camera_to_robot_frame(point_camera)

            # Stage 4: Kalman filter
            tracker.predict()
            filtered = tracker.update(point_robot, confidence=fused.confidence)

            # 3D Localisation error
            # Compare filtered position against ground truth transformed
            gt_robot = transformer.camera_to_robot_frame(gt["position_3d"])
            error_m = np.linalg.norm(filtered - gt_robot)
            error_mm = error_m * 1000  # Convert to millimeters
            localization_errors_mm.append(error_mm)

        elapsed_ms = (time.time() - start_time) * 1000
        latencies_ms.append(elapsed_ms)

    # Calculate final metrics
    print("=" * 60)
    print("PERCEPTION PIPELINE EVALUATION RESULTS")
    print("=" * 60)
    print()

    # Detection metrics
    recall = detection_count / num_frames * 100
    print(f"Detection Recall:        {recall:.1f}% ({detection_count}/{num_frames} frames)")

    if detection_ious:
        mean_iou = np.mean(detection_ious)
        print(f"Detection IoU:           {mean_iou:.4f} (mean)")
    else:
        mean_iou = 0
        print("Detection IoU:           N/A (no detections)")

    # Fusion agreement
    agreed_count = fusion_statuses.count("AGREED")
    agreement_rate = agreed_count / num_frames * 100
    print(f"Fusion Agreement Rate:   {agreement_rate:.1f}% ({agreed_count}/{num_frames} frames)")

    # 3D localisation
    if localization_errors_mm:
        median_error = np.median(localization_errors_mm)
        p95_error = np.percentile(localization_errors_mm, 95)
        print(f"3D Localisation Error:   {median_error:.1f} mm (median)")
        print(f"3D Error (95th pct):     {p95_error:.1f} mm")
    else:
        median_error = 0
        p95_error = 0
        print("3D Localisation Error:   N/A")

    # Latency
    if latencies_ms:
        mean_latency = np.mean(latencies_ms)
        print(f"Inference Latency:       {mean_latency:.1f} ms (mean)")
    else:
        mean_latency = 0

    print()
    print("-" * 60)
    print("README TABLE (copy this):")
    print("-" * 60)
    print()
    print("| Metric                    | Value    | Notes                           |")
    print("|---------------------------|----------|---------------------------------|")
    print(
        f"| Detection Recall          | {recall:.1f}%   | {num_frames} frames evaluated           |"
    )
    print(f"| Detection IoU             | {mean_iou:.2f}     | vs ground truth box             |")
    print(
        f"| Fusion Agreement Rate     | {agreement_rate:.1f}%  | GDINO + OWL-ViT IoU > 0.5      |"
    )
    if localization_errors_mm:
        print(
            f"| 3D Localisation Error     | {median_error:.1f} mm  "
            f"| Median; 95th pct: {p95_error:.1f} mm      |"
        )
    print(
        f"| Inference Latency         | {mean_latency:.1f} ms  | Per frame (mock models)         |"
    )
    print()

    return {
        "recall": recall,
        "mean_iou": mean_iou,
        "agreement_rate": agreement_rate,
        "median_error_mm": median_error,
        "p95_error_mm": p95_error,
        "mean_latency_ms": mean_latency,
    }


if __name__ == "__main__":
    results = run_evaluation(num_frames=200)
