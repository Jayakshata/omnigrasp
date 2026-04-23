"""Day 5: Run full OmniGrasp pipeline on GraspNet-format real-world data, locally with mocks.
Validates the integration end-to-end before spending GPU money on real models."""

import sys

sys.path.insert(0, "src/omnigrasp_perception")

import numpy as np
from omnigrasp_perception.data.graspnet_loader import GraspNetLoader
from omnigrasp_perception.detectors.grounding_dino import GroundingDINODetector
from omnigrasp_perception.detectors.owl_vit import OWLViTDetector
from omnigrasp_perception.detectors.detection_fusion import DetectionFusion
from omnigrasp_perception.geometry.camera_model import PinholeCamera, CameraIntrinsics

print("=" * 60)
print("OMNIGRASP — GRASPNET LOCAL INTEGRATION TEST (MOCK)")
print("=" * 60)

loader = GraspNetLoader("data/graspnet/scene_0000")
frame = loader.load_frame(loader.frame_ids[0])

K = frame["intrinsics"]
H, W = frame["rgb"].shape[:2]
cam = PinholeCamera(
    CameraIntrinsics(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=W, height=H)
)

gdino = GroundingDINODetector(use_mock=True)
owlvit = OWLViTDetector(use_mock=True)
fusion = DetectionFusion()

target = "object on table"
g_res = gdino.detect(frame["rgb"], target)
o_res = owlvit.detect(frame["rgb"], target)
fused = fusion.fuse(g_res, o_res)

print(f"GDINO mock: detected={g_res.detected}, conf={g_res.confidence:.2f}")
print(f"OWL-ViT mock: detected={o_res.detected}, conf={o_res.confidence:.2f}")
print(f"Fusion: status={fused.fusion_status}, box={fused.box}")

if fused.box is not None:
    cx_px = int((fused.box[0] + fused.box[2]) / 2)
    cy_px = int((fused.box[1] + fused.box[3]) / 2)
    z = float(frame["depth"][cy_px, cx_px])
    point_3d = cam.deproject(cx_px, cy_px, z)
    print(f"3D point (camera frame): {point_3d[:3]} m at depth {z:.2f}m")

print("\n=== INTEGRATION OK ===")
