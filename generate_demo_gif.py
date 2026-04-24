"""
generate_demo_gif.py — Create a demo GIF showing the full perception pipeline.

Runs locally on synthetic GraspNet frames using mock detectors.
Output: demo.gif (embedded in README)

Pipeline visualised per frame:
  1. RGB input
  2. Bounding box (GDINO, green)
  3. Segmentation mask (SAM2, blue overlay)
  4. 3D point projected back to image (red dot)
  5. Grasp axes (XYZ arrows)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "src/omnigrasp_perception")
from omnigrasp_perception.data.graspnet_loader import GraspNetLoader
from omnigrasp_perception.detectors.grounding_dino import GroundingDINODetector
from omnigrasp_perception.detectors.detection_fusion import DetectionFusion
from omnigrasp_perception.detectors.owl_vit import OWLViTDetector
from omnigrasp_perception.geometry.camera_model import CameraIntrinsics, PinholeCamera
from omnigrasp_perception.segmentation.sam2_segmentor import SAM2Segmentor

PROMPT = "red box"
OUT_PATH = "demo.gif"
FRAME_DURATION_MS = 600  # ms per frame in GIF


def draw_pipeline(rgb: np.ndarray, K, frame_label: str) -> np.ndarray:
    """Run mock pipeline on one frame and return annotated BGR image."""
    gdino = GroundingDINODetector(use_mock=True)
    owlvit = OWLViTDetector(use_mock=True)
    fusion = DetectionFusion()
    segmentor = SAM2Segmentor(use_mock=True)
    intrinsics = CameraIntrinsics(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=rgb.shape[1], height=rgb.shape[0]
    )
    camera = PinholeCamera(intrinsics)

    img = rgb.copy()

    # --- Stage 1: Detection
    g_res = gdino.detect(rgb, PROMPT)
    o_res = owlvit.detect(rgb, PROMPT)
    fused = fusion.fuse(g_res, o_res)

    if not fused.detected:
        _put_label(img, frame_label, status="NO DETECTION")
        return img

    x1, y1, x2, y2 = [int(v) for v in fused.box]

    # --- Stage 2: Segmentation mask overlay (blue, semi-transparent)
    seg = segmentor.segment(rgb, fused.box)
    if seg.valid and seg.mask is not None:
        overlay = img.copy()
        overlay[seg.mask > 0] = [100, 149, 237]  # cornflower blue
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # --- Stage 3: Bounding box (green)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(
        img,
        f"{PROMPT} {fused.confidence:.2f}",
        (x1, max(y1 - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 220, 0),
        1,
        cv2.LINE_AA,
    )

    # --- Stage 4: 3D point — use mask centroid, project back to image
    if seg.valid:
        cx_px, cy_px = int(seg.centroid[0]), int(seg.centroid[1])
    else:
        cx_px, cy_px = (x1 + x2) // 2, (y1 + y2) // 2

    # Mock depth: centre of box at 0.6m
    z = 0.6
    pt3d = camera.deproject(cx_px, cy_px, z)
    # Project back (it's the same pixel, but this proves the round-trip)
    u = int(pt3d[0] * K[0, 0] / pt3d[2] + K[0, 2])
    v = int(pt3d[1] * K[1, 1] / pt3d[2] + K[1, 2])
    cv2.circle(img, (u, v), 6, (0, 0, 220), -1)  # red dot
    cv2.putText(
        img,
        f"3D ({pt3d[0]:.2f},{pt3d[1]:.2f},{pt3d[2]:.2f})m",
        (u + 8, v + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 220),
        1,
        cv2.LINE_AA,
    )

    # --- Stage 5: Grasp axes (XYZ arrows from centroid)
    AXIS_LEN = 30
    axes = {
        "X": ([1, 0, 0], (0, 0, 200)),  # red
        "Y": ([0, 1, 0], (0, 180, 0)),  # green
        "Z": ([0, 0, 1], (200, 0, 0)),  # blue
    }
    for label_ax, (direction, colour) in axes.items():
        tip_x = cx_px + int(direction[0] * AXIS_LEN)
        tip_y = cy_px + int(direction[1] * AXIS_LEN)
        cv2.arrowedLine(img, (cx_px, cy_px), (tip_x, tip_y), colour, 2, tipLength=0.3)
        cv2.putText(
            img,
            label_ax,
            (tip_x + 2, tip_y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            colour,
            1,
            cv2.LINE_AA,
        )

    _put_label(img, frame_label, status="DETECTED")
    return img


def _put_label(img, frame_label, status):
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 22), (img.shape[1], h), (20, 20, 20), -1)
    cv2.putText(
        img,
        f"{frame_label}  |  {status}",
        (6, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def main():
    loader = GraspNetLoader("data/graspnet/scene_0000")
    K = loader.intrinsics
    print(f"Loaded {len(loader)} frames. Generating GIF...")

    frames_pil = []
    for frame_id in loader.frame_ids:
        frame = loader.load_frame(frame_id)
        rgb = frame["rgb"]  # H x W x 3, uint8

        annotated_bgr = draw_pipeline(rgb, K, frame_label=f"Frame {frame_id}")

        # Resize to 640 wide for GIF size control
        h, w = annotated_bgr.shape[:2]
        new_w = 640
        new_h = int(h * new_w / w)
        resized = cv2.resize(annotated_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert BGR -> RGB -> PIL
        rgb_out = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(rgb_out).convert("P", palette=Image.ADAPTIVE, colors=256))

    # Save GIF
    frames_pil[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames_pil[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=False,
    )
    size_kb = Path(OUT_PATH).stat().st_size / 1024
    print(f"Saved {OUT_PATH} ({size_kb:.0f} KB, {len(frames_pil)} frames)")
    print("Done. Embed in README with: ![demo](demo.gif)")


if __name__ == "__main__":
    main()
