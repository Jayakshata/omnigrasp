"""Day 5: Real GDINO + OWL-ViT inference on GraspNet-format scenes (RTX 4090).
Measures per-frame detection IoU vs ground-truth labels, fusion agreement,
3D localization error, and end-to-end latency. Saves results to JSON
and one annotated demo image."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "src/omnigrasp_perception")
from omnigrasp_perception.data.graspnet_loader import GraspNetLoader
from omnigrasp_perception.detectors.grounding_dino import GroundingDINODetector
from omnigrasp_perception.detectors.owl_vit import OWLViTDetector
from omnigrasp_perception.detectors.detection_fusion import DetectionFusion
from omnigrasp_perception.geometry.camera_model import PinholeCamera, CameraIntrinsics

print("=" * 60)
print("OMNIGRASP — GRASPNET REAL-INFERENCE EVAL (GPU)")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


def label_to_bbox(label: np.ndarray, obj_id: int):
    """Convert binary mask to [x1, y1, x2, y2] bbox in pixels."""
    ys, xs = np.where(label == obj_id)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0])* (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --- Load data
loader = GraspNetLoader("data/graspnet/scene_0000")
print(f"\nLoaded {len(loader)} frames from synthetic GraspNet scene")

K = loader.intrinsics
cam = PinholeCamera(CameraIntrinsics(
    fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], width=1280, height=720
))

# --- Init real detectors (GPU)
print("\nLoading models (this takes 30-60s on first run)...")
t0 = time.time()
gdino = GroundingDINODetector(use_mock=False)
owlvit = OWLViTDetector(use_mock=False)
fusion = DetectionFusion()
print(f"Models loaded in {time.time()-t0:.1f}s")
print(f"GPU memory after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# --- Eval loop
TARGETS = {1: "red box", 2: "blue can", 3: "green bowl"}
results = []
demo_saved = False

for frame_id in loader.frame_ids:
    frame = loader.load_frame(frame_id)
    rgb = frame["rgb"]
    depth = frame["depth"]
    label = frame["label"]

    for obj_id, prompt in TARGETS.items():
        gt_box = label_to_bbox(label, obj_id)
        if gt_box is None:
            continue

        t0 = time.time()
        g_res = gdino.detect(rgb, prompt)
        o_res = owlvit.detect(rgb, prompt)
        fused = fusion.fuse(g_res, o_res)
        latency_ms = (time.time() - t0) * 1000

        gdino_iou = iou(g_res.box, gt_box) if g_res.detected else 0.0
        owlvit_iou = iou(o_res.box, gt_box) if o_res.detected else 0.0
        fused_iou = iou(fused.box, gt_box) if fused.detected else 0.0

        # 3D error if fusion succeeded
        err_mm = None
        if fused.detected and fused.box is not None:
            cx_px = int((fused.box[0] + fused.box[2]) / 2)
            cy_px = int((fused.box[1] + fused.box[3]) / 2)
            cx_px = np.clip(cx_px, 0, 1279); cy_px = np.clip(cy_px, 0, 719)
            z = float(depth[cy_px, cx_px])
            pred_3d = cam.deproject(cx_px, cy_px, z)[:3]
            # Ground-truth 3D from meta poses (object_id -> pose index)
            poses = frame["meta"]["poses"]  # shape (4, 4, N)
            cls_idx = frame["meta"]["cls_indexes"].flatten().tolist()
            if obj_id in cls_idx:
                pose_idx = cls_idx.index(obj_id)
                gt_3d = poses[:3, 3, pose_idx]
                err_mm = float(np.linalg.norm(pred_3d - gt_3d) * 1000)

        results.append({
            "frame_id": frame_id, "object_id": obj_id, "prompt": prompt,
            "gdino_detected": bool(g_res.detected), "gdino_iou": float(gdino_iou),
            "gdino_confidence": float(g_res.confidence),
            "owlvit_detected": bool(o_res.detected), "owlvit_iou": float(owlvit_iou),
            "owlvit_confidence": float(o_res.confidence),
            "fused_detected": bool(fused.detected), "fused_iou": float(fused_iou),
            "fusion_status": str(fused.fusion_status),
            "error_mm": err_mm,
            "latency_ms": float(latency_ms),
        })
        print(f"  Frame {frame_id} obj={obj_id} ({prompt}): "
              f"GDINO IoU={gdino_iou:.2f} | OWL IoU={owlvit_iou:.2f} | "
              f"fused={fused.fusion_status} | err={err_mm}mm | {latency_ms:.0f}ms")

    # Save one demo image
    if not demo_saved and any(r["frame_id"] == frame_id and r["fused_detected"] for r in results):
        img = Image.fromarray(rgb).copy()
        draw = ImageDraw.Draw(img)
        for r in [x for x in results if x["frame_id"] == frame_id]:
            obj_id = r["object_id"]
            gt = label_to_bbox(label, obj_id)
            if gt is not None:
                draw.rectangle(gt.tolist(), outline=(0, 255, 0), width=4)
                draw.text((gt[0], gt[1] - 15), f"GT: {r['prompt']}", fill=(0, 255, 0))
        img.save("eval_demo.png")
        demo_saved = True
        print(f"  Saved demo image: eval_demo.png")

# --- Summary
print("\n" + "=" * 60)
print("EVAL SUMMARY")
print("=" * 60)
n = len(results)
gdino_recall = sum(r["gdino_detected"] for r in results) / n
owlvit_recall = sum(r["owlvit_detected"] for r in results) / n
fused_recall = sum(r["fused_detected"] for r in results) / n
mean_gdino_iou = np.mean([r["gdino_iou"] for r in results])
mean_owlvit_iou = np.mean([r["owlvit_iou"] for r in results])
mean_fused_iou = np.mean([r["fused_iou"] for r in results])
mean_latency = np.mean([r["latency_ms"] for r in results])
errs = [r["error_mm"] for r in results if r["error_mm"] is not None]
mean_err = np.mean(errs) if errs else None

print(f"Frames evaluated: {len(loader)}, total detections: {n}")
print(f"GDINO recall: {gdino_recall:.1%}, mean IoU: {mean_gdino_iou:.3f}")
print(f"OWL-ViT recall: {owlvit_recall:.1%}, mean IoU: {mean_owlvit_iou:.3f}")
print(f"Fused recall: {fused_recall:.1%}, mean IoU: {mean_fused_iou:.3f}")
print(f"Mean 3D error: {mean_err:.1f} mm" if mean_err else "No 3D errors computed")
print(f"Mean latency: {mean_latency:.1f} ms / object")
print(f"GPU memory peak: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

with open("eval_results.json", "w") as f:
    json.dump({
        "summary": {
            "n_frames": len(loader), "n_detections": n,
            "gdino_recall": gdino_recall, "gdino_mean_iou": float(mean_gdino_iou),
            "owlvit_recall": owlvit_recall, "owlvit_mean_iou": float(mean_owlvit_iou),
            "fused_recall": fused_recall, "fused_mean_iou": float(mean_fused_iou),
            "mean_error_mm": mean_err,
            "mean_latency_ms": float(mean_latency),
            "gpu_memory_gb": torch.cuda.max_memory_allocated()/1024**3,
        },
        "per_detection": results,
    }, f, indent=2)
print("\nSaved: eval_results.json, eval_demo.png")
print("=== EVAL COMPLETE ===")
