"""Generate a synthetic GraspNet-1Billion-format scene for pipeline validation.
Real GraspNet requires Google Drive download (120GB). For development/CI we
generate textured tabletop scenes in the exact same on-disk format so the
loader, pipeline, and evaluation code remain identical when swapping in real data."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import scipy.io as sio

OUT = Path("scene_0000/kinect")
for sub in ["rgb", "depth", "label", "meta"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# GraspNet Kinect intrinsics (real values from the dataset)
K = np.array([[631.5, 0.0, 638.4], [0.0, 631.2, 366.6], [0.0, 0.0, 1.0]], dtype=np.float64)
np.save(OUT / "camK.npy", K)

H, W = 720, 1280
np.random.seed(42)
NUM_FRAMES = 5
OBJECTS = [
    {"name": "red_box", "color": (200, 40, 40), "bbox": (500, 320, 640, 440), "depth_m": 0.85},
    {"name": "blue_can", "color": (40, 60, 200), "bbox": (700, 350, 800, 500), "depth_m": 0.78},
    {"name": "green_bowl", "color": (60, 180, 60), "bbox": (350, 380, 480, 470), "depth_m": 0.92},
]

for f in range(NUM_FRAMES):
    # Wood-grain table background
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    grain = (np.sin(np.arange(W) * 0.05) * 20 + 120).astype(np.uint8)
    bg[:] = np.stack([grain * 1.1, grain * 0.85, grain * 0.6], axis=-1).clip(0, 255)
    bg += np.random.randint(-15, 15, bg.shape, dtype=np.int8).astype(np.uint8)

    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)
    label = np.zeros((H, W), dtype=np.uint8)
    depth = np.full((H, W), 1.5, dtype=np.float32)  # background table at 1.5m
    poses = []

    for i, obj in enumerate(OBJECTS, start=1):
        # Jitter object position per frame to simulate camera motion
        dx = np.random.randint(-30, 30)
        dy = np.random.randint(-15, 15)
        x1, y1, x2, y2 = obj["bbox"]
        x1, x2 = x1 + dx, x2 + dx
        y1, y2 = y1 + dy, y2 + dy
        draw.rectangle([x1, y1, x2, y2], fill=obj["color"], outline=(20, 20, 20), width=3)
        label[y1:y2, x1:x2] = i
        depth[y1:y2, x1:x2] = obj["depth_m"]
        # Object pose (4x4): identity rotation, translation = bbox centre deprojected
        cx_px, cy_px = (x1 + x2) / 2, (y1 + y2) / 2
        z = obj["depth_m"]
        x = (cx_px - K[0, 2]) * z / K[0, 0]
        y = (cy_px - K[1, 2]) * z / K[1, 1]
        T = np.eye(4)
        T[:3, 3] = [x, y, z]
        poses.append(T)

    img.save(OUT / "rgb" / f"{f:04d}.png")
    Image.fromarray((depth * 1000).astype(np.uint16)).save(OUT / "depth" / f"{f:04d}.png")
    Image.fromarray(label).save(OUT / "label" / f"{f:04d}.png")
    sio.savemat(
        OUT / "meta" / f"{f:04d}.mat",
        {
            "cls_indexes": np.array([1, 2, 3]),
            "poses": np.stack(poses, axis=-1),
            "object_names": np.array([o["name"] for o in OBJECTS], dtype=object),
            "intrinsic_matrix": K,
            "factor_depth": 1000.0,
        },
    )

print(f"Generated {NUM_FRAMES} synthetic frames in {OUT}")
