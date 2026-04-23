"""GraspNet-1Billion data loader for OmniGrasp evaluation."""

from pathlib import Path
import numpy as np
from PIL import Image
import scipy.io as sio


class GraspNetLoader:
    """Loads RGB, depth, segmentation labels, object poses, and camera intrinsics
    from GraspNet-1Billion scene directories (Kinect or RealSense capture)."""

    def __init__(self, scene_dir: str, camera: str = "kinect"):
        self.root = Path(scene_dir) / camera
        if not self.root.exists():
            raise FileNotFoundError(f"GraspNet scene not found: {self.root}")
        self.intrinsics = np.load(self.root / "camK.npy")
        self.frame_ids = sorted(p.stem for p in (self.root / "rgb").glob("*.png"))

    def __len__(self) -> int:
        return len(self.frame_ids)

    def load_frame(self, frame_id: str) -> dict:
        rgb = np.array(Image.open(self.root / "rgb" / f"{frame_id}.png"))
        depth_raw = np.array(Image.open(self.root / "depth" / f"{frame_id}.png"))
        depth = depth_raw.astype(np.float32) / 1000.0
        label = np.array(Image.open(self.root / "label" / f"{frame_id}.png"))
        meta = sio.loadmat(self.root / "meta" / f"{frame_id}.mat")
        return {
            "rgb": rgb,
            "depth": depth,
            "label": label,
            "meta": meta,
            "intrinsics": self.intrinsics,
            "frame_id": frame_id,
        }
