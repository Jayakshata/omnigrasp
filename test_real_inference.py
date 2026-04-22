
import time
import torch
import numpy as np
from PIL import Image
from transformers import pipeline

print("=" * 60)
print("OMNIGRASP - REAL MODEL INFERENCE TEST")
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

gt_box = [250, 180, 390, 300]
prompt = "red block"

img = np.full((480, 640, 3), 40, dtype=np.uint8)
img[180:300, 250:390] = [200, 50, 50]
pil_img = Image.fromarray(img)

print("\nLoading Grounding DINO...")
gdino = pipeline("zero-shot-object-detection", model="IDEA-Research/grounding-dino-base", device=0)

print("Loading OWL-ViT...")
owlvit = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32", device=0)

print("Warmup...")
_ = gdino(pil_img, candidate_labels=[prompt])
_ = owlvit(pil_img, candidate_labels=[prompt])

num_frames = 50
gdino_times = []
owlvit_times = []
gdino_dets = []

print(f"\nRunning {num_frames} frames...")
for i in range(num_frames):
    t0 = time.time()
    gr = gdino(pil_img, candidate_labels=[prompt])
    gdino_times.append((time.time()-t0)*1000)
    filtered = [r for r in gr if r["score"]>0.3]
    if filtered:
        gdino_dets.append(max(filtered, key=lambda x:x["score"]))
    t0 = time.time()
    owlvit(pil_img, candidate_labels=[prompt])
    owlvit_times.append((time.time()-t0)*1000)
    if (i+1)%10==0: print(f"  Frame {i+1}/{num_frames}")

def iou(a,b):
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    if x1>=x2 or y1>=y2: return 0.0
    inter=(x2-x1)*(y2-y1)
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"GDINO Detection Rate: {len(gdino_dets)}/{num_frames}")
if gdino_dets:
    ious=[iou([d["box"]["xmin"],d["box"]["ymin"],d["box"]["xmax"],d["box"]["ymax"]],gt_box) for d in gdino_dets]
    print(f"GDINO Mean IoU: {np.mean(ious):.4f}")
    print(f"GDINO Mean Conf: {np.mean([d['score'] for d in gdino_dets]):.4f}")
    print(f"GDINO Best Box: {gdino_dets[0]['box']}")
print(f"GDINO Latency: {np.mean(gdino_times):.1f}ms (p95: {np.percentile(gdino_times,95):.1f}ms)")
print(f"OWL-ViT Latency: {np.mean(owlvit_times):.1f}ms (p95: {np.percentile(owlvit_times,95):.1f}ms)")
print(f"GPU Mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print("DONE!")
