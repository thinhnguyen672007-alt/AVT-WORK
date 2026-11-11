"""HƯỚNG 1 — Re-ID bằng ngoại hình, CHUNG 1 video cố định (Wikimedia Commons).
Luồng: video phố đi bộ → YOLO+BoT-SORT bám người → lấy các crop TO NHẤT của cùng 1 ID
+ crop của 1 ID khác → OSNet (weights train trên MSMT17) trích mã ngoại hình → ma trận độ giống.
Đã kiểm chứng (chạy thử của Milo): cùng người ~0.86, khác người ~0.40.
Chạy: colab exec -s avt -f huong1_osnet.py  (script tự tải video + weights ở lần đầu)
"""
import os, subprocess

WORK = os.environ.get("DEMO_DIR", "/content")
os.makedirs(WORK, exist_ok=True)
VIDEO = os.path.join(WORK, "video.mp4")
if not os.path.exists(VIDEO):
    ogv = os.path.join(WORK, "video.ogv")
    subprocess.run(["wget", "-q", "--user-agent=Mozilla/5.0 (compatible; AVT-Demo/1.0)", "https://upload.wikimedia.org/wikipedia/commons/0/0b/Calle_preciados.ogv", "-O", ogv], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", ogv, "-q:v", "3", VIDEO], check=True)
    print("Đã tải + đổi video cố định sang mp4.")

WEIGHTS = os.path.join(WORK, "weights_osnet_msmt17.pth")
if not os.path.exists(WEIGHTS):
    url = ("https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
           "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
    subprocess.run(["wget", "-q", "--user-agent=Mozilla/5.0 (compatible; AVT-Demo/1.0)", url, "-O", WEIGHTS], check=True)
    print("Đã tải weights OSNet train trên MSMT17 (KHÔNG dùng weights ImageNet — điểm sẽ sai).")

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
samples = {}  # track_id -> list[(frame_idx, crop, area)]
results = model.track(source=VIDEO, tracker="botsort.yaml", classes=[0], stream=True, verbose=False)
for fi, r in enumerate(results):
    if r.boxes is None or r.boxes.id is None:
        continue
    img = r.orig_img
    for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = [int(v) for v in box]
        if (x2 - x1) < 40 or (y2 - y1) < 90:  # bỏ người quá nhỏ/xa — crop phải rõ
            continue
        lst = samples.setdefault(int(tid), [])
        if not lst or fi - lst[-1][0] >= 30:  # lấy mẫu thưa theo thời gian
            lst.append((fi, img[y1:y2, x1:x2].copy(), (x2 - x1) * (y2 - y1)))
ranked = sorted(samples.items(), key=lambda kv: len(kv[1]), reverse=True)
main_id, ml = ranked[0]
other_id, ol = ranked[1]
ml_by_area = sorted(ml, key=lambda t: -t[2])
ol_by_area = sorted(ol, key=lambda t: -t[2])
first = ml_by_area[0]
second = next((t for t in ml_by_area[1:] if abs(t[0] - first[0]) >= 45), ml_by_area[1])
jobs = [("crop_cungnguoi_1.jpg", first[1]), ("crop_cungnguoi_2.jpg", second[1]), ("crop_khacnguoi.jpg", ol_by_area[0][1])]
paths = []
for name, crop in jobs:
    p = os.path.join(WORK, name)
    cv2.imwrite(p, crop)
    paths.append(p)
print(f"Cùng người = ID {main_id} (2 crop to nhất, cách nhau {abs(second[0]-first[0])} frame); khác người = ID {other_id}.")

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None  # chặn tensorboard kéo TF vào làm vỡ torchreid (lỗi numpy/jax)
try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    from torchreid.reid.utils import FeatureExtractor
device = os.environ.get("DEMO_DEVICE", "cuda")
extractor = FeatureExtractor(model_name="osnet_x1_0", model_path=WEIGHTS, device=device)

extra = []
for t in ml_by_area[1:4]:
    p = os.path.join(WORK, f"_them_{t[0]}.jpg")
    cv2.imwrite(p, t[1])
    extra.append(p)
feats = extractor(paths + extra)
feats = (feats / feats.norm(dim=1, keepdim=True)).cpu().numpy()
sims = feats[:3] @ feats[:3].T
names = [os.path.basename(p) for p in paths]
print("\nMa trận độ giống OSNet (1.0 = giống hệt):")
print("               " + "".join(f"{n[:12]:>13}" for n in names))
for i, n in enumerate(names):
    print(f"{n[:12]:>12} " + "".join(f"{sims[i, j]:>13.3f}" for j in range(3)))
proto = feats[[0] + list(range(3, 3 + len(extra)))].mean(0)
proto /= np.linalg.norm(proto)
print(f"\nĐiểm 'hồ sơ trung bình' của ID {main_id} (gộp nhiều crop) so với người khác: {float(proto @ feats[2]):.3f}")
print("Kỳ vọng: cùng người CAO (~0.8+), khác người THẤP (~0.4).")
