"""HƯỚNG 1 — BẢN VIDEO: Re-ID bằng ngoại hình trình bày dạng video cho dễ hiểu.
Kịch bản: lấy 1 người trong video làm "người mẫu" (đăng ký bằng các crop to nhất của họ),
rồi chạy lại video: ai giống người mẫu (điểm OSNet >= 0.60) thì khung XANH + nhãn G1 kèm điểm,
người khác khung xám + Local ID. Đây là Hướng 1 (so mã ngoại hình) được "vẽ" lên video.
Chạy: colab exec -s avt --timeout 900 -f huong1_video.py ; tải về: /content/huong1-reid-video.mp4
"""
import os, subprocess

WORK = os.environ.get("DEMO_DIR", "/content")
os.makedirs(WORK, exist_ok=True)
VIDEO = os.path.join(WORK, "video.mp4")
WEIGHTS = os.path.join(WORK, "weights_osnet_msmt17.pth")
assert os.path.exists(VIDEO), "Thiếu /content/video.mp4 — upload video trước (xem HUONG-DAN-TU-CHAY.md)"
assert os.path.exists(WEIGHTS), "Thiếu weights — upload weights_osnet_msmt17.pth trước"

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    from torchreid.reid.utils import FeatureExtractor
device = os.environ.get("DEMO_DEVICE", "cuda")
extractor = FeatureExtractor(model_name="osnet_x1_0", model_path=WEIGHTS, device=device)

def embed(crops):
    if not crops:
        return None
    tmp = []
    for i, c in enumerate(crops):
        p = os.path.join(WORK, f"_v{i}.jpg")
        cv2.imwrite(p, c)
        tmp.append(p)
    f = extractor(tmp)
    return (f / f.norm(dim=1, keepdim=True)).cpu().numpy()

model = YOLO("yolov8n.pt")

# --- Lượt 1: tìm người mẫu (ID bám lâu nhất), dựng "hồ sơ ngoại hình" từ các crop to nhất ---
samples = {}
for fi, r in enumerate(model.track(source=VIDEO, tracker="botsort.yaml", classes=[0], stream=True, verbose=False)):
    if r.boxes is None or r.boxes.id is None:
        continue
    img = r.orig_img
    for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = [int(v) for v in box]
        if (x2 - x1) < 40 or (y2 - y1) < 90:
            continue
        lst = samples.setdefault(int(tid), [])
        if not lst or fi - lst[-1][0] >= 30:
            lst.append((fi, img[y1:y2, x1:x2].copy(), (x2 - x1) * (y2 - y1)))
main_id, ml = sorted(samples.items(), key=lambda kv: len(kv[1]), reverse=True)[0]
top_crops = [c for _, c, _ in sorted(ml, key=lambda t: -t[2])[:4]]
proto = embed(top_crops).mean(0)
proto /= np.linalg.norm(proto)
print(f"Người mẫu = ID {main_id} (gán nhãn toàn cục G1). Bắt đầu dựng video...")

# --- Lượt 2: chạy lại video, chấm điểm từng người so với người mẫu, vẽ khung ---
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 24
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
raw_out = os.path.join(WORK, "_huong1_raw.avi")
writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
score_cache, last_check = {}, {}
for fi, r in enumerate(model.track(source=VIDEO, tracker="botsort.yaml", classes=[0], stream=True, verbose=False)):
    frame = r.orig_img.copy()
    if r.boxes is not None and r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        tids = r.boxes.id.cpu().numpy().astype(int)
        due = [k for k, tid in enumerate(tids) if tid not in last_check or fi - last_check[tid] >= 20]
        if due:
            crops = []
            for k in due:
                x1, y1, x2, y2 = [int(v) for v in boxes[k]]
                crops.append(frame[y1:y2, x1:x2].copy())
            embs = embed(crops)
            for k, e in zip(due, embs):
                s = float(e @ proto)
                tid = int(tids[k])
                score_cache[tid] = max(s, score_cache.get(tid, 0) * 0.9) if tid in score_cache else s
                last_check[tid] = fi
        for box, tid in zip(boxes, tids):
            x1, y1, x2, y2 = [int(v) for v in box]
            tid = int(tid)
            s = score_cache.get(tid)
            if s is not None and s >= 0.60:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
                cv2.putText(frame, f"G1 - NGUOI MAU {s:.2f}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                cv2.putText(frame, f"ID {tid}" + (f" {s:.2f}" if s is not None else ""), (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "HUONG 1 - Re-ID bang ngoai hinh (OSNet)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    writer.write(frame)
writer.release()
final = os.path.join(WORK, "huong1-reid-video.mp4")
r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_out, "-c:v", "libx264", "-crf", "23", final])
print("XONG:", final if r.returncode == 0 else raw_out)
