"""HƯỚNG 3 — BẢN VIDEO: Fusion ngoại hình + khuôn mặt trình bày dạng video.
Cùng kịch bản video cố định: người mẫu G1 được nhận diện bằng ĐIỂM FUSION = 0.6*ngoại hình (OSNet)
+ 0.4*khuôn mặt (buffalo_l). Thanh dưới video hiện 3 con số theo thời gian thực của người giống nhất:
Ngoại hình / Khuôn mặt / Fusion. Không thấy mặt -> ghi "khong thay mat" và dùng riêng ngoại hình (fallback).
Chạy: colab exec -s avt --timeout 900 -f huong3_video.py ; kết quả: /content/huong3-fusion-video.mp4
"""
import os, subprocess

WORK = os.environ.get("DEMO_DIR", "/content")
os.makedirs(WORK, exist_ok=True)
VIDEO = os.path.join(WORK, "video.mp4")
WEIGHTS = os.path.join(WORK, "weights_osnet_msmt17.pth")
assert os.path.exists(VIDEO) and os.path.exists(WEIGHTS), "Thiếu video/weights trên /content — upload trước"

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

from insightface.app import FaceAnalysis
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
face_app = FaceAnalysis(name="buffalo_l", providers=providers)
face_app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

def body_embed(crops):
    tmp = []
    for i, c in enumerate(crops):
        p = os.path.join(WORK, f"_f{i}.jpg")
        cv2.imwrite(p, c)
        tmp.append(p)
    f = extractor(tmp)
    return (f / f.norm(dim=1, keepdim=True)).cpu().numpy()

def face_embed(crop):
    big = cv2.resize(crop, None, fx=2, fy=2)
    faces = face_app.get(big)
    return faces[0].normed_embedding if faces else None

model = YOLO("yolov8n.pt")
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
ranked = sorted(samples.items(), key=lambda kv: len(kv[1]), reverse=True)
main_id, top, face_proto = None, None, None
for cand_id, cand_list in ranked[:5]:
    cand_top = [c for _, c, _ in sorted(cand_list, key=lambda t: -t[2])[:4]]
    for c in cand_top:
        e = face_embed(c)
        if e is not None:
            main_id, top, face_proto = cand_id, cand_top, e
            break
    if main_id is not None:
        break
if main_id is None:
    main_id, ml0 = ranked[0]
    top = [c for _, c, _ in sorted(ml0, key=lambda t: -t[2])[:4]]
proto_body = body_embed(top).mean(0); proto_body /= np.linalg.norm(proto_body)
print(f"Người mẫu = ID {main_id} (G1). Có mặt mẫu để fusion: {face_proto is not None}. Dựng video...")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 24
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
raw_out = os.path.join(WORK, "_huong3_raw.avi")
writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + 46))
cache, last_check = {}, {}
panel = {"body": None, "face": None, "fused": None}
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
            embs = body_embed(crops)
            for k, e, crop in zip(due, embs, crops):
                tid = int(tids[k])
                b = float(e @ proto_body)
                f = None
                if face_proto is not None:
                    fe = face_embed(crop)
                    if fe is not None:
                        f = float(fe @ face_proto)
                fused = 0.6 * b + 0.4 * f if f is not None else b
                old = cache.get(tid)
                if old is None or fused > old[2]:
                    cache[tid] = (b, f, fused)
                last_check[tid] = fi
        best_tid, best = None, None
        for box, tid in zip(boxes, tids):
            tid = int(tid)
            v = cache.get(tid)
            if v and (best is None or v[2] > best[2]):
                best_tid, best = tid, v
        for box, tid in zip(boxes, tids):
            x1, y1, x2, y2 = [int(v) for v in box]
            tid = int(tid)
            v = cache.get(tid)
            if v is not None and v[2] >= 0.60:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
                cv2.putText(frame, f"G1 - FUSION {v[2]:.2f}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                cv2.putText(frame, f"ID {tid}", (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if best is not None:
            panel = {"body": best[0], "face": best[1], "fused": best[2]}
    cv2.putText(frame, "HUONG 3 - Fusion ngoai hinh + khuon mat", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    strip = np.zeros((46, W, 3), dtype=np.uint8)
    if panel["fused"] is not None:
        fstr = f"{panel['face']:.2f}" if panel["face"] is not None else "khong thay mat"
        txt = f"Nguoi giong nhat: Ngoai hinh {panel['body']:.2f} | Khuon mat {fstr} | FUSION {panel['fused']:.2f}"
    else:
        txt = "Dang quet..."
    cv2.putText(strip, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    writer.write(np.vstack([frame, strip]))
writer.release()
final = os.path.join(WORK, "huong3-fusion-video.mp4")
r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_out, "-c:v", "libx264", "-crf", "23", final])
print("XONG:", final if r.returncode == 0 else raw_out)
