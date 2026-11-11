"""HƯỚNG 3 — Fusion + bảng ablation, dùng crops do huong1_osnet.py đã cắt từ video cố định.
Chạy SAU Hướng 1: colab exec -s avt -f huong3_fusion.py
In 3 cột: chỉ ngoại hình (OSNet) / chỉ khuôn mặt (buffalo_l) / fusion 0.6-0.4 cho từng cặp.
Không thấy mặt trong crop -> ghi rõ fallback chỉ dùng ngoại hình (không crash).
"""
import os
import numpy as np
import cv2

WORK = os.environ.get("DEMO_DIR", "/content")
A1 = os.path.join(WORK, "crop_cungnguoi_1.jpg")
A2 = os.path.join(WORK, "crop_cungnguoi_2.jpg")
B = os.path.join(WORK, "crop_khacnguoi.jpg")
for p in (A1, A2, B):
    if not os.path.exists(p):
        raise SystemExit(f"Thiếu {p} — chạy huong1_osnet.py trước.")

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    from torchreid.reid.utils import FeatureExtractor
import subprocess
WEIGHTS = os.path.join(WORK, "weights_osnet_msmt17.pth")
if not os.path.exists(WEIGHTS):
    url = ("https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
           "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
    subprocess.run(["wget", "-q", "--user-agent=Mozilla/5.0 (compatible; AVT-Demo/1.0)", url, "-O", WEIGHTS], check=True)
device = os.environ.get("DEMO_DEVICE", "cuda")
extractor = FeatureExtractor(model_name="osnet_x1_0", model_path=WEIGHTS, device=device)
feats = extractor([A1, A2, B])
feats = (feats / feats.norm(dim=1, keepdim=True)).cpu().numpy()
body = {(0, 1): float(feats[0] @ feats[1]), (0, 2): float(feats[0] @ feats[2])}

from insightface.app import FaceAnalysis
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
app = FaceAnalysis(name="buffalo_l", providers=providers)
app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

def face_emb(p):
    img = cv2.imread(p)
    img = cv2.resize(img, None, fx=2, fy=2)  # phóng to để dễ bắt mặt trong crop
    faces = app.get(img)
    return faces[0].normed_embedding if faces else None

embs = [face_emb(p) for p in (A1, A2, B)]
def face_score(i, j):
    if embs[i] is None or embs[j] is None:
        return None
    return float(embs[i] @ embs[j])

pairs = [("cùng người (2 thời điểm)", 0, 1), ("khác người", 0, 2)]
print(f"{'Cặp':<26}{'Ngoại hình':>12}{'Khuôn mặt':>12}{'Fusion':>10}")
for label, i, j in pairs:
    b = body[(i, j)]
    f = face_score(i, j)
    fused = 0.6 * b + 0.4 * f if f is not None else None
    print(f"{label:<26}{b:>12.3f}{(f'{f:.3f}' if f is not None else 'không thấy mặt'):>12}{(f'{fused:.3f}' if fused is not None else 'chỉ dùng ngoại hình'):>10}")
