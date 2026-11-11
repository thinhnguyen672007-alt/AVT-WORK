import sys, os, json, glob
sys.modules["tensorflow"] = None
sys.path.insert(0, "/home/hatch/workspace/prid-work")
import numpy as np, cv2
from vidlib import draw_text, panel, paste_crop, bar

S = json.load(open("/home/hatch/workspace/prid-work/scan.json"))
CAM_A, CAM_B, HERO = S["CAM_A"], S["CAM_B"], S["hero"]
HARD, MID, EASY = S["hard"], S["mid"], S["easy"]
OUT = "/home/hatch/workspace/prid-work/out"; os.makedirs(OUT, exist_ok=True)
W, H, FPS = 1280, 720, 10
GREEN, GREY, YEL = (60,200,90), (150,150,150), (40,190,240)

def frames(cam, pid):
    fs = sorted(glob.glob(f"{cam}/{pid}/*.png") + glob.glob(f"{cam}/{pid}/*.jpg"))
    return [cv2.imread(f) for f in fs]

from torchreid.reid.utils import FeatureExtractor
ext = FeatureExtractor(model_name="osnet_x1_0", model_path="/home/hatch/workspace/prid-work/weights_osnet_msmt17.pth", device="cpu")
def emb(imgs):
    tmp = []
    for i, im in enumerate(imgs):
        p = f"/home/hatch/workspace/prid-work/_e{i}.png"; cv2.imwrite(p, im); tmp.append(p)
    f = ext(tmp).numpy(); f = f / np.linalg.norm(f, axis=1, keepdims=True)
    for p in tmp: os.remove(p)
    return f

heroA = frames(CAM_A, HERO); heroB = frames(CAM_B, HERO)
hardB = frames(CAM_B, HARD); midB = frames(CAM_B, MID); easyB = frames(CAM_B, EASY)
e_heroA = emb(heroA); proto = e_heroA.mean(0); proto /= np.linalg.norm(proto)
def sims(imgs):
    e = emb(imgs); return e @ proto, e
s_heroB, e_heroB = sims(heroB); s_hardB, e_hardB = sims(hardB)
s_midB, _ = sims(midB); s_easyB, _ = sims(easyB)
protoB = e_heroB.mean(0); protoB /= np.linalg.norm(protoB)
print("proto self:", float(proto @ protoB), "| heroB", float(s_heroB.mean()),
      "| hard", float(s_hardB.mean()), "| mid", float(s_midB.mean()), "| easy", float(s_easyB.mean()))

face_proto, face_stat = None, {}
try:
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(320,320))
    def face_emb(imgs):
        out = []
        for im in imgs:
            big = cv2.resize(im, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            fs = fa.get(big)
            if fs: out.append(fs[0].normed_embedding)
        return out
    fh = face_emb(heroA)
    if len(fh) >= 2:
        face_proto = np.mean(fh, 0); face_proto /= np.linalg.norm(face_proto)
    for name, imgs in [("hero", heroB), ("hard", hardB), ("mid", midB), ("easy", easyB)]:
        fe = face_emb(imgs)
        if face_proto is not None and fe:
            face_stat[name] = (len(fe), float(np.mean([float(face_proto @ e) for e in fe])))
        else:
            face_stat[name] = (len(fe), None)
    print("face_stat:", face_stat)
except Exception as ex:
    print("face pass failed:", ex)

def writer(name, fps=None):
    p = f"{OUT}/{name}.avi"
    return cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), fps or FPS, (W, H)), p

def base(title):
    f = np.full((H, W, 3), (18,18,22), np.uint8)
    return draw_text(f, (28, 14), title, 29, (255,255,255), bold=True)

def cambox(f, x, title, img, label=None, lcolor=(255,255,255)):
    f = panel(f, x, 70, 600, 470, title)
    f = paste_crop(f, img, x+10, 120, 580, 380)
    if label: f = draw_text(f, (x+16, 505), label, 24, lcolor, bold=True)
    return f

T1 = "HƯỚNG 1 — RE-ID NGOẠI HÌNH: đăng ký ở CAM A, nhận ra ở CAM B"
vw, _ = writer("h1")
proto_img = heroA[len(heroA)//2]
for i, im in enumerate(heroA):
    f = base(T1)
    f = cambox(f, 28, "CAM A", im, "Người mẫu — đang trích mã ngoại hình...")
    f = draw_text(f, (660, 200), f"Đã đăng ký: {i+1}/{len(heroA)} ảnh", 27, YEL)
    f = draw_text(f, (660, 252), "Mã ngoại hình (vector) = “dấu vân tay”", 24, (220,220,220))
    f = draw_text(f, (660, 292), "nhận diện một người: quần áo, dáng người.", 24, (220,220,220))
    f = draw_text(f, (660, 350), "Người này sẽ rời khỏi CAM A...", 24, (220,220,220))
    vw.write(f)
cands = [("easy", easyB, s_easyB, "Ứng viên B-1"), ("mid", midB, s_midB, "Ứng viên B-2"),
         ("hard", hardB, s_hardB, "Ứng viên B-3 (mặc rất giống!)"), ("hero", heroB, s_heroB, "Ứng viên B-4")]
for name, imgs, sc, lab in cands:
    for i, im in enumerate(imgs):
        f = base(T1)
        f = cambox(f, 28, "CAM A — NGƯỜI MẪU ĐÃ ĐĂNG KÝ", proto_img)
        col = GREEN if name == "hero" else GREY
        f = cambox(f, 652, "CAM B", im, f"{lab} — điểm giống: {sc[i]:.2f}", col)
        vw.write(f)
    for _ in range(12):
        f = base(T1)
        f = cambox(f, 28, "CAM A — NGƯỜI MẪU ĐÃ ĐĂNG KÝ", proto_img)
        if name == "hero":
            f = cambox(f, 652, "CAM B", imgs[-1], f"{lab} — điểm trung bình: {sc.mean():.2f}", GREEN)
            f = draw_text(f, (652, 585), "KHỚP! Cùng một người → G1", 30, GREEN, bold=True)
        else:
            f = cambox(f, 652, "CAM B", imgs[-1], f"{lab} — điểm trung bình: {sc.mean():.2f}", GREY)
            f = draw_text(f, (652, 585), "Không khớp — người khác", 27, GREY, bold=True)
        vw.write(f)
vw.release(); print("H1 done")

T2 = "HƯỚNG 2 — TRACKING + GHÉP GLOBAL ID"
vw, _ = writer("h2", fps=5)
def at(imgs, i): return imgs[min(i, len(imgs)-1)]
for i in range(len(hardB)):
    f = base(T2)
    f = cambox(f, 28, "CAM A", at(heroA, i), "Local ID: A-12", YEL)
    f = cambox(f, 652, "CAM B", at(hardB, i), "Local ID: B-05", YEL)
    f = draw_text(f, (28, 585), "Mỗi camera tự bám người của mình, ID chỉ có nghĩa trong 1 camera.", 25, (220,220,220))
    vw.write(f)
for i in range(len(heroB)):
    f = base(T2)
    if i >= 8:
        f = cambox(f, 28, "CAM A", at(heroA, i), "Local ID: A-12 → GLOBAL ID: G1", GREEN)
        f = cambox(f, 652, "CAM B", at(heroB, i), "Local ID: B-21 → GLOBAL ID: G1", GREEN)
        f = draw_text(f, (28, 585), f"Mã ngoại hình khớp (điểm {float(proto @ protoB):.2f}) → ghép thành một người: G1", 25, GREEN, bold=True)
    else:
        f = cambox(f, 28, "CAM A", at(heroA, i), "Local ID: A-12", YEL)
        f = cambox(f, 652, "CAM B", at(heroB, i), f"Local ID: B-21 — đang so mã... {s_heroB[i]:.2f}", YEL)
        f = draw_text(f, (28, 585), "Người ở CAM A đã rời khung hình, xuất hiện ở CAM B với Local ID mới.", 25, (220,220,220))
    vw.write(f)
vw.release(); print("H2 done")

T3 = "HƯỚNG 3 — FUSION: ĐIỂM NGOẠI HÌNH + ĐIỂM KHUÔN MẶT"
vw, _ = writer("h3", fps=5)
for name, imgs, sc, lab in [("hard", hardB, s_hardB, "Ứng viên khó: mặc rất giống người mẫu"),
                            ("hero", heroB, s_heroB, "Ứng viên: người thật")]:
    fstat = face_stat.get(name, (0, None))
    for i, im in enumerate(imgs):
        f = base(T3)
        f = cambox(f, 28, "CAM A — NGƯỜI MẪU", proto_img)
        f = cambox(f, 652, "CAM B", im, lab, GREEN if name == "hero" else YEL)
        f = bar(f, 28, 572, 300, "Ngoại hình", float(sc[i]), YEL)
        if fstat[1] is not None:
            f = bar(f, 28, 612, 300, "Khuôn mặt", fstat[1], (230,160,60))
            f = bar(f, 28, 652, 300, "FUSION 0,6/0,4", 0.6*float(sc[i]) + 0.4*fstat[1], GREEN if name == "hero" else GREY)
        else:
            f = draw_text(f, (28, 615), "Khuôn mặt: KHÔNG bắt được (mặt quá nhỏ/xa trên camera giám sát)", 23, (200,120,120))
            f = draw_text(f, (28, 652), "→ Fusion tự fallback về điểm ngoại hình", 23, (220,220,220))
        vw.write(f)
    for _ in range(12):
        f = base(T3)
        f = cambox(f, 28, "CAM A — NGƯỜI MẪU", proto_img)
        f = cambox(f, 652, "CAM B", imgs[-1], lab, GREEN if name == "hero" else GREY)
        f = bar(f, 28, 572, 300, "Ngoại hình (TB)", float(sc.mean()), YEL)
        if fstat[1] is not None:
            f = bar(f, 28, 612, 300, "Khuôn mặt", fstat[1], (230,160,60))
            f = bar(f, 28, 652, 300, "FUSION 0,6/0,4", 0.6*float(sc.mean()) + 0.4*fstat[1], GREEN if name == "hero" else GREY)
        if fstat[1] is None:
            f = draw_text(f, (28, 615), "Khuôn mặt: KHÔNG bắt được (mặt quá nhỏ/xa trên camera giám sát)", 23, (200,120,120))
            f = draw_text(f, (28, 652), "→ Fusion tự fallback về điểm ngoại hình", 23, (220,220,220))
        f = draw_text(f, (900, 618), "Đúng người → G1" if name == "hero" else "Người khác — loại", 27,
                      GREEN if name == "hero" else GREY, bold=True)
        vw.write(f)
vw.release(); print("H3 done")
json.dump({"proto_self": float(proto @ protoB), "heroB": float(s_heroB.mean()), "hardB": float(s_hardB.mean()),
           "midB": float(s_midB.mean()), "easyB": float(s_easyB.mean()),
           "face_stat": {k: list(v) for k, v in face_stat.items()}},
          open("/home/hatch/workspace/prid-work/results.json", "w"), indent=1)
print("ALL DONE")
