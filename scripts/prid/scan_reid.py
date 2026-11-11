import sys, os, json, glob
sys.modules["tensorflow"] = None
import numpy as np
BASE = "/home/hatch/workspace/prid-work/data/multi_shot"
CAM_A, CAM_B = f"{BASE}/cam_a", f"{BASE}/cam_b"
def persons(cam):
    out = {}
    for d in sorted(glob.glob(cam + "/*")):
        if os.path.isdir(d):
            imgs = sorted(glob.glob(d + "/*.png"))
            if imgs: out[os.path.basename(d)] = imgs
    return out
pa, pb = persons(CAM_A), persons(CAM_B)
common = [p for p in pa if p in pb and len(pa[p]) >= 6 and len(pb[p]) >= 6]
from torchreid.reid.utils import FeatureExtractor
ext = FeatureExtractor(model_name="osnet_x1_0", model_path="/home/hatch/workspace/prid-work/weights_osnet_msmt17.pth", device="cpu")
def proto(imgs, n=12):
    picks = sorted(imgs, key=lambda f: -os.path.getsize(f))[:n]
    f = ext(picks).numpy(); f /= np.linalg.norm(f, axis=1, keepdims=True)
    p = f.mean(0); return p / np.linalg.norm(p)
PA = {p: proto(pa[p]) for p in common}
PB = {p: proto(pb[p]) for p in common}
rows = []
for h in common:
    sims = sorted(((float(PA[h] @ PB[p]), p) for p in common), reverse=True)
    self_s = float(PA[h] @ PB[h])
    best_other = max(s for s, p in sims if p != h)
    rows.append({"hero": h, "self": self_s, "best_other": best_other, "margin": self_s - best_other,
                 "hard": max(((s, p) for s, p in sims if p != h))[1],
                 "rank": [p for s, p in sims].index(h) + 1})
rows.sort(key=lambda r: -r["margin"])
for r in rows[:8]: print(f"{r['hero']}: self={r['self']:.3f} best_other={r['best_other']:.3f} margin={r['margin']:.3f} rank={r['rank']} hard={r['hard']}")
best = rows[0]
sims_b = sorted(((float(PA[best["hero"]] @ PB[p]), p) for p in common), reverse=True)
mid = sims_b[len(sims_b)//2][1]; easy = sims_b[-1][1]
json.dump({"CAM_A": CAM_A, "CAM_B": CAM_B, "hero": best["hero"], "self": best["self"],
           "hard": best["hard"], "hard_sim": best["best_other"], "mid": mid, "easy": easy,
           "easy_sim": sims_b[-1][0], "margin": best["margin"], "top_rows": rows[:8]},
          open("/home/hatch/workspace/prid-work/scan.json", "w"), indent=1)
print("chosen hero:", best["hero"], "hard:", best["hard"], "mid:", mid, "easy:", easy)
