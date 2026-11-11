# Shared helpers for the PRID demo videos: PIL text (Vietnamese-safe), panels, letterbox.
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_cache = {}
def font(sz, bold=False):
    k = (sz, bold)
    if k not in _cache: _cache[k] = ImageFont.truetype(FONTB if bold else FONT, sz)
    return _cache[k]

def draw_text(frame_bgr, xy, text, sz=26, color=(255,255,255), bold=False, anchor="la"):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.text(xy, text, font=font(sz, bold), fill=(color[2], color[1], color[0]), anchor=anchor)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def text_w(text, sz, bold=False):
    f = font(sz, bold); d = ImageDraw.Draw(Image.new("RGB", (10,10)))
    return d.textlength(text, font=f)

def panel(frame, x, y, w, h, title, title_color=(255,255,255)):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (28,28,32), -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (90,90,100), 2)
    frame = draw_text(frame, (x+16, y+10), title, 25, title_color, bold=True)
    return frame

def paste_crop(frame, img_bgr, x, y, w, h):
    """Letterbox-fit a crop into box (x,y,w,h) on dark bg."""
    if img_bgr is None: return frame
    ih, iw = img_bgr.shape[:2]
    s = min(w/iw, h/ih)
    nw, nh = max(1,int(iw*s)), max(1,int(ih*s))
    r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    ox, oy = x + (w-nw)//2, y + (h-nh)//2
    frame[oy:oy+nh, ox:ox+nw] = r
    return frame

def bar(frame, x, y, w, label, val, color, vmax=1.0, sz=23):
    frame = draw_text(frame, (x, y), label, sz, (230,230,230))
    bx, bw, bh = x + 300, w, 20
    cv2.rectangle(frame, (bx, y+2), (bx+bw, y+2+bh), (60,60,66), -1)
    cv2.rectangle(frame, (bx, y+2), (bx+int(bw*max(0,min(1,val/vmax))), y+2+bh), color, -1)
    frame = draw_text(frame, (bx+bw+14, y), f"{val:.2f}" if val is not None else "—", sz, (255,255,255), bold=True)
    return frame
