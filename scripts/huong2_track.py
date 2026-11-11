"""HƯỚNG 2 — Tracking YOLO + BoT-SORT trên CHUNG video cố định.
Chạy: colab exec -s avt -f huong2_track.py
Kết quả: video có khung + Local ID bám từng người, ở thư mục runs/detect/track/ — tải về bằng colab download.
"""
import os, subprocess

WORK = os.environ.get("DEMO_DIR", "/content")
VIDEO = os.path.join(WORK, "video.mp4")
if not os.path.exists(VIDEO):
    ogv = os.path.join(WORK, "video.ogv")
    subprocess.run(["wget", "-q", "--user-agent=Mozilla/5.0 (compatible; AVT-Demo/1.0)", "https://upload.wikimedia.org/wikipedia/commons/0/0b/Calle_preciados.ogv", "-O", ogv], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", ogv, "-q:v", "3", VIDEO], check=True)
    print("Đã tải + đổi video cố định sang mp4.")

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.track(source=VIDEO, tracker="botsort.yaml", classes=[0], save=True, verbose=False)
print("Xong. Video kết quả: runs/detect/track/video.mp4 (kiểm tra bằng colab ls).")
