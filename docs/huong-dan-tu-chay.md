# TỰ CHẠY DEMO RE-ID TRÊN COLAB CLI — từng bước
*Mọi lệnh chạy trong terminal, thư mục ~/avt-demo (nơi đã có sẵn: 3 script, video.mp4, weights_osnet_msmt17.pth).*

## Bước 0 — Vào đúng thư mục + kiểm tra phiên
```
cd ~/avt-demo
colab status -s avt
```
Phiên còn sống thì đi tiếp. Nếu báo lỗi/hết phiên:
```
colab new -s avt --gpu T4
colab install -s avt torchreid gdown tensorboard "numpy==1.26.4" ultralytics
```

## Bước 1 — Đẩy video + weights lên máy Colab (bắt buộc, đừng để script tự tải — Wikimedia chặn IP Colab)
```
colab upload -s avt video.mp4 /content/video.mp4
colab upload -s avt weights_osnet_msmt17.pth /content/weights_osnet_msmt17.pth
```

## Bước 2 — HƯỚNG 1: Re-ID bằng ngoại hình
```
colab exec -s avt --timeout 600 -f huong1_osnet.py
```
(--timeout 600 rất quan trọng: mặc định CLI chỉ chờ 30 giây không có output là cắt.)
**Mốc kiểm tra:** cuối output có ma trận điểm — cùng người ~0,72, khác người ~0,35–0,40 là ĐÚNG. Chụp màn hình bảng này cho slide.

## Bước 3 — HƯỚNG 2: Tracking
```
colab exec -s avt --timeout 600 -f huong2_track.py
colab ls -s avt /content/runs/detect/track
colab download -s avt /content/runs/detect/track/video.avi ./video-tracking.avi
```
**Mốc kiểm tra:** mở video.avi — mỗi người có khung + số ID bám theo. (Muốn bản mp4 thì đã có sẵn video-tracking.mp4 trong thư mục này, OpenCode chuyển hộ từ trước.)

## Bước 4 — HƯỚNG 3: Fusion (chạy SAU Hướng 1, cùng phiên — nó dùng lại crops Hướng 1 đã cắt)
```
colab install -s avt insightface onnxruntime-gpu
colab exec -s avt --timeout 600 -f huong3_fusion.py
```
**Mốc kiểm tra:** bảng 3 cột (ngoại hình / khuôn mặt / fusion). Dòng nào ghi "không thấy mặt → chỉ dùng ngoại hình" là bình thường, đó là fallback để trình bày.

## Bước 5 — Dọn phiên (chỉ khi đã tải hết kết quả về)
```
colab stop -s avt
```

## Lỗi thường gặp
- 404 kernels / phiên chết → làm lại Bước 0 phần tạo phiên mới.
- wget exit 8 → đã né bằng Bước 1, đừng bỏ bước này khi mở phiên mới.
- Lỗi nhắc tensorflow/jax khi import torchreid → script trong thư mục này đã vá sẵn (dòng sys.modules["tensorflow"] = None), cứ chạy bản ở đây.
