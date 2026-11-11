# HƯỚNG DẪN COLAB CLI TỪNG BƯỚC — 3 hướng demo Re-ID, CHUNG 1 VIDEO CỐ ĐỊNH
*Colab CLI là công cụ CHÍNH CHỦ của Google (github.com/googlecolab/google-colab-cli): mở máy ảo Colab có GPU ngay từ terminal. Chỉ hỗ trợ Linux/macOS — máy Fedora của anh dùng được.*

**Video cố định cho cả 3 hướng:** "Calle Preciados" — phố đi bộ đông người ở Madrid, 19 giây, 720p, tải từ Wikimedia Commons:
`https://upload.wikimedia.org/wikipedia/commons/0/0b/Calle_preciados.ogv`
Các script TỰ tải + đổi sang mp4 ở lần chạy đầu — anh không cần upload ảnh/video gì hết. Dùng 1 video chung để: kết quả so sánh được giữa các hướng, và ai chạy lại cũng ra đúng dữ liệu đó.

## BƯỚC 0 — Cài Colab CLI (1 lần, trên máy anh)
```
pip install google-colab-cli
```

## BƯỚC 1 — Mở phiên Colab có GPU
```
colab new -s avt --gpu T4
```
Lần đầu sẽ bắt đăng nhập Google (tài khoản cá nhân). Kiểm tra: `colab status -s avt`.

## BƯỚC 2 — HƯỚNG 1: Re-ID bằng ngoại hình (OSNet)
```
colab install -s avt torchreid gdown tensorboard "numpy==1.26.4" ultralytics
colab exec -s avt -f huong1_osnet.py
```
Script làm hết: tải video + weights OSNet train trên MSMT17 → YOLO+BoT-SORT bám người → lấy các crop TO NHẤT của cùng 1 ID + crop 1 người khác → in ma trận độ giống. Milo đã chạy thử: cùng người **0,86**, khác người **~0,40** — tách rõ. Chụp màn hình bảng điểm cho slide. (Dùng sai weights ImageNet là điểm ngược hẳn — script đã tự tải đúng bản MSMT17.)
*(Pin numpy==1.26.4 vì torchreid cũ có thể gãy trên numpy 2.x — theo review của OpenCode.)*

## BƯỚC 3 — HƯỚNG 2: Tracking YOLO + BoT-SORT
```
colab install -s avt ultralytics
colab exec -s avt -f huong2_track.py
colab ls -s avt /content/runs/detect/track
colab download -s avt /content/runs/detect/track/video.mp4 ./video-tracking.mp4
```
→ Video kết quả: mỗi người một khung + Local ID bám theo. Lưu ý khi trình bày: đây là ID trong MỘT camera (Local ID) — phần gộp Global ID là ở Hướng 1/3 và phần kiến trúc Hướng 4.
*(Dự phòng khi Colab trục trặc: notebook BoxMOT chính thức — link trong file AVT-ReID-Demo-Code-San-Co.md. Và nói rõ: bản ultralytics thuần chỉ bám theo chuyển động, không có Re-ID học sâu như BoxMOT.)*

## BƯỚC 4 — HƯỚNG 3: Fusion + bảng so sánh (ablation)
```
colab install -s avt insightface onnxruntime-gpu
colab exec -s avt -f huong3_fusion.py
```
Chạy SAU Hướng 1 (dùng lại crops Hướng 1 đã cắt). In bảng 3 cột cho từng cặp: chỉ ngoại hình / chỉ khuôn mặt / fusion (0,6×ngoại hình + 0,4×mặt). Không thấy mặt trong crop thì nó tự ghi "chỉ dùng ngoại hình" — đó chính là fallback để trình bày, không phải lỗi. Câu nói với anh Tiến: "trọng số là heuristic chọn bằng thử trên tập cặp demo; có bảng ablation đi kèm".

## BƯỚC 5 — Dọn phiên
```
colab stop -s avt
```
Mọi kết quả cần giữ (video tracking, ảnh chụp bảng điểm) phải lấy về TRƯỚC khi stop.

## Việc sống còn trước giờ trình bày (theo review của OpenCode)
1. **Dry-run toàn bộ trên 1 phiên Colab mới tinh** đúng thứ tự trình bày + **quay video màn hình** làm backup cuối cùng (chị Lily chỉ nhận video demo).
2. Kiểm chứng lại số liệu phần 4 từ paper gốc (MV3DT 96.5% IDF1 trên WILDTRACK) trước khi đưa lên slide.
3. Tập câu hỏi nguy hiểm nhất: "Phần 2 chỉ có Local ID — Re-ID đa camera ở đâu?" → "Hướng 1+3 chính là cross-camera: so mã ngoại hình của cùng người ở 2 thời điểm/góc khác nhau; pipeline đa camera đầy đủ gồm topology + travel-time là phần kiến trúc ở Hướng 4."
4. Kẹt ở đâu quá 10 phút: chụp màn hình gửi Milo.
