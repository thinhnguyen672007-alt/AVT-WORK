# AVT WORK — Nghiên cứu & demo Person Re-ID đa camera

Repo lưu trữ quá trình nghiên cứu và demo bài toán **Person Re-Identification (Re-ID) đa camera** trong dự án FaceID AI tại AVT: làm sao nhận ra *cùng một người* khi họ xuất hiện ở các camera khác nhau.

Mô hình làm việc: **research** — nghiên cứu các phương án, chạy thử bằng **code/model có sẵn** trên Google Colab, trình bày kết quả kèm video demo. Không train model, không xây hệ thống production trong repo này.

## Chuỗi 4 việc của bài toán

1. **Thấy người** — phát hiện người trong khung hình (YOLO).
2. **Bám theo trong 1 camera** — gắn Local ID tạm thời (BoT-SORT).
3. **Nhận ra qua camera khác** — trích "mã ngoại hình" (OSNet) hoặc cộng thêm khuôn mặt, so độ giống.
4. **Gộp Global ID** — hợp nhất danh tính xuyên camera (có xét topology + thời gian di chuyển).

## 4 phương án đã nghiên cứu

| # | Phương án | Model / thư viện | Trạng thái demo |
|---|---|---|---|
| 1 | Re-ID bằng ngoại hình | OSNet qua Torchreid (weights MSMT17) | ✅ Có video + ma trận điểm |
| 2 | Tracking + Re-ID | YOLOv8 + BoT-SORT | ✅ Có video tracking (Local ID) |
| 3 | Fusion ngoại hình + khuôn mặt | OSNet + InsightFace buffalo_l (0,6/0,4) | ✅ Có video + bảng ablation |
| 4 | DeepStream + MV3DT, travel-time, camera topology | NVIDIA DeepStream | 📐 Chỉ trình bày kiến trúc |

Giải thích chi tiết từng phương án (giúp gì, hạn chế gì): `docs/giai-thich-cac-huong-demo.md`.

## Cách chạy demo

- **Colab CLI** (chạy từ terminal): xem `docs/huong-dan-tu-chay.md`.
- **Colab web**: mở `notebook/AVT-ReID-Demo.ipynb`, chọn runtime T4 GPU, chạy từng ô.
- Dữ liệu demo v1: video mẫu công khai "Calle Preciados" (phố đi bộ Madrid, 19 giây, 720p, Wikimedia Commons) — dùng chung cho cả 3 hướng để so sánh được.
- Lưu ý kỹ thuật đã kiểm chứng: phải dùng weights OSNet train trên **MSMT17** (weights ImageNet cho điểm sai hẳn); ghim `numpy==1.26.4`; Wikimedia chặn tải trực tiếp từ IP Colab nên đẩy file bằng `colab upload`.

## Cấu trúc repo

- `scripts/` — các script demo từng hướng (code nối do nhóm viết; model/thư viện là bản công bố có sẵn).
- `notebook/` — notebook Colab web.
- `docs/` — tài liệu nghiên cứu, hướng dẫn chạy, review chuyên sâu.
- `requirements.txt` — thư viện cần cài trên Colab.

## Kết quả tham chiếu (video mẫu v1)

- Hướng 1: cùng người **0,72** — khác người **0,35–0,40** (ma trận độ giống OSNet).
- Hướng 2: video tracking 19 giây, mỗi người một Local ID.
- Hướng 3: bảng ablation 3 cột (ngoại hình / khuôn mặt / fusion) + video có thanh điểm trực tiếp.

## Việc tiếp theo

- Làm lại bộ video demo trên dữ liệu 2 camera thật (PRID 2011): người rời CAM A → xuất hiện ở CAM B → vẫn nhận ra cùng Global ID.
- Trình bày kết quả cùng team (Hiếu) cho tech lead.
