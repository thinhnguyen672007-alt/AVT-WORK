# GIẢI THÍCH CÁC HƯỚNG DEMO RE-ID — model gì, giúp gì, hạn chế gì
*Để nói khi trình bày 12/11 và dán vào slide/notes. Mỗi hướng: Demo cho model gì → Giúp gì cho dự án → Hạn chế.*

## Hướng 1 — Re-ID bằng ngoại hình
- **Demo cho model:** OSNet (dùng qua thư viện Torchreid), weights đã train trên tập MSMT17. YOLOv8 + BoT-SORT chỉ hỗ trợ cắt ảnh từng người từ video.
- **Giúp gì:** biến mỗi người thành một "mã ngoại hình" (dãy số). Hai camera khác nhau so mã của nhau là biết cùng một người hay không — đây là nền tảng để gộp Global ID và đếm người không bị trùng.
- **Hạn chế:** dễ nhầm khi hai người mặc giống nhau, quay lưng hoặc bị che khuất; điểm thấp khi ảnh cắt nhỏ/mờ; và mã ngoại hình mới là *một thành phần điểm*, chưa tự quyết định được Global ID.

## Hướng 2 — Tracking trong một camera
- **Demo cho model:** YOLOv8 (tìm người) + BoT-SORT (bám theo và gắn số tạm — Local ID).
- **Giúp gì:** bám liên tục từng người trong một camera, đếm được số người trong khung hình, và cắt đúng ảnh của từng người theo thời gian để đưa sang bước Re-ID.
- **Hạn chế:** Local ID chỉ có nghĩa trong đúng camera đó — người đi sang camera khác là mất dấu; bị đổi ID khi người bị che khuất lâu (đây cũng là lỗi team đã ghi nhận ở Phase 1: đổi ID sau 3–5 giây che khuất). Bản ultralytics thuần không có Re-ID học sâu đi kèm.

## Hướng 3 — Fusion: ngoại hình + khuôn mặt
- **Demo cho model:** OSNet (ngoại hình) cộng điểm với InsightFace buffalo_l — ArcFace (khuôn mặt), theo tỉ lệ 0,6 ngoại hình + 0,4 khuôn mặt.
- **Giúp gì:** khi ngoại hình dễ nhầm (đồng phục, mặc giống nhau), điểm khuôn mặt kéo độ tin cậy lên. Là hướng nâng cấp độ chính xác cho Re-ID.
- **Hạn chế:** mặt trong video camera giám sát thường nhỏ, xa, quay nghiêng → nhiều lúc không bắt được mặt, hệ thống phải quay về dùng riêng ngoại hình; tỉ lệ 0,6/0,4 là chọn bằng thử nghiệm (heuristic), muốn dùng thật phải hiệu chỉnh trên dữ liệu camera của dự án.

## Hướng 4 — DeepStream + MV3DT (chỉ trình bày kiến trúc, không chạy demo)
- **Demo cho model/hệ thống:** NVIDIA DeepStream (hạ tầng xử lý nhiều luồng camera) + MV3DT (gộp danh tính xuyên camera, dùng vị trí 3D theo thời gian thực); kết hợp ý tưởng Re-ID + thời gian di chuyển (travel-time) + bản đồ vị trí các camera.
- **Giúp gì:** là đích đến khi dự án mở rộng ra nhiều camera thật: gộp Global ID xuyên camera có xét cả yếu tố "người này có kịp đi từ camera A sang camera B trong ngần ấy giây không".
- **Hạn chế:** cần máy chủ GPU NVIDIA riêng và cài đặt phức tạp; cần dữ liệu nhiều camera quay đồng bộ mới đánh giá được; nằm ngoài phạm vi "chạy code có sẵn" của đợt này — clip minh họa dùng video 2 camera của Hiếu (cùng người được gán G1 ở cả CAM 0 và CAM 1).

## Câu chốt khi bị hỏi "sao chưa phải hệ thống hoàn chỉnh?"
Mỗi hướng mới giải quyết MỘT đoạn trong chuỗi 4 việc: thấy người → bám theo (Local ID) → nhận ra qua camera (mã ngoại hình/mặt) → gộp Global ID. Ghép cả 4 đoạn + luật thời gian di chuyển mới thành hệ thống đa camera hoàn chỉnh — đó là phần kiến trúc ở Hướng 4.
