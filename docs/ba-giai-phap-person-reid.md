# 3 GIẢI PHÁP PERSON RE-ID — BẢN STUDY CHO THỊNH
*Nguồn gốc: tệp "Ba giải pháp Person ReID.pdf" anh Tiến gửi vào nhóm chat "Project FaceID AI" lúc 11:25 ngày 10/11/2026. Lưu ý nguồn: tệp này là bản in từ ChatGPT (câu hỏi gốc: "nêu 3 giải pháp thông dụng nhất của Person Re-Identification") — anh Tiến chia sẻ để team tham khảo, không phải văn bản anh tự viết, và trong chat anh không giao tệp này cho riêng ai. Nó nằm trong mạch trao đổi về slide ReID của Hiếu.*

## Trước hết: Re-ID là bài toán gì?
Dự án có 2 bài toán khác nhau, đừng lẫn:
- **Face Recognition** (phần của Đức + backend của Thịnh): nhận diện bằng KHUÔN MẶT — biết "đây là ai".
- **Person Re-ID** (phần của Hiếu): nhận diện lại một người qua NGOẠI HÌNH (quần áo, dáng) giữa NHIỀU CAMERA — khi người quay lưng, đeo khẩu trang, hay mặt không rõ, vẫn phải biết "người ở camera 1 và người ở camera 3 là cùng một người". Đầu ra quan trọng nhất của Re-ID là **Global ID**: một mã duy nhất cho mỗi người trên toàn hệ thống camera.

3 giải pháp dưới đây là 3 cách làm bài toán Re-ID, từ đơn giản đến phức tạp.

## Giải pháp 1 — Re-ID bằng ngoại hình (Appearance-based)
- Cách làm: camera phát hiện người (YOLO) → cắt vùng chứa người → model Re-ID (OSNet, FastReID, TransReID) trích "mã ngoại hình" (embedding) → so sánh mã giữa các camera bằng độ giống cosine; đủ cao thì kết luận cùng một người.
- Ưu: không cần thấy mặt, người quay lưng vẫn làm được; đơn giản nhất, dễ triển khai.
- Nhược: hai người mặc đồ giống nhau là nhầm; nhạy với ánh sáng, góc camera, tư thế.
- Độ phức tạp: thấp. Đây là cách team đang làm ở Phase 1 (kết quả hiện tại: đúng khoảng 50–60% trên 1 camera).

## Giải pháp 2 — Tracking + Re-ID (gán ID toàn cục)
- Cách làm: mỗi camera có một tracker (ByteTrack, BoT-SORT, DeepSORT) theo dõi người trong khung hình của nó và cấp **Local ID** (mã tạm trong 1 camera). Khi người sang camera khác, Re-ID đối chiếu ngoại hình để gộp các Local ID thành một **Global ID** duy nhất. Ví dụ trong tệp: Camera 01–Person 12, Camera 02–Person 05, Camera 03–Person 28, Camera 04–Person 09 → cùng là Global ID "Person 1001".
- Ưu: giữ ID liên tục trong từng camera; kết hợp thêm thông tin thời gian + vị trí để giảm nhầm; đây là cách "phù hợp triển khai thực tế" nhất.
- Nhược: phải xử lý lúc người biến mất rồi xuất hiện lại ở camera khác; cần cơ chế quản lý Global ID chống trùng; đồng bộ thời gian giữa các camera rất quan trọng.
- **Điểm mấu chốt của tệp (quan trọng nhất trong cả 3 giải pháp):** không nên chỉ dùng độ giống cosine để quyết Global ID — phải xây một "bộ máy gán ID" (Association Engine) kết hợp: độ giống ngoại hình + thời gian di chuyển giữa 2 camera + vùng giao nhau của 2 camera + lịch sử tracking. Đây chính là chỗ quyết định hệ thống giữ ID ổn định hay liên tục nhầm người — và nó khớp đúng lỗi số 1 của team hiện nay (ID bị đổi sau khi người bị che khuất 3–5 giây).

## Giải pháp 3 — Re-ID đa tín hiệu (Multi-Modal / Fusion)
- Cách làm: không dựa vào một tín hiệu duy nhất mà cộng dồn nhiều tín hiệu: ngoại hình (quần áo, màu sắc) + khuôn mặt (khi đủ rõ) + tư thế cơ thể + dáng đi (gait) + vị trí/hướng di chuyển + thời gian xuất hiện giữa các camera. Mỗi tín hiệu cho một điểm số, hệ thống tổng hợp thành xác suất "cùng một người".
- Ưu: độ tin cậy cao nhất trong môi trường phức tạp, các camera góc nhìn khác nhau.
- Nhược: kiến trúc phức tạp, tốn tài nguyên tính toán, cần nhiều dữ liệu để hiệu chỉnh.

## Bảng so sánh nhanh
| Tiêu chí | GP1: Ngoại hình | GP2: Tracking + Re-ID | GP3: Đa tín hiệu |
|---|---|---|---|
| Độ phức tạp | Thấp | Trung bình | Cao |
| Nhu cầu GPU | Thấp–TB | Trung bình | Trung bình–Cao |
| Global ID | Cần bổ sung | Có | Có |
| Bền khi đổi góc camera | Khá | Khá | Tốt (nếu kết hợp tốt) |
| Khả năng triển khai | Dễ | Tốt nhất | Khó hơn |

## Kiến trúc được đề xuất trong tệp
Với hệ thống 2–4 camera + GPU mạnh (ví dụ RTX 3090 24GB): **làm giải pháp 1 + 2 trước, bổ sung giải pháp 3 khi cần.**
Luồng: Camera 01–04 → YOLO phát hiện người → ByteTrack/BoT-SORT (Local ID) → OSNet/FastReID (mã ngoại hình) → Association Engine (ngoại hình + thời gian + vùng giao nhau + lịch sử) → **Cơ sở dữ liệu Global ID**.

## Việc này liên quan gì đến phần của Thịnh?
- Cả kiến trúc trên đều cần một nơi lưu trữ và quản lý: **Global Person ID Database** — danh sách người, mã ngoại hình, và lịch sử xuất hiện ở từng camera theo thời gian. Đó chính xác là kỹ năng backend (cơ sở dữ liệu + API + log sự kiện) — cùng một "cuốn sổ" như phần FaceID, chỉ khác loại dữ liệu ghi vào.
- Giải pháp 3 dùng cả nhận diện khuôn mặt làm một tín hiệu → hai hướng của dự án (FaceID và Re-ID) gặp nhau ở đây; phần backend làm tốt từ đầu (log có cấu trúc: ai, camera nào, lúc nào, điểm số bao nhiêu) thì phục vụ được cả hai hướng.
- Nói cách khác: phần backend không phải phần phụ — nó là nơi cả hai hướng nghiên cứu phải "đi qua" mới thành hệ thống.

## Phạm vi của Thịnh với 3 giải pháp này (anh chốt tối 10/11)
3 giải pháp này anh chỉ cần **nghiên cứu để hiểu** — để trình bày được toàn cảnh dự án và trả lời khi được hỏi — **không phải phần anh phải làm ra giải pháp**. Phần "làm" của anh vẫn chỉ là Backend FaceID (study + demo code của chính phần đó). Đề tài Re-ID và code demo của nó thuộc về Hiếu.
