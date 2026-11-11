# GIẢI THÍCH TRACKING + RE-ID BẰNG LỜI THƯỜNG — cho Thịnh
*Đọc 5 phút là nắm được toàn cảnh để mai trình bày. Mọi cái tên lạ (YOLO, BoT-SORT, OSNet...) chỉ là CÔNG CỤ cho 4 việc dưới đây.*

## Bài toán gốc
Nhiều camera trong một tòa nhà. Một người đi từ camera 1 sang camera 3. Hệ thống phải trả lời: "người ở camera 3 có phải cùng người ở camera 1 không, và đó là người số mấy của cả tòa nhà?" Muốn vậy phải làm 4 việc nối tiếp nhau:

## Việc 1 — THẤY người trong khung hình (detection)
- Việc: nhìn vào ảnh/video, khoanh khung quanh từng người.
- Công cụ: **YOLO** (phổ biến nhất), RT-DETR (bản khác cùng việc).
- Ví von: con mắt của bảo vệ — chỉ "thấy có người ở đó", chưa biết là ai, chưa nhớ gì.

## Việc 2 — BÁM theo người trong MỘT camera (tracking)
- Việc: người đó di chuyển trong cùng 1 camera thì giữ nguyên một mã tạm — gọi là **Local ID** (mã tạm của riêng camera này).
- Công cụ: **ByteTrack, BoT-SORT, DeepSORT** — cùng một việc, khác nhau ở độ khéo khi người bị che khuất hoặc đi chồng lên nhau. BoT-SORT = ByteTrack + gắn thêm chút Re-ID + bù trừ khi camera rung/động.
- Ví von: nhân viên bảo vệ ở tầng 1 dán cho mỗi khách một số áo tạm của tầng mình: "khách số 12 tầng 1".
- Lỗi team đang gặp (đổi ID sau khi bị che 3–5 giây) nằm đúng ở việc này + việc 4.

## Việc 3 — NHẬN RA người khi SANG CAMERA KHÁC (Re-ID)
- Việc: camera khác không biết "số 12 tầng 1" là ai. Phải nhận diện lại bằng NGOẠI HÌNH: quần áo, màu sắc, dáng người → trích ra một dãy số gọi là **mã ngoại hình** (đúng cơ chế "mã vạch" như khuôn mặt, nhưng cho cả thân người), rồi so độ giống giữa các camera.
- Công cụ trích mã: **OSNet, FastReID, TransReID** — cùng một việc, OSNet nhẹ và phổ biến nhất. **Torchreid** không phải model — nó là THƯ VIỆN chứa sẵn OSNet để gọi ra dùng.
- Ví von: bảo vệ tầng 3 không nhớ số áo tầng 1, nhưng nhớ "anh áo đen quần jean lúc nãy" — nhận người bằng vẻ ngoài.
- Phần Face Recognition của Thịnh chính là việc 3 này nhưng dùng KHUÔN MẶT thay vì ngoại hình — cùng một nguyên lý "trích mã → so độ giống".

## Việc 4 — GỘP thành một mã DUY NHẤT cho cả tòa nhà (Global ID)
- Việc: quyết định các mã tạm ở mọi camera nào thuộc về cùng một người, rồi cấp một mã chung — **Global ID**.
- Đây KHÔNG phải model AI — đây là LOGIC ra quyết định (gọi là Association Engine): cộng dồn bằng chứng — mã ngoại hình giống bao nhiêu + thời gian di chuyển giữa 2 camera có hợp lý không + 2 camera có gần nhau không.
- Ví von: quản lý tòa nhà cầm sổ của mọi tầng, đối chiếu rồi kết luận "số 12 tầng 1 = số 5 tầng 2 = số 28 tầng 4 → khách số 1001".
- Đây là chỗ quyết định hệ thống nhầm người hay không — và là ý quan trọng nhất trong tài liệu 3 giải pháp.

## Các tên còn lại hay gặp
- **BoxMOT**: không phải model — là HỘP CÔNG CỤ đóng gói sẵn việc 2 + việc 3 (chứa sẵn ByteTrack/BoT-SORT/DeepSORT + OSNet), gõ 1 lệnh là chạy demo. Mai dùng cái này là vì thế.
- **DeepStream / MV3DT** (NVIDIA): hạ tầng tăng tốc để chạy nhiều camera cùng lúc trên máy NVIDIA — phần "đường ống nhà máy", không phải model nhận diện. Mai chỉ trình bày kiến trúc, không chạy.
- **Gait** (dáng đi), **pose** (tư thế): các tín hiệu phụ cho giải pháp 3 (đa tín hiệu) — như bảo vệ nhớ thêm "dáng đi khập khiễng" ngoài quần áo.

## Nhớ đúng 1 câu này là trình bày được
"Thấy người (YOLO) → bám theo trong 1 camera bằng mã tạm (tracker) → nhận ra khi sang camera khác bằng mã ngoại hình (Re-ID) → gộp tất cả thành một Global ID duy nhất bằng logic cộng bằng chứng."
