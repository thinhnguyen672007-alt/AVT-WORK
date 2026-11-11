# REVIEW CỦA OPENCODE VỀ KẾ HOẠCH DEMO RE-ID (11/11/2026)
*Nguyên văn câu trả lời của OpenCode (chạy trên máy của Thịnh, agent plan) cho 5 câu hỏi: (a) lỗ hổng kỹ thuật, (b) code tốt hơn, (c) fusion 0.6/0.4, (d) câu hỏi xoáy của lead, (e) thứ tự ưu tiên 3 giờ.*

## (a) Lỗ hổng kỹ thuật / rủi ro sập demo

**Phần 1 — Torchreid/OSNet:**
- Repo `deep-person-reid` ngừng maintain. Colab hiện numpy 2.x + torch 2.x có thể làm import fail. Fix: pin `numpy==1.26.4`, install từ GitHub, **test lại trên runtime mới từ đầu**, đừng tin runtime cũ.
- Trọng số OSNet tải từ server cũ của tác giả — link chết từng xảy ra. Pre-download `.pth` vào Drive, load bằng `model_path` local.
- Kiểm tra trọng số đang load là bản train trên MSMT17 hay ImageNet. Bản ImageNet yếu hơn. Nên thống nhất 1 model/1 trọng số cho cả phần 1+2+3 — lead sẽ hỏi vì sao phần 1 dùng `x1_0` mà phần 2 dùng `x0_25_msmt17`.
- OSNet cần **crop bbox người** (256×128), feed ảnh nguyên khung nhiều nền = embedding rác. Ảnh "toàn thân" phải là crop, không phải photo nguyên cảnh.
- Điểm ~0.43 sàn bạn thấy đúng bản chất: OSNet là model người đi bộ, không phải face. Đừng trình bày nó như "bug".

**Phần 2 — BoxMOT:**
- Rủi ro lớn nhất: **notebook cũ vs version mới**. BoxMOT đổi API mạnh giữa các version. Pin đúng `boxmot==x.y.z` khớp notebook, chạy lại toàn bộ.
- Backup `model.track(tracker="botsort.yaml")`: ultralytics BoT-SORT **mặc định không có appearance model học** (không fast-reid config) → chỉ tracking motion+IoU, không Re-ID. Nói rõ điều này, đừng để lead đào ra.
- Video demo: dùng sẵn video test trong repo BoxMOT, đừng tải video lạ lúc demo.

**Phần 3 — Fusion:**
- InsightFace `buffalo_l` auto-download ~300MB lúc chạy lần đầu → pre-download vào Drive. Mặt quá nhỏ/quay nghiêng = không detect được → cần nhánh fallback body-only, không được để crash.
- Demo phải dùng ảnh **toàn thân có mặt rõ** (medium shot). Dùng ảnh chân dung thì nhánh OSNet thành noise (chính con số 0.43 của bạn chứng minh).

**Chung:**
- "Colab CLI" thêm 1 lớp điểm lỗi. Mở sẵn notebook thường trong tab khác làm backup.
- Lưu toàn bộ: notebook đã run có output, video output, screenshot, bảng điểm. Colab sập giữa chừng vẫn trình bày được.
- Kiểm tra quota GPU Colab hôm trước, cấm để runtime idle bị disconnect.

**Phần 4:** quyết định không chạy DeepStream là đúng (cần dGPU + GStreamer riêng). Nhưng **mở paper gốc xác minh lại con số 96.5 IDF1 và STCRA** — trích sai số liệu trước dân tech là chết nhất.

## (b) Code có sẵn tốt hơn?

| Hướng | Verdict |
|---|---|
| Appearance Re-ID | Giữ torchreid — chuẩn mực ngành. Backup: dùng luôn OSNet weights của BoxMOT (host GitHub release, tải ổn hơn server torchreid). Model mạnh hơn nếu bị hỏi: `JDAI-CV/fast-reid`. |
| Tracking+ReID | BoxMOT là lựa chọn đúng, không có gì nhanh hơn cho mục đích này. Xem `examples/` trong repo + docs mikelbrostrom.github.io/boxmot. |
| Face | InsightFace giữ. Fallback 1-dòng nếu sập: `serengil/deepface` — `DeepFace.verify(img1, img2, model_name="ArcFace")`. |
| Phần 4 | Giữ slide, không thay. |

Không cần đổi gì — rủi ro nằm ở pin version + pre-download, không nằm ở chọn sai thư viện.

## (c) Fusion 0.6/0.4

Cộng thô 2 cosine từ 2 model khác thang đo = điểm yếu lead sẽ xoáy. Ba lựa chọn, tăng dần độ đúng:

1. **Cách đơn giản mà đúng hơn — cascade:** body (OSNet) làm match chính, face là lớp "xác nhận" khi 2 ảnh đều detect được mặt (face là sinh trắc mạnh hơn nhiều). Không cần trọng số nào.
2. **Score normalization:** min-max/z-score từng nhánh trên tập cặp demo trước rồi mới cộng. Khác biệt: cosine ArcFace same-person ~0.6+, OSNet same-person ~0.5-0.7 — thang khác nhau.
3. **Giữ nguyên weighted-sum** nhưng bắt buộc kèm: bảng ablation 3 cột (body-only / face-only / fused) trên ~8-10 cặp + fallback body-only khi không có mặt + ghi rõ "trọng số heuristic, chọn bằng thử trên tập cặp demo".

Khuyến nghị: cách 3 cho demo (ít sửa code nhất), nhưng mở slide ghi rõ limitation. Nếu còn sức thì cách 1.

## (d) 5 câu lead gần chắc hỏi

1. **"Phần 2 chỉ có Local ID trong 1 video — Re-ID đa camera ở đâu?"** → Phần 1+3 chính là cross-camera: so embedding ảnh từ 2 góc camera khác nhau. Pipeline MTMC đầy đủ (topology + travel-time) là phần 4, không chạy theo quy định chỉ dùng code có sẵn. *(Đây là câu nguy hiểm nhất — tập trước.)*
2. **"Ngưỡng cosine bao nhiêu là cùng người?"** → Không dùng ngưỡng tuyệt đối, dùng top-k retrieval trong gallery. Score phụ thuộc domain, production phải calibrate trên dữ liệu camera thật.
3. **"Vì sao 0.6/0.4?"** → Heuristic + bảng ablation cho thấy fusion thắng khi một nhánh fail; có fallback khi không detect mặt.
4. **"OSNet train trên dữ liệu gì? Domain gap với camera của mình?"** → Market-1501/MSMT17, người đi bộ ngoài trời campus. Có gap với indoor/low-res. Hướng xử lý: fine-tune — ngoài phạm vi demo.
5. **"96.5% IDF1 nghĩa là gì, đo ở đâu?"** → IDF1 = tỉ lệ detection được gán đúng ID (trung bình giữa GT và prediction), trên WILDTRACK 7 camera chồng lấp, điều kiện khác hẳn production — chỉ tham chiếu trần trên của hướng tiếp cận. Biết trả lời công thức IDF1 trước khi bị hỏi.

## (e) Còn 3 giờ — thứ tự

1. **60' — Dry-run toàn bộ trên runtime Colab mới**: pin version, mount Drive có sẵn weights, chạy đúng thứ tự trình bày, đo FPS từng bước. Save output + **quay video màn hình** làm backup cuối cùng.
2. **30' — Chuẩn bị data demo**: 3-5 cặp ảnh toàn thân frontal từ 2 "góc camera", thêm 1-2 cặp negative khó (đồng phục/giống nhau). Crop bbox người. Chất lượng data quyết định demo, không phải code.
3. **30' — Bảng ablation fusion** (body/face/fused trên ~10 cặp) + 1 slide limitations.
4. **20' — Slide phần 4**: mở paper gốc, xác minh lại từng con số + link repo MV3DT trước khi đưa lên slide.
5. **25' — Cheat sheet 5 câu ở (d) + tập nói 10 phút.**
6. **15' — Buffer**: kiểm tra GPU quota, mở sẵn notebook + video backup, tắt màn hình sleep.

Chuẩn bị data và dry-run quan trọng hơn mọi thứ — code đã chạy rồi, thứ giết demo là runtime mới và ảnh xấu.
