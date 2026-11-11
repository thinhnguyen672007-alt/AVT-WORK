# Demo Re-ID có sẵn để chạy ngay — cho buổi trình bày 12/11/2026

Nguyên tắc của anh Tiến: **dùng code có sẵn** (Kaggle/Colab/GitHub), không tự viết hệ thống. Tài liệu này liệt kê, cho từng hướng giải pháp, 1–2 nguồn **đã kiểm chứng link còn sống** (đã mở trực tiếp ngày 11/11/2026), kèm: demo cho thấy gì, cần gì để chạy, độ khó thực tế cho một buổi chiều.

> Ghi chú trung thực về Kaggle: đã tìm kỹ, **không có notebook Kaggle về Person Re-ID dạng "mở là chạy" đủ tin cậy** (kết quả Kaggle cho Re-ID rất loãng, phần lớn là dataset/notebook train rời rạc). Đường nhanh và chắc nhất là **Google Colab** (GPU miễn phí). Nếu muốn dùng Kaggle: tạo notebook mới → bật GPU → chạy y nguyên các cell của BoxMOT/torchreid bên dưới, Kaggle chỉ đóng vai trò máy chạy thay Colab.

---

## Hướng 1 — Appearance-based Re-ID (OSNet / FastReID / TransReID qua torchreid)

### 1.1 ⭐ Torchreid — thư viện chính thức của OSNet (Kaiyang Zhou)
- **Link:** https://github.com/KaiyangZhou/deep-person-reid (đã mở kiểm chứng: repo sống, 4.9k stars, có sẵn API `torchreid/utils/feature_extractor.py`)
- **Demo cho thấy gì:** Dùng class `FeatureExtractor` (~15–20 dòng): đưa vào một danh sách ảnh crop người (query + gallery — có thể dùng **ảnh chính team tự chụp** bằng điện thoại, cắt lấy khung người) → trích embedding cho từng ảnh → tính ma trận cosine similarity → in/cuộn ra cặp ảnh nào là cùng một người, cặp nào khác người. Đây đúng nguyên lý "Giải pháp 1" trong file PDF: YOLO crop → embedding → cosine.
- **Cần gì để chạy:** Colab GPU (hoặc CPU cũng được, OSNet rất nhẹ). Cài: `git clone` repo + `pip install -r requirements.txt` + `python setup.py develop` (đừng chỉ `pip install torchreid` — bản PyPI dễ lỗi version; clone repo chắc hơn). Weights OSNet pretrained **tự tải** từ model zoo (Google Drive) hoặc HuggingFace `kaiyangzhou/osnet`. Không cần tải dataset lớn nào nếu dùng ảnh tự chụp; muốn số đẹp hơn thì lấy vài ảnh mẫu Market-1501.
- **Độ khó:** **Dễ–Trung bình** (~30–45 phút kể cả xử lý lỗi cài đặt). Rủi ro chính: khớp version torch trên Colab.

### 1.2 FastReID — toolbox chính thức (JD AI Research)
- **Link:** https://github.com/JDAI-CV/fast-reid (đã kiểm chứng: có sẵn thư mục `demo/` và `MODEL_ZOO.md`)
- **Demo cho thấy gì:** Script chính thức `demo/demo.py`: đưa ảnh query + thư mục gallery → xuất **ảnh trực quan xếp hạng** (query bên trái, các ảnh giống nhất xếp theo điểm giảm dần, khung xanh = đúng người, đỏ = sai). Hình ảnh này rất hợp để chiếu khi trình bày.
- **Cần gì để chạy:** Colab GPU; cài theo `INSTALL.md` (~10–15 phút); tải 1 config + 1 weights từ Model Zoo (ví dụ họ bagtricks ResNet50 train trên Market-1501, link sẵn trong `MODEL_ZOO.md`); cần một ít ảnh gallery — lấy subset nhỏ của Market-1501 hoặc ảnh tự chụp.
- **Độ khó:** **Trung bình** (cài nặng hơn torchreid một chút, bù lại output đẹp nhất trong nhóm Hướng 1).

> TransReID: trong thời gian khảo sát không kiểm chứng được notebook/repo demo "chạy ngay" đáng tin — ngày mai nên đại diện nhánh này bằng OSNet (torchreid) hoặc FastReID như trên.

---

## Hướng 2 — Tracking + Re-ID đa camera (Local ID → Global ID)

### 2.1 ⭐ BoxMOT — Colab chính thức (BoT-SORT / ByteTrack / DeepSORT + OSNet Re-ID trong 1 notebook)
- **Link Colab:** https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing (link lấy từ badge Colab trong README chính thức của BoxMOT, đã mở kiểm chứng còn sống)
- **Repo:** PyPI/README — `pip install boxmot`; lệnh mẫu: `boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4 --save`
- **Demo cho thấy gì:** Video người đi bộ (dùng video mẫu của notebook hoặc video team tự quay) → video kết quả có bounding box + **ID bám theo từng người**, bật Re-ID (OSNet) thì ID giữ ổn định hơn khi người bị che khuất/đi khuất rồi quay lại. Đổi tracker chỉ bằng 1 tham số: `bytetrack` / `botsort` / `deepocsort` / `strongsort` — trình bày được cả 3 tracker nêu trong PDF trong cùng một demo.
- **Cần gì để chạy:** Colab GPU; weights YOLO (yolov8n) và OSNet **tự tải** khi chạy lệnh. Không cần dataset.
- **Độ khó:** **Dễ** — đây là đường nhanh nhất cho toàn bộ Hướng 2 ở mức 1 camera (Local ID).
- **Giới hạn phải nói rõ khi trình bày:** BoxMOT cho Local ID trong 1 camera. Lớp gộp **Global ID** xuyên camera là phần các hệ thi đấu (2.2) làm thêm bằng clustering embedding + ràng buộc không–thời gian.

### 2.2 Nota Inc. — AIC2024 Track 1: hệ multi-camera hoàn chỉnh (YOLO + BoT-SORT + FastReID)
- **Link tài liệu/repo:** https://deepwiki.com/nota-github/AIC2024_Track1_Nota (trang DeepWiki của repo GitHub `nota-github/AIC2024_Track1_Nota`, đã mở kiểm chứng nội dung)
- **Demo cho thấy gì:** Nhiều video từ các camera **đồng bộ** → detect YOLO → tracking BoT-SORT từng camera → FastReID trích embedding → **clustering gán Global ID xuyên camera** (+ mmpose và biến đổi phối cảnh) → xuất file tracking chuẩn MOT + video đã gán nhãn.
- **Cần gì để chạy:** Máy GPU + Docker (repo đóng gói Docker, có `run_mcpt.sh`, thư mục `pretrained/`); dữ liệu AI City Challenge phải xin qua form của BTC — **không tải ngay được**.
- **Độ khó:** **Khó** cho hạn ngày mai. Vai trò thực tế: tài liệu kiến trúc để vẽ sơ đồ Global ID và trả lời câu hỏi "gộp ID kiểu gì", không phải demo chạy tại chỗ.

---

## Hướng 3 — Multi-Modal / Fusion Re-ID (appearance + face + pose/gait + thời gian/không gian)

> Kết luận khảo sát: **không tồn tại notebook Colab/Kaggle "mở là chạy" cho fusion Re-ID.** Các cài đặt fusion thật đều là code nghiên cứu/thi đấu, độ khó cao. Dưới đây là 2 nguồn thật, kèm phương án demo thay thế khả thi.

### 3.1 UW-IPL — Quán quân AI City Challenge 2023 Track 1 (fusion đầy đủ nhất, đúng nghĩa đen của Hướng 3)
- **Link:** https://github.com/ipl-uw/AIC23_Track1_UWIPL_ETRI (đã mở kiểm chứng toàn bộ README)
- **Fusion gồm những gì:** appearance = OSNet (torchreid) + **pose** = MMPose trích keypoint bàn chân → chiếu 3D vị trí + **không–thời gian** = module STCRA (Spatio-Temporal Consistency Re-Assignment) dùng *camera link model* (topology + thời gian di chuyển giữa các camera) để gán lại Global ID.
- **Demo cho thấy gì:** Pipeline đa camera hoàn chỉnh: detection có sẵn trong `data/test_det` (đường "fast reproduce" bỏ qua bước detect), embedding tải dạng `.npy` từ Google Drive → chạy tracking + STCRA → ra kết quả tracking đa camera với ID nhất quán (IDF1 95.36%, hạng 1 AIC23).
- **Cần gì để chạy:** **4 môi trường conda riêng** (mmyolo, mmpose, torchreid, BoT-SORT); dataset AIC23 phải điền form xin mật khẩu tải; weights qua Google Drive.
- **Độ khó:** **Khó** — gần như không thể chạy trọn vẹn trong một buổi chiều nếu chưa chuẩn bị từ trước. Dùng làm **tài liệu kiến trúc + sơ đồ** cho Hướng 3 (và cả 4d bên dưới).

### 3.2 OpenGait — mảnh ghép "dáng đi" (gait) của fusion
- **Link:** https://github.com/ShiqiYu/OpenGait (link công bố chính thức trong bài báo MultiGait++/OpenGait)
- **Demo cho thấy gì:** Nhận dạng người chỉ bằng dáng đi (chuỗi silhouette) trên các bộ CASIA-B/Gait3D/GREW — minh họa được modality "gait" khi thuyết trình.
- **Cần gì:** Tải dataset gait + tiền xử lý silhouette + pretrained theo config của repo; không có Colab turnkey.
- **Độ khó:** **Khó** — chỉ nên nhắc tên khi trình bày, không khuyến nghị chạy ngày mai.

### 3.3 Phương án demo thay thế cho Hướng 3 (khả thi trong vài giờ) — "fusion ở mức điểm số"
Ghép 2 mảnh **đã có sẵn và đã kiểm chứng ở trên**: với cùng một cặp ảnh người, tính (a) điểm cosine của **OSNet** (Hướng 1.1 — tín hiệu ngoại hình) và (b) điểm của **model mặt InsightFace buffalo_l** (chính là code FaceID team đã làm — tín hiệu khuôn mặt), rồi cộng có trọng số thành điểm chung và ra quyết định cùng/khác người. Đây là score-level fusion đúng như mô tả trong PDF (appearance 85% + face/gait… → xác suất chung), trình bày được nguyên lý fusion mà không cần cài hệ thống nghiên cứu. **Độ khó: Dễ–Trung bình** (chỉ cộng thêm ~10 dòng vào notebook của Hướng 1.1).

---

## Hướng 4 — Bộ 4 options tracking của team

### 4a. YOLO + BoT-SORT
- **Nguồn:** Ultralytics YOLO track mode — tài liệu chính thức https://docs.ultralytics.com/modes/track (đã kiểm chứng: `tracker="botsort.yaml"`, có sẵn tùy chọn `with_reid` trong file cấu hình)
- **Demo cho thấy gì:** 3 dòng code trong Colab: `model = YOLO("yolo11n.pt")` (hoặc yolov8n) → `model.track(source=<video>, tracker="botsort.yaml")` → video kết quả: mỗi người một ID ổn định, có bù chuyển động camera (camera-motion compensation) — đúng đặc trưng BoT-SORT.
- **Cần gì:** Colab GPU; weights YOLO **tự tải**; video bất kỳ (video mẫu của Ultralytics hoặc tự quay).
- **Độ khó:** **Dễ — nhanh nhất trong toàn bộ danh sách này** (~10–15 phút).
- **Dự phòng cho ByteTrack thuần (nếu muốn đối chiếu):** notebook chính thức của Roboflow Supervision — https://nbviewer.org/github/roboflow/supervision/blob/develop/docs/notebooks/object-tracking.ipynb (đã mở kiểm chứng toàn bộ nội dung): YOLOv8n + `sv.ByteTrack`, video mẫu `PEOPLE_WALKING` **tự tải trong notebook**, xuất `output.mp4` có tracker_id. Độ khó: **Dễ**.

### 4b. YOLO + BoT-SORT + Person Re-ID + Global ID
- **Nguồn:** chính là **BoxMOT (mục 2.1)** — `--tracker botsort --reid osnet_x0_25_msmt17` là đúng tổ hợp YOLO + BoT-SORT + Re-ID, chạy trên Colab, weights tự tải. **Độ khó: Dễ.**
- **Phần Global ID:** thể hiện ở mức 1 camera (Local ID + embedding). Muốn nói đúng tầng Global ID xuyên camera thì chiếu kèm kiến trúc của **Nota AIC2024 (2.2)** hoặc **UW AIC23 (3.1)** — hai hệ này đều gộp ID bằng clustering trên embedding + ràng buộc không–thời gian.

### 4c. DeepStream + MV3DT
- **Nguồn:** NVIDIA DeepStream (monorepo chính thức) — https://github.com/NVIDIA/DeepStream (đã mở kiểm chứng: bản hiện tại DeepStream 9.2, trong thư mục `skills/` có sẵn **`deepstream-run-mv3dt`** — skill chạy reference app MV3DT trên dataset mẫu hoặc dữ liệu tự có; kèm công cụ `auto-magic-calib` để tự hiệu chuẩn camera). Bài giới thiệu chính thức: https://developer.nvidia.com/blog/?p=119991 — MV3DT chiếu detection từ nhiều camera đã hiệu chuẩn vào **hệ tọa độ 3D chung**, gán **một Global ID duy nhất** xuyên camera; repo kèm **dataset mẫu 4 camera và 12 camera đã có sẵn calibration**.
- **Cần gì để chạy:** Máy **Ubuntu 24.04 + GPU NVIDIA thật** (yêu cầu driver 595+, CUDA 13.2) + Docker (`docker pull ghcr.io/nvidia/deepstream:9.2-triton-multiarch`) hoặc build từ source. **Không chạy được trên Colab/Kaggle** (không có Docker-in-Docker, driver không khớp).
- **Độ khó:** **Khó** — chỉ khả thi nếu công ty có sẵn máy NVIDIA/Ubuntu đã cài DeepStream. Nếu không: trình bày Hướng 4c bằng sơ đồ + bài báo MV3DT (arXiv "Fully Distributed Multi-View 3D Tracking in Real-Time": https://arxiv.org/pdf/2606.13127) và nói rõ đây là hướng triển khai production của NVIDIA, demo phần mềm để giai đoạn sau.

### 4d. Re-ID + thời gian di chuyển + sơ đồ (topology) camera
- **Nguồn:** module **STCRA trong repo UW-IPL AIC23 (mục 3.1)** chính là cài đặt đúng nghĩa của option này: sau khi có tracklet + embedding OSNet, STCRA dùng **camera link model** (camera nào nối với camera nào, thời gian di chuyển hợp lệ trong khoảng nào) + tính nhất quán không–thời gian để **gán lại Global ID**, sửa các lỗi ID mà appearance đơn thuần gây ra.
- **Cần gì / độ khó:** như mục 3.1 — **Khó**, dataset phải xin qua form. Không có bản Colab turnkey cho option này (đã tìm, không có).
- **Cách trình bày khả thi ngày mai:** chạy BoxMOT (4b) để có tracklet + Local ID thật, sau đó trình bày lớp 4d bằng sơ đồ camera-link của UW AIC23 và giải thích: "cặp tracklet ở 2 camera chỉ được gộp khi (1) embedding đủ giống VÀ (2) thời gian di chuyển nằm trong cửa sổ hợp lệ theo topology" — đúng công thức Re-ID + travel time + topology.

---

## 🎯 KHUYẾN NGHỊ — đường nhanh nhất tới demo trình bày được ngày mai (mỗi hướng 1 nguồn)

| Hướng | Chạy cái này | Nền tảng | Thời gian ước tính |
|---|---|---|---|
| 1. Appearance Re-ID | **Torchreid `FeatureExtractor` + OSNet** (1.1), dùng ảnh team tự chụp làm query/gallery | Colab GPU | ~30–45 phút |
| 2. Tracking + Re-ID | **BoxMOT Colab chính thức** (2.1), `botsort` + `osnet_x0_25_msmt17` | Colab GPU | ~20–30 phút |
| 3. Multi-Modal/Fusion | **Fusion mức điểm số** (3.3): OSNet + InsightFace buffalo_l của team, cộng điểm có trọng số; kiến trúc đầy đủ lấy từ UW AIC23 (3.1) để thuyết trình | Colab GPU | ~1 giờ (sau khi Hướng 1 chạy xong) |
| 4a. YOLO + BoT-SORT | **Ultralytics `model.track(tracker="botsort.yaml")`** (4a) | Colab GPU | ~10–15 phút |
| 4b. YOLO + BoT-SORT + Re-ID + Global ID | **BoxMOT** như Hướng 2 (cùng một notebook, khỏi chuẩn bị hai lần) | Colab GPU | dùng chung Hướng 2 |
| 4c. DeepStream + MV3DT | Chỉ chạy nếu có máy NVIDIA/Ubuntu sẵn: repo **NVIDIA/DeepStream**, skill `deepstream-run-mv3dt` + dataset mẫu 4 camera; nếu không có máy → trình bày bằng blog/paper của NVIDIA | Máy NVIDIA local | Khó — không hứa demo |
| 4d. Re-ID + travel time + topology | Không có bản chạy kịp: trình bày qua **STCRA của UW AIC23** (3.1) + minh họa trên kết quả BoxMOT của 4b | — (thuyết trình) | — |

**Thứ tự làm việc gợi ý cho chiều nay:** 4a (15 phút, chắc chắn có demo) → 2/4b BoxMOT (30 phút) → 1 Torchreid (45 phút) → 3.3 fusion mini (1 giờ). Tổng ~2.5–3 giờ là có 4 demo chạy thật; 4c và 4d trình bày bằng kiến trúc của NVIDIA DeepStream/MV3DT và UW-IPL AIC23 — hai hệ thật, có số liệu thật (MV3DT: 96.5% IDF1 trên WILDTRACK; UW: 95.36% IDF1, hạng 1 AI City 2023), không cần bịa demo.
