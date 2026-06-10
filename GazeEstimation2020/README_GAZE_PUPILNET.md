# Hướng dẫn pipeline nhận diện hướng nhìn mới (PupilNet + best_model)

Tài liệu cho pipeline chính **`machine_a.py` + package `machine_a/`** (Smart Billboard — Máy A)
sau khi thay logic nhận diện hướng ánh mắt cũ (model_x/model_y) bằng phương pháp PupilNet mới.

> File `HUONG_DAN_CHAY.md` cũ nói về demo đơn lẻ `EyeTracking.py` (legacy). File này là tài liệu
> cho pipeline đám đông đang dùng thực tế.

---

## 1. Tổng quan logic mới

Chuỗi xử lý hướng nhìn cho **từng khuôn mặt** trong khung hình:

```
Frame camera
  └─ MediaPipe FaceLandmarker (chạy trên frame thu nhỏ 640px — giữ tốc độ)
       ├─ 468 landmark (tọa độ chuẩn hóa → nhân với kích thước frame GỐC)
       └─ facial_transformation_matrix → góc quay ĐẦU (head yaw/pitch)
  └─ segment_eyes: cắt 2 crop mắt 160×96 từ frame GỐC (full-res, không bị mờ)
  └─ PupilNet_v2 (pupilnet_v5.pt — trọng số train lại) → tâm đồng tử
  └─ 7 điểm/mắt (khóe trong, khóe ngoài, 4 mí, đồng tử) → vector đặc trưng 14 chiều
  └─ best_model.joblib (StandardScaler + MLP, val MAE ≈ 2.22°)
       → đồng thời (yaw, pitch) của MẮT theo ĐỘ
  └─ Góc nhìn thực = góc mắt + góc đầu × GAZE_HEAD_WEIGHT + góc phương vị vị trí mặt
  └─ Làm mượt EMA THEO TỪNG khuôn mặt → so ngưỡng → "Looking" / "Not looking"
```

**Quy ước dấu** (thống nhất toàn pipeline, theo hướng ảnh):
- `yaw > 0` = nhìn về **bên phải ảnh**; `yaw < 0` = bên trái ảnh
- `pitch > 0` = nhìn **xuống dưới**; `pitch < 0` = nhìn lên trên

Model liên quan trong `models/`:

| File | Vai trò |
|---|---|
| `pupilnet_v5.pt` | Trọng số PupilNet_v2 **mới train lại** (lấy từ folder `PupilNet/models/`) |
| `pupilnet_v5_old_backup.pt` | Backup trọng số cũ trước khi thay (có thể xóa khi đã yên tâm) |
| `best_model.joblib` | MLP dự đoán đồng thời (yaw, pitch) theo độ — thay cho model_x/model_y |
| `best_model_meta.json` | Metadata huấn luyện (kiến trúc, MAE, thứ tự 14 cột đặc trưng) |
| `face_landmarker.task` | Model MediaPipe FaceLandmarker |
| `model_x.pkl`, `model_y.pkl` | **Không còn dùng** trong pipeline chính (chỉ còn file legacy tham chiếu) |

---

## 2. Cách chạy

### Chuẩn bị môi trường

```powershell
# Tạo môi trường (lần đầu)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Cần có: `mediapipe`, `torch`, `opencv-python`, `scikit-learn`, `joblib`, `tensorflow` (cho model tuổi/giới tính).

### Chạy pipeline chính (Smart Billboard)

Mở terminal tại thư mục `GazeEstimation2020`:

```powershell
# Chạy với webcam máy tính (test nhanh)
$env:VIDEO_SOURCE = "0"
python machine_a.py

# Chạy thực tế với Raspberry Pi camera qua UDP + Máy B
$env:VIDEO_SOURCE = "udp://0.0.0.0:5000"
$env:MAY_B_IP = "192.168.1.10"
python machine_a.py
```

Nhấn **Q** trong cửa sổ hiển thị để thoát.

Trên khung hình, mỗi khuôn mặt hiển thị:
- Chấm đỏ = tâm đồng tử; đường **xanh lá** = vector hướng nhìn (mắt + đầu)
- `Looking (y=.. p=..)` / `Not looking (y=.. p=..)` — góc yaw/pitch đã làm mượt, **dùng số này để hiệu chỉnh ngưỡng** (mục 3)

---

## 3. Hiệu chỉnh góc cho đúng với camera thực tế

Tất cả tham số chỉnh qua **biến môi trường** (hoặc sửa mặc định trong `machine_a/config.py`,
phần cuối file) — **không cần sửa code logic**.

### 3.1. Ngưỡng vùng "đang nhìn màn hình"

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GAZE_YAW_MIN` / `GAZE_YAW_MAX` | `-20` / `20` | Khoảng góc ngang (độ) tính là đang nhìn |
| `GAZE_PITCH_MIN` / `GAZE_PITCH_MAX` | `-15` / `25` | Khoảng góc dọc (độ); mặc định nới xuống dưới nhiều hơn vì màn hình thường thấp hơn camera |

**Cách hiệu chỉnh thực địa (quan trọng nhất):**
1. Chạy chương trình, đứng ở vị trí người xem điển hình.
2. Lần lượt nhìn vào **4 góc màn hình/biển quảng cáo** và đọc số `(y=.. p=..)` hiển thị trên khung mặt.
3. Lấy min/max của yaw và pitch đọc được, cộng thêm lề ~3–5°, đặt vào 4 biến trên:

```powershell
$env:GAZE_YAW_MIN = "-18"; $env:GAZE_YAW_MAX = "22"
$env:GAZE_PITCH_MIN = "-10"; $env:GAZE_PITCH_MAX = "30"
python machine_a.py
```

### 3.2. Bù góc quay đầu (head pose)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GAZE_HEAD_WEIGHT` | `1.0` | Trọng số cộng góc đầu vào góc nhìn. `0` = tắt (chỉ dùng góc mắt như bản PupilNet demo gốc) |

- Nếu thấy mũi tên xanh lá chỉ **ngược** hướng quay đầu → quy ước dấu head pose của camera/phiên bản
  mediapipe khác giả định → báo lại để lật dấu trong `head_angles_from_matrix()`
  (file `machine_a/vision_utils.py`).
- Nếu kết quả "Looking" nhạy quá mức khi quay đầu nhẹ → giảm còn `0.7`–`0.8`.

### 3.3. Bù góc phương vị (người đứng lệch biên khung hình)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GAZE_BEARING_CORRECTION` | `1` | Bật/tắt bù theo vị trí khuôn mặt trong khung hình |
| `CAMERA_HFOV_DEG` | `60` | **Góc mở ngang thật của camera (độ)** — tra spec camera và đặt đúng |

- Giả định của phép bù: **camera đặt sát/trên biển quảng cáo** (người nhìn biển ≈ nhìn camera).
  Nếu camera đặt xa biển quảng cáo, tắt đi (`GAZE_BEARING_CORRECTION = "0"`) và nới ngưỡng yaw thay thế.
- `CAMERA_HFOV_DEG` sai → người đứng biên bị tính lệch. Ví dụ: Pi Camera Module 2 ≈ 62°,
  Module 3 ≈ 66°, webcam thường 55–70°.

### 3.4. Độ mượt / độ trễ phản hồi

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GAZE_SMOOTH_ALPHA` | `0.3` | Hệ số EMA: tăng (→ 0.5) = phản hồi nhanh nhưng rung hơn; giảm (→ 0.15) = mượt nhưng trễ |
| `GAZE_STATE_TTL_SECONDS` | `1.0` | Mặt biến mất quá lâu thì xóa trạng thái làm mượt của mặt đó |

---

## 4. Những gì đã sửa / đã làm (changelog)

### Đợt 1 — Thay logic hướng nhìn sang phương pháp PupilNet

1. **Copy model mới** vào `models/`: `pupilnet_v5.pt` (train lại, bản cũ backup), `best_model.joblib`, `best_model_meta.json`.
2. **`machine_a/config.py`** — `GAZE_MODEL_PATH` (best_model.joblib) thay cho `GAZE_MODEL_X_PATH`/`GAZE_MODEL_Y_PATH`; thêm các ngưỡng góc/làm mượt cấu hình bằng env.
3. **`machine_a/vision_utils.py`** — port từ `PupilNet/EyeTracking.py`: ánh xạ `MP_RIGHT`/`MP_LEFT`, `eye_points_7()`, `build_feature_14d()` (chuẩn hóa + tự lật về canonical), `predict_gaze_deg()`.
4. **`machine_a/models.py`** — load `best_model.joblib`; chữ ký `load_models()` đổi thành `(device, gaze_model, pupil_model, age_gender_model, face_landmarker)`.
5. **`machine_a/pipeline.py`** — thay khối dự đoán điểm-nhìn (model_x/model_y, đơn vị chuẩn hóa) bằng (yaw, pitch) theo độ; vẽ vector hướng nhìn; hiển thị góc cạnh trạng thái Looking; thêm `gaze_yaw`/`gaze_pitch` vào detection payload.
6. **`machine_a.py`** — cập nhật lời gọi theo chữ ký mới.

7. **FIX LỖI đám đông — làm mượt dùng chung:** code cũ chỉ có một cặp `smooth_x/smooth_y`
   **toàn cục cho mọi người** → góc nhìn của nhiều người bị trộn lẫn (người nhìn + người không nhìn
   → cả hai ra trung bình sai). Đã viết `smooth_gaze_for_face()` trong `pipeline.py`: mỗi khuôn mặt
   một trạng thái EMA riêng, ghép giữa các frame theo tâm bbox gần nhất, tự xóa sau TTL.

### Đợt 2 — Tăng độ chính xác (mục 1 & 2 danh sách điểm yếu)

8. **FIX mắt nhỏ/đứng xa (mục 1):** trước đây crop mắt lấy từ frame đã thu nhỏ 640px —
   mắt người đứng xa chỉ còn vài pixel, PupilNet đoán tâm đồng tử rất kém. Giờ MediaPipe vẫn
   detect trên frame nhỏ (giữ FPS) nhưng landmark chuẩn hóa được nhân với kích thước frame
   **gốc**, crop mắt cắt từ frame gốc → sắc nét. Đồng thời bỏ toàn bộ logic scale tọa độ
   (`scale_x`/`scale_y`) và một lần copy frame thừa trong `segment_eyes`.

9. **FIX góc mắt-trong-đầu vs góc nhìn-tới-camera (mục 2):** best_model train trên UnityEyes
   nên chỉ cho góc mắt **so với đầu** — người quay đầu về màn hình nhưng mắt thẳng vẫn bị
   tính "không nhìn". Đã:
   - Bật `output_facial_transformation_matrixes` của FaceLandmarker (`models.py`);
   - Thêm `head_angles_from_matrix()` — góc quay đầu theo quy ước dấu khớp với góc mắt;
   - Thêm `face_bearing_deg()` — góc phương vị của mặt trong khung hình (người đứng lệch biên
     nhìn về camera/màn hình được quy về ~0°);
   - Công thức: `góc nhìn = mắt + GAZE_HEAD_WEIGHT × đầu + phương_vị` (từng thành phần tắt được).

### Đợt 3 — Tự chứa trong repo (chạy được ngay sau khi clone từ git)

10. **Thay `EyeTracking.py` bằng bản mới** từ `PupilNet/EyeTracking.py` (bản cũ trong repo dùng
    model_x/model_y; bản mới dùng pupilnet_v5 + best_model, mọi đường dẫn đều tương đối trong repo).
    Demo đơn lẻ này chạy webcam: `python EyeTracking.py`.
11. **Copy `calibrate_mediapipe.py`** từ PupilNet — tool hiệu chỉnh/kiểm tra ánh xạ chỉ số landmark
    MediaPipe cho 7 điểm mắt (chạy webcam hoặc ảnh tĩnh, tự chứa). KHÔNG copy `check_feature.py`
    vì nó cần dataset UnityEyes nằm ngoài repo.
12. **FIX đường dẫn model tuổi/giới tính trỏ ra ngoài repo:** `AGE_GENDER_MODEL_PATH` mặc định cũ
    trỏ `../TrainModelAgeAndGender/age_gender_range_efficientnetv2s_v1.keras` (NGOÀI repo — sau khi
    push lên git sẽ không tồn tại). Đổi mặc định thành `GazeEstimation2020/models/age_gender_range_efficientnetv2s_v1.keras`.

> ⚠️ **VIỆC CẦN LÀM TRƯỚC KHI PUSH:** file `.keras` của model tuổi/giới tính hiện
> **không có trong repo** (và không tìm thấy trên máy này). Hãy copy
> `age_gender_range_efficientnetv2s_v1.keras` vào `GazeEstimation2020/models/`
> (hoặc đặt biến môi trường `AGE_GENDER_MODEL_PATH` khi chạy). Thiếu file này
> `machine_a.py` sẽ báo lỗi ngay lúc `load_models()`.

### Kiểm thử đã chạy

- Load model thật: `best_model.joblib` (Pipeline) + `pupilnet_v5.pt` mới với kiến trúc `PupilNet_v2` → output đúng shape.
- Chuỗi đặc trưng: 7 điểm → vector (1, 14), logic lật canonical/mirror đúng.
- Làm mượt theo từng mặt: 2 mặt cùng frame giữ 2 trạng thái riêng, EMA blend đúng.
- Hình học: ma trận quay đầu identity/quanh-Y/quanh-X ra đúng (0°, ±30°); phương vị tâm/biên khung hình
  ra đúng (0°, ±HFOV/2); kịch bản "đứng lệch trái + quay đầu nhìn màn hình" → tổng góc = 0° (đang nhìn) ✓.
- `py_compile` sạch toàn bộ file đã sửa.

> Lưu ý: môi trường kiểm thử không có `mediapipe` nên phần FaceLandmarker thật chưa chạy được ở đây —
> cần chạy `python machine_a.py` trong môi trường thật để xác nhận end-to-end (đặc biệt là **dấu** của
> head yaw/pitch — xem mục 3.2).

---

## 5. Điểm yếu còn lại (chưa xử lý — theo thứ tự nên làm)

1. **Ngưỡng góc cần hiệu chỉnh thực địa** — làm theo mục 3.1 (giờ rất dễ vì góc hiện ngay trên khung hình).
2. **Giới hạn 5 khuôn mặt** — `num_faces=5` trong `machine_a/models.py`; tăng nếu đám đông đông hơn (đổi FPS).
3. **Ghép mặt theo tâm bbox có thể nhầm khi 2 người đi cắt ngang nhau** — hiếm, chỉ sai trong ~1s EMA;
   giải pháp triệt để là gắn làm mượt vào `ViewerTrackManager` (IoU tracking).
4. **File legacy chưa cập nhật** (vẫn dùng model_x/model_y, không ảnh hưởng pipeline chính):
   `machine_a_copy.py` và `machine_a.py` ở thư mục gốc repo (file gốc repo trỏ đến thư mục
   `models/` không tồn tại — code chết, chạy là crash). Nên xóa nếu không còn dùng để tránh nhầm lẫn.
   (`EyeTracking.py` đã được thay bằng bản mới ở Đợt 3.)
