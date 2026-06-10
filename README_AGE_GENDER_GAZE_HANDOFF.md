# Machine A - Luồng nhận diện tuổi, giới tính và gaze estimation

Tài liệu này dành cho người cần đọc, thay đổi hoặc thay thế mô hình xác định người xem có nhìn màn hình hay không (`gaze estimation`) mà vẫn giữ nguyên luồng nhận diện tuổi/giới tính và giao tiếp với Máy B.

## 1. Mục tiêu của Machine A

Machine A nhận video camera từ Raspberry Pi và xử lý từng khuôn mặt để tạo một detection có dạng:

```python
{
    "bbox": (x1, y1, x2, y2),
    "audience_segment_id": 9,
    "looking": True,
    "gender_label": "Female",
    "age_range": "18-25",
    "age_range_confidence": 0.82,
}
```

Trong đó:

- `bbox`: vị trí khuôn mặt trên frame gốc.
- `gender_label`, `age_range`, `age_range_confidence`: kết quả mô hình tuổi/giới tính.
- `audience_segment_id`: ID kết hợp giới tính và nhóm tuổi, dùng để chọn quảng cáo.
- `looking`: kết quả gaze estimation, dùng để tính thời gian người đó nhìn quảng cáo.

Hai nhánh tuổi/giới tính và gaze sử dụng chung khuôn mặt được MediaPipe phát hiện, nhưng suy luận độc lập với nhau.

## 2. Các file quan trọng

| File | Vai trò |
|---|---|
| `GazeEstimation2020/machine_a.py` | Vòng lặp chính, quản lý trạng thái lấy mẫu và phát quảng cáo |
| `GazeEstimation2020/machine_a/pipeline.py` | Phát hiện khuôn mặt, gọi hai nhánh age/gender và gaze, tạo detection |
| `GazeEstimation2020/machine_a/demographics.py` | Worker nền và cache kết quả tuổi/giới tính |
| `GazeEstimation2020/machine_a/vision_utils.py` | Tiền xử lý khuôn mặt, giải mã output tuổi/giới tính, xử lý mắt |
| `GazeEstimation2020/machine_a/models.py` | Load MediaPipe, PupilNet, gaze SVR và model Keras tuổi/giới tính |
| `GazeEstimation2020/machine_a/tracking.py` | Ghép detection thành từng người và tính thời gian nhìn |
| `GazeEstimation2020/machine_a/config.py` | Đường dẫn model, nhóm tuổi, segment ID và tham số hiệu năng |

## 3. Luồng xử lý tổng quát

```text
Raspberry Pi camera
        |
        v
LatestFrameGrabber lấy frame mới nhất
        |
        v
MediaPipe FaceLandmarker phát hiện tối đa 5 khuôn mặt
        |
        +-----------------------------+
        |                             |
        v                             v
Nhánh tuổi/giới tính             Nhánh gaze estimation
chạy bằng worker nền             chạy trong pipeline chính
        |                             |
        v                             v
gender + age range + segment ID   looking = True/False
        |                             |
        +-------------+---------------+
                      v
              Detection của mỗi mặt
                      |
                      v
             ViewerTrackManager
                      |
          +-----------+-----------+
          |                       |
          v                       v
Chọn quảng cáo              Report sau khi phát
```

Hàm trung tâm của luồng này là:

```python
collect_viewer_detections(...)
```

trong `GazeEstimation2020/machine_a/pipeline.py`.

## 4. Phát hiện khuôn mặt

Model MediaPipe được load trong `machine_a/models.py`:

```python
vision.FaceLandmarkerOptions(
    num_faces=5,
    running_mode=vision.RunningMode.IMAGE,
)
```

Mỗi frame được xử lý như sau:

1. Nếu frame gốc rộng hơn `VISION_PROCESS_WIDTH`, frame được resize nhỏ lại.
2. MediaPipe phát hiện landmark trên frame đã resize.
3. Tọa độ khuôn mặt được chuyển ngược về kích thước frame gốc.
4. `build_face_bbox()` tạo bounding box với padding 30%.
5. Face crop từ frame gốc được gửi cho model tuổi/giới tính.

Việc xử lý landmark trên ảnh nhỏ giúp giảm tải CPU, nhưng crop tuổi/giới tính vẫn lấy từ frame gốc để giữ độ chi tiết.

## 5. Luồng nhận diện tuổi và giới tính

### 5.1 Model và input

Model được load bằng:

```python
tf.keras.models.load_model(AGE_GENDER_MODEL_PATH, compile=False)
```

Tiền xử lý trong `preprocess_face_for_inference()`:

1. Nhận face crop dạng OpenCV BGR.
2. Chuyển BGR sang RGB.
3. Thêm vùng đen để ảnh thành hình vuông, không kéo méo khuôn mặt.
4. Resize về `256x256`.
5. Chuyển sang `float32`.
6. Chuẩn hóa pixel về khoảng `0..1`.
7. Thêm batch dimension, tạo shape `(1, 256, 256, 3)`.

### 5.2 Output bắt buộc của model

Hàm `predict_age_gender()` yêu cầu model trả về dictionary có đúng hai output:

```python
{
    "gender_output": ...,
    "age_distribution_output": ...,
}
```

Ý nghĩa:

- `gender_output`: xác suất giới tính. Xác suất từ `0.5` trở lên là `Female`, thấp hơn là `Male`.
- `age_distribution_output`: phân phối xác suất theo từng tuổi từ 0 đến 116.

Nếu model mới đổi tên output, kiểu output hoặc cách chuẩn hóa input thì phải sửa đồng bộ `predict_age_gender()` và `preprocess_face_for_inference()`.

### 5.3 Chuyển tuổi thành nhóm tuổi

Phân phối tuổi được cộng lại thành bảy nhóm:

```text
0-17
18-25
26-35
36-45
46-54
55-65
66+
```

Nhóm có tổng xác suất cao nhất được chọn làm `age_range`. Tổng xác suất của nhóm đó trở thành `age_range_confidence`.

### 5.4 Chuyển thành audience segment

`audience_segment_id` kết hợp giới tính và nhóm tuổi:

| Giới tính | Nhóm tuổi | Segment ID |
|---|---:|---:|
| Male | 0-17 đến 66+ | 1 đến 7 |
| Female | 0-17 đến 66+ | 8 đến 14 |

Các ID này phải khớp bảng `audience_segments` trên Máy B.

### 5.5 Worker nền và cache

Tuổi/giới tính không được dự đoán trực tiếp trong vòng lặp giao diện. `DemographicPredictor` chạy model trong một thread nền để tránh làm camera và quảng cáo bị đứng.

Luồng cache:

1. Pipeline tìm kết quả cũ theo độ giao nhau bounding box (`bbox IoU`).
2. Nếu có cache hợp lệ, kết quả cũ được dùng ngay.
3. Theo chu kỳ `AGE_GENDER_EVERY_N_FRAMES`, face crop được đưa vào hàng đợi dự đoán.
4. Worker chạy model và cập nhật cache.
5. Cache hết hạn sau `DEMOGRAPHIC_CACHE_TTL_SECONDS`.

Các biến mặc định:

```text
DEMOGRAPHIC_REFRESH_SECONDS = 1.0
DEMOGRAPHIC_CACHE_TTL_SECONDS = 3.0
```

Do chạy bất đồng bộ, một khuôn mặt mới có thể xuất hiện vài frame đầu với `audience_segment_id = None` trước khi kết quả tuổi/giới tính sẵn sàng.

## 6. Luồng gaze estimation hiện tại

Nhánh gaze hiện tại thực hiện:

1. Chuyển landmark MediaPipe của mắt sang dạng tương tự landmark dlib.
2. Cắt và chuẩn hóa ảnh mắt trái, mắt phải.
3. Dùng `PupilNet_v2` để dự đoán tâm đồng tử.
4. Dùng hai model SVR `model_x.pkl` và `model_y.pkl` để dự đoán hướng nhìn.
5. Lấy trung bình kết quả hai mắt.
6. Làm mượt bằng exponential smoothing:

```python
smooth = 0.2 * current_prediction + 0.8 * previous_smooth
```

7. Xác định đang nhìn màn hình nếu hướng nhìn nằm trong vùng:

```text
-0.5 < gaze_x < 0.5
-0.15 < gaze_y < 0.8
```

Kết quả cuối cùng phải là boolean:

```python
is_looking_at_screen = True  # hoặc False
```

## 7. Hợp đồng khi thay thế mô hình nhìn/không nhìn

Bạn có thể thay toàn bộ phần xử lý gaze bên trong `collect_viewer_detections()`, nhưng cần giữ các điều kiện sau:

1. Mỗi khuôn mặt phải có một kết quả `looking` kiểu boolean.
2. Không thay đổi `bbox` của detection sang hệ tọa độ khác. `bbox` phải theo kích thước frame gốc.
3. Không xóa các trường `audience_segment_id`, `gender_label`, `age_range` và `age_range_confidence`.
4. Khi không thể dự đoán gaze, nên trả về `False`, không làm crash toàn bộ vòng lặp.
5. Phần gaze vẫn phải chạy trong cả trạng thái lấy mẫu và trạng thái đang phát quảng cáo.
6. Không gọi API trực tiếp từ model gaze. Pipeline và tracker chịu trách nhiệm tổng hợp rồi gửi API.

Interface tối thiểu được khuyến nghị cho model mới:

```python
def predict_is_looking(frame, face_bbox, face_landmarks, state=None) -> bool:
    ...
```

Sau đó gán kết quả vào detection:

```python
detection = {
    "bbox": (x1, y1, x2, y2),
    "audience_segment_id": audience_segment_id,
    "looking": bool(is_looking_at_screen),
    "gender_label": gender_label,
    "age_range": age_range,
    "age_range_confidence": age_range_confidence,
}
```

## 8. Tracker sử dụng kết quả như thế nào

`ViewerTrackManager` ghép các detection qua nhiều frame bằng `bbox IoU`.

Với mỗi người:

- `audience_segment_id` được lấy theo majority vote từ nhiều lần dự đoán.
- `looking=True` làm tăng `watch_duration`.
- `looking=False` vẫn giữ người đó trong track nhưng không tăng thời gian nhìn.

Vì vậy mô hình gaze mới không cần tự tính tổng thời gian. Nó chỉ cần trả lời đúng câu hỏi cho từng detection tại thời điểm hiện tại:

```text
Người này hiện có đang nhìn màn hình hay không?
```

## 9. Dữ liệu gửi sang Máy B

### 9.1 Chọn quảng cáo

Sau cửa sổ lấy mẫu mặc định 4 giây, Machine A gửi:

```http
POST /api/advertisements/select
```

Payload quan trọng:

```json
{
  "timestamp": "2026-06-10T10:00:00",
  "viewer_count": 2,
  "audience_segment_id": 9
}
```

Lưu ý:

- `viewer_count` hiện chỉ đếm những người đã có `audience_segment_id`.
- Nếu `viewer_count > 0`, Máy B yêu cầu `audience_segment_id` không được `null`.
- Nếu chưa phân loại được người nào, Machine A gửi `viewer_count = 0` để Máy B chọn quảng cáo fallback.
- Kết quả `looking` không tham gia trực tiếp vào API chọn quảng cáo.

### 9.2 Report sau khi phát quảng cáo

Sau khi quảng cáo kết thúc, Machine A gửi:

```http
POST /api/ad-play-logs/report
```

Payload chính:

```json
{
  "ad_id": 12,
  "start_time": "2026-06-10T10:00:05",
  "end_time": "2026-06-10T10:00:35",
  "total_viewers": 2,
  "viewers": [
    {
      "audience_segment_id": 9,
      "watch_duration": 18.4
    }
  ]
}
```

`watch_duration` phụ thuộc trực tiếp vào độ ổn định của kết quả `looking`.

## 10. Điểm cần chú ý trước khi sửa

### Trạng thái gaze hiện đang dùng chung

Hiện tại `gaze_state` chỉ có một bộ:

```python
{"smooth_x": 0.0, "smooth_y": 0.0}
```

Bộ trạng thái này được dùng chung cho tất cả khuôn mặt. Khi có nhiều người, kết quả làm mượt của người trước có thể ảnh hưởng người sau. Nếu sửa gaze, nên lưu state riêng theo từng track/người.

### Tracking chỉ dựa trên bbox IoU

Khi người di chuyển nhanh, che khuất nhau hoặc đổi vị trí, tracker có thể tạo track mới hoặc ghép nhầm. Việc này ảnh hưởng cả majority vote tuổi/giới tính lẫn `watch_duration`.

### Nhận diện tuổi/giới tính có độ trễ

Worker nền và cache giúp giao diện mượt hơn nhưng tạo độ trễ ngắn trước khi có segment. Không nên xóa worker và gọi model Keras trên mọi frame vì vòng lặp camera sẽ dễ bị khựng.

### Chất lượng khuôn mặt ở xa

Độ phân giải stream cao hơn giúp face crop rõ hơn, nhưng MediaPipe và gaze vẫn bị giới hạn bởi `VISION_PROCESS_WIDTH`. Tăng biến này có thể giúp nhận diện khuôn mặt xa nhưng sẽ tăng tải CPU.

## 11. Cách kiểm thử sau khi sửa gaze

Nên kiểm thử theo thứ tự:

1. Một người đứng yên nhìn thẳng màn hình: `Looking` phải ổn định.
2. Một người quay mặt hoặc nhìn ra ngoài: `Not looking` phải ổn định.
3. Một người chuyển qua lại giữa nhìn và không nhìn: `watch_duration` chỉ tăng lúc nhìn.
4. Hai người xuất hiện cùng lúc: kết quả người này không được ảnh hưởng người kia.
5. Đứng xa và đứng gần camera: detection và gaze vẫn hoạt động trong phạm vi mong muốn.
6. Trong lúc quảng cáo đang chạy: gaze vẫn tiếp tục được tính và report vẫn được gửi.
7. Tạm mất frame UDP: chương trình không crash.

Sau một lần phát quảng cáo, kiểm tra terminal Máy A có:

```text
Ad report queued: ...
```

và terminal Máy B có request thành công:

```text
POST /api/ad-play-logs/report ... 200/201
```

## 12. Tiêu chí để thay đổi được xem là tương thích

Thay đổi gaze được xem là không phá luồng hệ thống khi:

- Camera và quảng cáo vẫn hiển thị mượt.
- Age/gender vẫn trả về đúng cấu trúc cũ.
- Detection vẫn chứa `bbox`, `audience_segment_id` và `looking`.
- API chọn quảng cáo vẫn nhận được payload hợp lệ.
- API report vẫn nhận được `watch_duration >= 0`.
- Gaze vẫn chạy trong lúc quảng cáo phát.
- Lỗi dự đoán của một khuôn mặt không làm dừng chương trình.

