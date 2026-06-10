# Hướng dẫn chạy `EyeTracking.py`

> ⚠️ **Tài liệu này nói về demo đơn lẻ `EyeTracking.py` (legacy).**
> Pipeline đám đông chính (`machine_a.py`) đã chuyển sang logic hướng nhìn mới
> (PupilNet train lại + best_model, góc theo độ, bù head pose) — xem
> **`README_GAZE_PUPILNET.md`** để biết cách chạy, hiệu chỉnh góc theo camera thực tế
> và danh sách thay đổi.

File này mô tả cách chạy demo theo dõi mắt trong thư mục `GazeEstimation2020`.

## 1. Chuẩn bị môi trường

Dự án đang chạy tốt trong virtual environment ở:

```powershell
D:\PBL5\.venv\Scripts\python.exe
```

Nếu bạn chưa có môi trường này, có thể tạo mới bằng:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Sau đó cài thư viện:

```powershell
pip install -r requirements.txt
```

## 2. Kiểm tra file model

Trong thư mục `models/` cần có các file sau:

- `pupilnet_v5.pt`
- `model_x.pkl`
- `model_y.pkl`
- `face_landmarker.task`

Nếu thiếu `face_landmarker.task`, tải lại file này vào `models/` trước khi chạy.

## 3. Chạy chương trình

Mở terminal tại thư mục `GazeEstimation2020` rồi chạy:

```powershell
python EyeTracking.py
```

Nếu bạn muốn dùng đúng môi trường đã được kiểm tra trước đó, có thể chạy trực tiếp:

```powershell
D:\PBL5\.venv\Scripts\python.exe EyeTracking.py
```

## 4. Dữ liệu camera đầu vào

Hiện tại script đang dùng nguồn video:

```python
cv2.VideoCapture("tcp://192.168.1.8:5000")
```

Nếu bạn muốn chạy bằng webcam mặc định trên máy, đổi dòng này thành:

```python
cv2.VideoCapture(0)
```

## 5. Cách thoát chương trình

Trong cửa sổ hiển thị, nhấn phím `q` để dừng.

## 6. Ghi chú

- Khi khởi động, script có thể hiện cảnh báo từ `scikit-learn` do khác phiên bản khi mở file `pkl`. Đây là cảnh báo, không phải lỗi dừng chương trình.
- Nếu gặp lỗi thiếu thư viện, hãy kiểm tra lại đúng môi trường Python đang được kích hoạt.

## 7. Chế độ Smart Billboard cho Máy A

Nếu bạn chạy file `combined_tracking_age_gender.py`, script sẽ hoạt động theo luồng sau:

1. Nhận video từ Raspberry Pi Camera qua UDP: `udp://0.0.0.0:5000`.
2. Đọc tín hiệu từ ESP32 để biết trạng thái đèn.
3. Khi tín hiệu là `Light`, hệ thống bắt đầu gom dữ liệu người xem trong một cửa sổ khoảng 3-5 giây.
4. Sau đó gửi yêu cầu chọn quảng cáo đến Máy B tại:

```text
http://<MAY_B_IP>:5000/api/advertisements/select
```

5. Khi nhận được `ad_id`, `media_filename` và `duration_seconds`, script sẽ phát file video quảng cáo cục bộ.
6. Trong lúc phát quảng cáo, hệ thống tiếp tục ghi nhận từng người xem và thời gian xem.
7. Ngay khi quảng cáo kết thúc, script gửi báo cáo cuối cùng đến:

```text
http://<MAY_B_IP>:5000/api/ad-play-logs/report
```

### Biến môi trường nên đặt

Bạn có thể đặt các biến sau trước khi chạy:

```powershell
$env:MAY_B_IP = "192.168.1.10"
$env:VIDEO_SOURCE = "udp://0.0.0.0:5000"
$env:SENSOR_SERIAL_PORT = "COM3"
$env:AUDIENCE_WINDOW_SECONDS = "4"
$env:AD_MEDIA_ROOT = "D:\PBL5\Gaze-Estimation\GazeEstimation2020"
```

Nếu ESP32 gửi tín hiệu qua socket thay vì serial, dùng:

```powershell
$env:SENSOR_SOCKET_HOST = "192.168.1.20"
$env:SENSOR_SOCKET_PORT = "5001"
```

### Lưu ý quan trọng

- Tín hiệu hợp lệ của cảm biến là `Light` và `Dark`.
- Các mốc thời gian trong payload đều theo chuẩn ISO 8601, ví dụ: `2026-04-27T22:30:00`.
- File video quảng cáo phải tồn tại trên máy Máy A, hoặc được tìm thấy trong thư mục `AD_MEDIA_ROOT`.
