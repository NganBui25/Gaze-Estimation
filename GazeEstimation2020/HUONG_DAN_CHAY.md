# Hướng dẫn chạy `EyeTracking.py`

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
