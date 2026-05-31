# GazeEstimation2026

Dự án `GazeEstimation2026` là một hệ thống nhận diện và theo dõi ánh mắt (eye tracking) dùng MediaPipe + PyTorch.

## 🚀 Tính năng
- Nhận diện mặt và landmark bằng MediaPipe Face Mesh
- Trích xuất điểm mắt và ước lượng vị trí nhìn
- Sử dụng mô hình `PupilNet_v2` cho định vị trung tâm đồng tử
- Kết hợp mô hình scikit-learn (`model_x.pkl`, `model_y.pkl`) để dự đoán vector gaze (x, y)

## 📦 Cài đặt
1. Tạo và kích hoạt virtual environment:
   - Windows PowerShell:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - Windows CMD:
     ```cmd
     python -m venv venv
     venv\Scripts\activate.bat
     ```

2. Cài thư viện:
   ```bash
   pip install -r requirements.txt
   ```

3. Kiểm tra/ tải model (nếu cần):
   ```bash
   bash get_pretrained_model.sh
   ```

## ▶️ Chạy demo
Trong thư mục dự án, chạy:
```bash
python EyeTracking.py
```

## 🗂 Cấu trúc thư mục
- `EyeTracking.py`: script chính demo.
- `models/PupilNet.py`: định nghĩa mạng neuron, file trọng số `pupilnet_v5.pt`.
- `models/model_x.pkl`, `models/model_y.pkl`: model học máy cho gaze.
- `utils/eye_sample.py`, `utils/eye_prediction.py`: tiện ích xử lý dữ liệu mắt.

## ⚠️ Lưu ý
- Sử dụng Python >= 3.11 khuyến nghị (bạn đã tạo venv python3.11).
- Nếu báo lỗi `ModuleNotFoundError: No module named 'mediapipe'`, kiểm tra đúng venv đã active và chạy lại `pip install -r requirements.txt`.
- Nếu báo thiếu file model (e.g., `pupilnet_v5.pt`), cần đặt file trong `models/`.

## 🛠 Kéo lên GitHub nhánh mới
1. Tạo nhánh mới:
   ```bash
   git checkout -b gazeEstimation
   ```
2. Push nhánh:
   ```bash
   git push -u origin gazeEstimation
   ```

## 📌 Ghi chú
Bỏ các file `__pycache__` khỏi repo bằng `.gitignore` (nếu chưa có).