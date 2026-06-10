# PupilNet — Eye Tracking / Ước lượng hướng nhìn thời gian thực

Theo dõi mắt qua webcam và ước lượng **hướng nhìn (yaw, pitch)** theo thời gian thực:

1. **MediaPipe FaceMesh** → lấy landmark khuôn mặt, cắt vùng 2 mắt.
2. **PupilNet (CNN, PyTorch)** → dự đoán **tâm đồng tử** trong mỗi vùng mắt.
3. **best_model (MLP, scikit-learn)** → từ **vector đặc trưng 14 chiều** (6 điểm viền mắt + tâm đồng tử) dự đoán **ĐỒNG THỜI** góc lệch ngang `yaw` và dọc `pitch` (độ).

> **Thay đổi so với bản cũ:** trước đây dùng `model_x.pkl` + `model_y.pkl` (2 SVR, dự đoán **lần lượt** x rồi y).
> Nay thay bằng **một** `best_model.joblib` nhận vector 14 chiều và dự đoán **đồng thời** (yaw, pitch).
> `best_model` được huấn luyện ở dự án `EyeTracking/` (xem `../EyeTracking/`).

---

## 1. Cấu trúc thư mục

```
PupilNet/
├── EyeTracking.py            # ★ Chương trình chính: webcam -> mediapipe -> PupilNet -> best_model
├── check_feature.py          # Kiểm chứng định dạng vector 14 chiều khớp với CSV huấn luyện
├── calibrate_mediapipe.py    # Hiệu chỉnh chỉ số mediapipe của 4 điểm mí trên khuôn mặt thật
├── get_pretrained_model.sh   # Tải shape_predictor_68 (chỉ cần nếu dùng dlib thay mediapipe)
├── models/
│   ├── PupilNet.py           # Định nghĩa kiến trúc CNN PupilNet_v2
│   ├── pupilnet_v5.pt        # ⚠️ Trọng số CNN — PHẢI tự sinh từ notebooks/pytorch_pipeline.ipynb
│   ├── best_model.joblib     # Pipeline(StandardScaler + MLP) dự đoán (yaw, pitch) — copy từ EyeTracking/
│   └── best_model_meta.json  # tên model, val-MAE, siêu tham số, feature/target
├── notebooks/
│   ├── pytorch_pipeline.ipynb # Huấn luyện CNN PupilNet + trực quan hóa + cache (xem mục 4)
│   ├── sandbox.ipynb          # (CŨ) huấn luyện model_x/model_y — đã thay bằng best_model, để tham khảo
│   ├── imgs/                  # ảnh + json UnityEyes
│   ├── cache/                 # (sinh tự động) ảnh tiền xử lý .npy để train nhanh
│   └── *.png                  # (sinh tự động) loss_curve, test_scatter, test_samples
└── utils/
    ├── eye_sample.py          # lớp EyeSample (ảnh mắt đã cắt + ma trận biến đổi)
    └── eye_prediction.py      # lớp EyePrediction (kết quả dò đồng tử)
```

---

## 2. Yêu cầu môi trường

```bash
pip install opencv-python mediapipe numpy torch scikit-learn==1.7.2 joblib
```

> `best_model.joblib` lưu bằng **scikit-learn 1.7.2** — nạp lại nên dùng đúng phiên bản.

---

## 3. Vector đặc trưng 14 chiều (QUAN TRỌNG)

`best_model` được học trên `unityeyes_processed_14d.csv`. Định dạng đã **kiểm chứng** bằng
`check_feature.py` — gồm 7 điểm (mỗi điểm `x,y`) theo thứ tự:

| Điểm | Ý nghĩa | Đặc điểm chuẩn hóa |
|------|---------|--------------------|
| point_1 | **khóe mắt TRONG** | = gốc `(0,0)` |
| point_2 | **khóe mắt NGOÀI** | `|mag| = 1` (chuẩn hóa theo khoảng cách 2 khóe) |
| point_3 | mí dưới, giữa-ngoài | `+y` |
| point_4 | mí dưới, gần trong | `+y` |
| point_5 | mí trên, giữa | `−y` |
| point_6 | mí trên, gần ngoài | `−y` |
| point_7 | **tâm đồng tử** | từ PupilNet; phương sai lớn nhất |

Quy tắc: **tịnh tiến về khóe trong (gốc 0,0)**, **chuẩn hóa theo khoảng cách 2 khóe**, **y thô** (không lật),
và **lật về canonical** (khóe ngoài luôn ở `+x`) — mắt nào bị lật thì đổi dấu `yaw`.
Toàn bộ nằm trong `build_feature_14d()` / `eye_points_7()` của `EyeTracking.py`.

Ánh xạ MediaPipe (refine_landmarks=True) đang dùng:
- `MP_RIGHT = {inner: 133, outer: 33,  lids: [144, 154, 160, 161]}`
- `MP_LEFT  = {inner: 362, outer: 263, lids: [373, 381, 387, 388]}`

---

## 4. Quy trình chạy

### Bước 1 — Huấn luyện CNN dò tâm đồng tử (`pupilnet_v5.pt`)
Mở `notebooks/pytorch_pipeline.ipynb` (cần dữ liệu UnityEyes trong `notebooks/imgs/`). Notebook gồm 3 phần:

1. **Định nghĩa** (data, kiến trúc `PupilNet_v2`, `device`...) — chạy các cell này trước.
2. **📊 Trực quan hóa & đánh giá độ chính xác** — huấn luyện có `tqdm` + ghi lịch sử loss train/val,
   rồi vẽ: đường loss theo epoch, **bảng chỉ số train/val/test** (MSE, RMSE, sai số pixel),
   scatter dự đoán-vs-thực, và ảnh minh họa tâm đồng tử. Tập **test = 938 ảnh (index 100000–100937)**,
   không trùng train. Xuất `pupilnet_loss_curve.png`, `pupilnet_test_scatter.png`, `pupilnet_test_samples.png`.
3. **🚀 Tăng tốc bằng CACHE** — tiền xử lý ảnh 1 lần ra `.npy` (thư mục `cache/`), train đọc thẳng từ RAM
   → mỗi epoch nhanh hơn nhiều lần. Dùng cho máy nghẽn CPU / không có GPU mạnh.

Cả 2 đường (mục 2 và 3) đều dùng `num_workers=0` để **tránh treo** (DataLoader `num_workers>0` trong
notebook trên Windows hay deadlock) và đều lưu `../models/pupilnet_v5.pt`.

> Cell `train(model, optimizer)` GỐC dùng `num_workers=6` — **bỏ qua trên Windows** (dễ treo); dùng cell
> "Train có lịch sử" hoặc mục CACHE thay thế.

### Bước 2 — (đã có sẵn) model dự đoán góc nhìn
`models/best_model.joblib` đã được copy từ `EyeTracking/`. Muốn huấn luyện lại: chạy
`../EyeTracking/training.ipynb` rồi copy `best_model.joblib` + `best_model_meta.json` vào `models/`.

### Bước 3 — Kiểm chứng vector đặc trưng (khuyên chạy trước khi mở webcam)
```bash
python check_feature.py
```
→ In hợp đồng đặc trưng của CSV, MAE round-trip của best_model (~2.2), và đối chiếu công thức.

### Bước 4 — Hiệu chỉnh chỉ số mediapipe trên khuôn mặt thật
```bash
python calibrate_mediapipe.py            # chụp 1 ảnh webcam
python calibrate_mediapipe.py face.jpg   # hoặc từ ảnh có khuôn mặt
```
→ Với mỗi `point_2..6`, in chỉ số mediapipe ở viền mắt **gần nhất** với vị trí chuẩn của CSV.
Nếu khác chỉ số đang dùng và khoảng cách nhỏ hơn → cập nhật `MP_RIGHT/MP_LEFT["lids"]`
trong `EyeTracking.py` theo thứ tự `[point_3, point_4, point_5, point_6]`.

### Bước 5 — Chạy ước lượng hướng nhìn
```bash
python EyeTracking.py
```
Webcam hiện: khung mặt, tâm đồng tử, vector hướng nhìn, và trạng thái "đang nhìn vào màn hình/biển
quảng cáo" (ngưỡng tính bằng **độ**). Nhấn **q** để thoát.

---

## 5. Tham số cần hiệu chỉnh trong `EyeTracking.py`

| Tham số | Ý nghĩa |
|---------|---------|
| `MP_RIGHT/MP_LEFT["lids"]` | chỉ số mediapipe 4 điểm mí — chốt bằng `calibrate_mediapipe.py` |
| `GAZE_LEN` | độ dài mũi tên hướng nhìn khi vẽ (đổi dấu `sin` nếu vẽ ngược) |
| `YAW_MIN/MAX`, `PITCH_MIN/MAX` | vùng "đang nhìn màn hình" (độ) — nhìn 4 góc màn hình để căn |
| `alpha` | hệ số làm mượt EMA |

---

## 6. Ghi chú
- **Khóe mắt làm gốc:** dùng **khóe TRONG** (không phải khóe ngoài) — đúng với dữ liệu huấn luyện.
  Việc lật ảnh mắt trái trong `segment_eyes` chỉ phục vụ CNN dò đồng tử, đã được lật ngược lại,
  không ảnh hưởng vector đặc trưng góc nhìn.
- **Mirror trái/phải:** xử lý tự động trong `build_feature_14d` (khóe ngoài đưa về `+x`, đổi dấu `yaw`).
- **Hạn chế:** mediapipe chỉ cho ~16 điểm viền mắt (so với UnityEyes), nên 4 điểm mí là gần đúng;
  gốc + 2 khóe + đồng tử + orientation thì khớp chính xác.
