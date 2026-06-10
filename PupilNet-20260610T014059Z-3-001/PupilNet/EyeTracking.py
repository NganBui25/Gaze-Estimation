import cv2
import math
import mediapipe as mp
import os
import numpy as np
import joblib
import sklearn
import torch
from utils.eye_sample import EyeSample
from utils.eye_prediction import EyePrediction
from models.PupilNet import PupilNet_v2
import torch.nn as nn
from mediapipe.python.solutions import face_mesh as mp_face_mesh

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Mô hình hồi quy đa đầu ra: nhận vector đặc trưng 14 chiều -> dự đoán ĐỒNG THỜI (yaw, pitch) theo độ.
# Thay cho model_x.pkl + model_y.pkl (dự đoán lần lượt x rồi y) trước đây.
# best_model.joblib là Pipeline(StandardScaler + MLP) huấn luyện ở folder EyeTracking/.
gaze_model = joblib.load('models/best_model.joblib')

model = PupilNet_v2()                # model for predicting pupil center
model.load_state_dict(torch.load('models/pupilnet_v5.pt', map_location=device))

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")python EyeTracking.py
model = model.to(device)
print(device)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5, # Cho phép nhận diện nhiều người
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_mediapipe_landmarks(mesh_landmarks, w, h):
    coords = np.zeros((468, 2), dtype=int)
    for i, landmark in enumerate(mesh_landmarks.landmark[:468]):
        coords[i] = [int(landmark.x * w), int(landmark.y * h)]
    return coords

def map_to_dlib_style(mp_shape):
    fake_shape = np.zeros((68, 2), dtype=float)
    # Mắt phải (Dlib 36-41) -> MediaPipe indices
    right_eye_idx = [33, 160, 158, 133, 153, 144]
    # Mắt trái (Dlib 42-47) -> MediaPipe indices
    left_eye_idx = [362, 385, 387, 263, 373, 380]
    
    for i, idx in enumerate(right_eye_idx):
        fake_shape[36 + i] = mp_shape[idx]
    for i, idx in enumerate(left_eye_idx):
        fake_shape[42 + i] = mp_shape[idx]
    return fake_shape

def predict_pupil(eyes, ow=160, oh=96):
    result = []
    for eye in eyes:
        with torch.no_grad():
            x = torch.tensor([eye.img/255.0], dtype=torch.float32).to(device)
            pupil = model(x.view(1, 1, 96, 160))
            pupil = np.asarray(pupil.cpu().numpy())
            assert pupil.shape == (1, 2)
            tmp = pupil[0][0]
            pupil[0][0] = pupil[0][1] / 2
            pupil[0][1] = tmp / 2
            pupil = pupil * np.array([oh/48, ow/80])
            temp = np.zeros((1, 3))
            if eye.is_left:
                temp[:, 0] = ow - pupil[:, 1]
            else:
                temp[:, 0] = pupil[:, 1]
            temp[:, 1] = pupil[:, 0]
            temp[:, 2] = 1.0
            pupil = temp
            assert pupil.shape == (1, 3)
            pupil = np.asarray(np.matmul(pupil, eye.transform_inv.T))[:, :2]
            assert pupil.shape == (1, 2)
            result.append(EyePrediction(eye_sample=eye, landmarks=pupil, gaze=None))
    
    return result

# ---- Ánh xạ MediaPipe -> 7 điểm theo ĐÚNG thứ tự unityeyes_processed_14d.csv ----
# Đã kiểm chứng bằng check_feature.py (đối chiếu interior_margin của UnityEyes):
#   point_1 = khóe mắt TRONG (gốc (0,0))         -> MediaPipe inner corner
#   point_2 = khóe mắt NGOÀI (|mag|=1)           -> MediaPipe outer corner
#   point_3 = im4  (mí dưới, giữa-ngoài, x~0.64) -> mí dưới ~x0.7
#   point_4 = im1  (mí dưới, gần trong, x~0.17)  -> mí dưới ~x0.25
#   point_5 = im12 (mí trên, giữa,      x~0.63)  -> mí trên ~x0.7
#   point_6 = im10 (mí trên, gần ngoài, x~0.90)  -> mí trên ~x0.85
#   point_7 = tâm đồng tử (từ PupilNet)
# x = tỉ lệ vị trí dọc theo trục khóe-trong(0) -> khóe-ngoài(1); mí dưới = +y, mí trên = -y (y thô).
# Chỉ số mediapipe (refine_landmarks=True) cho viền mắt; có thể tinh chỉnh bằng calibrate_mediapipe.py.
MP_RIGHT = {"inner": 133, "outer": 33,  "lids": [144, 154, 160, 161]}  # mắt phải (ảnh bên trái)
MP_LEFT  = {"inner": 362, "outer": 263, "lids": [373, 381, 387, 388]}  # mắt trái (ảnh bên phải)

def eye_points_7(mp_shape, spec, pupil):
    """Lấy 7 điểm (2D) theo thứ tự CSV [trong, ngoài, mí×4, đồng tử] từ mediapipe landmarks."""
    pts = [mp_shape[spec["inner"]], mp_shape[spec["outer"]]]
    pts += [mp_shape[i] for i in spec["lids"]]
    pts.append(np.asarray(pupil, dtype=np.float32))
    return np.asarray(pts, dtype=np.float32)

def build_feature_14d(pts7):
    """Dựng vector 14 chiều từ 7 điểm ĐÃ theo thứ tự CSV [trong, ngoài, mí×4, đồng tử].
    Tịnh tiến về khóe trong (gốc (0,0)), chuẩn hóa theo khoảng cách 2 khóe; y thô.
    Tự LẬT NGANG về canonical (khóe ngoài ở +x) — trả về (feat (1,14), mirrored).
    """
    pts7 = np.asarray(pts7, dtype=np.float32)
    inner, outer = pts7[0], pts7[1]
    norm = np.linalg.norm(inner - outer)
    feat = (pts7 - inner) / (norm + 1e-9)
    mirrored = feat[1, 0] < 0          # khóe ngoài đang ở -x => lật về canonical (+x)
    if mirrored:
        feat[:, 0] *= -1.0
    return feat.reshape(1, -1).astype(np.float32), mirrored

def predict_gaze_deg(pts7):
    """Dự đoán đồng thời (yaw, pitch) theo ĐỘ. Tự lật dấu yaw nếu mắt bị mirror về canonical."""
    feat, mirrored = build_feature_14d(pts7)
    yaw, pitch = gaze_model.predict(feat)[0]
    if mirrored:
        yaw = -yaw                     # đưa yaw về hệ thực của ảnh
    return float(yaw), float(pitch)

def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []

    # Segment eyes
    for corner1, corner2, is_left in [(42, 45, True), (36, 39, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        estimated_radius = 0.5 * eye_width * scale

        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)

        if is_left:
            eye_image = np.fliplr(eye_image)
            
        eyes.append(EyeSample(orig_img=frame.copy(),
                              img=eye_image,
                              transform_inv=inv_transform_mat,
                              is_left=is_left,
                              estimated_radius=estimated_radius))
    return eyes

"""

detector = dlib.get_frontal_face_detector() # face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # pretrained facial landmarks detector
"""

left = [36, 37, 38, 39, 40, 41] # choosing only eye`s landmarks
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0) # initializing webcam
ret, img = cap.read()
shape = None
"""
while True:
    ret, img = cap.read()
    orig_frame = img.copy()
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        
        eye_samples = segment_eyes(gray, shape)
        pupil_predicts = predict_pupil(eye_samples)
        
        left_eyes = list(filter(lambda x: x.eye_sample.is_left, pupil_predicts))
        right_eyes = list(filter(lambda x: not x.eye_sample.is_left, pupil_predicts))

        center_left = int(round(left_eyes[0].landmarks[0][0])), int(round(left_eyes[0].landmarks[0][1]))
        center_right = int(round(right_eyes[0].landmarks[0][0])), int(round(right_eyes[0].landmarks[0][1]))
        
        cv2.circle(img, center_left, 2, (0, 0, 255), -1)
        cv2.circle(img, center_right, 2, (0, 0, 255), -1)
        
        for (x, y) in shape[36:48]:
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        
        norm_right = np.sqrt(np.sum((np.array([shape[36][0], shape[36][1]]) - \
                                         np.array([shape[39][0], shape[39][1]])) ** 2))
        norm_left = np.sqrt(np.sum((np.array([shape[42][0], shape[42][1]]) - \
                                         np.array([shape[45][0], shape[45][1]])) ** 2))
        try:
            ldmks_right = (np.array([[shape[36][0], shape[36][1]],
                          [shape[37][0], shape[37][1]],
                          [shape[38][0], shape[38][1]],
                          [shape[39][0], shape[39][1]],
                          [shape[40][0], shape[40][1]],
                          [shape[41][0], shape[41][1]],
                          list(center_right)]) - [shape[36][0], shape[36][1]]) / norm_right
            ldmks_left = (np.array([[shape[42][0], shape[42][1]],
                          [shape[43][0], shape[43][1]],
                          [shape[44][0], shape[44][1]],
                          [shape[45][0], shape[45][1]],
                          [shape[46][0], shape[46][1]],
                          [shape[47][0], shape[47][1]],
                          list(center_left)]) - [shape[42][0], shape[42][1]]) / norm_left
            
            lookpt_right_x = model_x.predict(ldmks_right.reshape(1, -1))
            temp = np.append(ldmks_right.reshape(1, -1), lookpt_right_x)
            lookpt_right_y = model_y.predict(temp.reshape(1, -1))
            
            lookpt_left_x = model_x.predict(ldmks_left.reshape(1, -1))
            temp = np.append(ldmks_left.reshape(1, -1), lookpt_left_x)
            lookpt_left_y = model_y.predict(temp.reshape(1, -1))
            
            cv2.line(img, center_right, tuple([int(lookpt_right_x * norm_right * 1.5 + shape[36][0]+3),
                     int(lookpt_right_y * norm_right + shape[36][1])]), (0,255,0), 2)
            cv2.line(img, center_left, tuple([int(lookpt_left_x * norm_left * 1.5 + shape[42][0]+3),
                     int(lookpt_left_y * norm_left + shape[42][1])]), (0,255,0), 2)
        except:
            pass
            
    cv2.imshow('eyes', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to stop the program
        break
    
cv2.destroyAllWindows()
cap.release()
"""
"""
while True:
    ret, img = cap.read()
    if not ret: break
    h, w, _ = img.shape
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Lấy các điểm mốc theo pixel
            shape = get_mediapipe_landmarks(face_landmarks, w, h)
            
            # Mapping các chỉ số MediaPipe tương ứng với dlib để cắt mắt
            # Dlib 36, 39 (Phải) -> MP 33, 133
            # Dlib 42, 45 (Trái) -> MP 362, 263
            
            # Lưu ý: MediaPipe dùng index khác dlib, ta cần tạo một mảng landmarks giả lập
            # để hàm segment_eyes của bạn không bị lỗi
            fake_dlib_shape = np.zeros((68, 2), dtype=int)
            fake_dlib_shape[36] = shape[33]   # Mắt phải góc ngoài
            fake_dlib_shape[39] = shape[133]  # Mắt phải góc trong
            fake_dlib_shape[42] = shape[362]  # Mắt trái góc trong
            fake_dlib_shape[45] = shape[263]  # Mắt trái góc ngoài
            # Bạn có thể map thêm các điểm khác nếu cần vẽ viền mắt
            
            # Tiếp tục logic cũ của bạn
            try:
                eye_samples = segment_eyes(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), fake_dlib_shape)
                pupil_predicts = predict_pupil(eye_samples)
                
                # Vẽ kết quả lên màn hình
                for pred in pupil_predicts:
                    center = (int(pred.landmarks[0][0]), int(pred.landmarks[0][1]))
                    cv2.circle(img, center, 2, (0, 0, 255), -1)
            except Exception as e:
                print(f"Lỗi xử lý: {e}")

    cv2.imshow('Gaze Tracking with MediaPipe', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""
smooth_yaw, smooth_pitch = 0.0, 0.0   # góc lệch ngang/dọc đã làm mượt (độ)
alpha = 0.2
while True:
    ret, img = cap.read()
    if not ret: break
    h, w, _ = img.shape
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Lấy toàn bộ landmarks
            full_mp_shape = get_mediapipe_landmarks(face_landmarks, w, h)
            # Tạo shape giả lập dlib để tương thích code cũ
            shape = map_to_dlib_style(full_mp_shape)
            
            x_min, y_min = np.min(full_mp_shape, axis=0)
            x_max, y_max = np.max(full_mp_shape, axis=0)
            # Vẽ hình vuông màu xanh dương (Blue)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            try:
                # 1. Dự đoán tâm đồng tử
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                eye_samples = segment_eyes(gray, shape)
                pupil_predicts = predict_pupil(eye_samples)
                
                # Phân loại mắt trái/phải
                left_eyes = list(filter(lambda x: x.eye_sample.is_left, pupil_predicts))
                right_eyes = list(filter(lambda x: not x.eye_sample.is_left, pupil_predicts))

                if len(left_eyes) > 0 and len(right_eyes) > 0:
                    center_left = tuple(left_eyes[0].landmarks[0].astype(int))
                    center_right = tuple(right_eyes[0].landmarks[0].astype(int))

                    # Vẽ đồng tử và viền mắt
                    cv2.circle(img, center_left, 2, (0, 0, 255), -1)
                    cv2.circle(img, center_right, 2, (0, 0, 255), -1)
                    for (x, y) in shape[36:48].astype(int):
                        cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

                    # 2. Dự đoán hướng nhìn ĐỒNG THỜI (yaw, pitch) bằng best_model.
                    #    Lấy 7 điểm theo đúng thứ tự CSV trực tiếp từ mediapipe landmarks (full_mp_shape),
                    #    dùng chỉ số đã ánh xạ MP_RIGHT/MP_LEFT (khớp interior_margin của UnityEyes).
                    pts7_r = eye_points_7(full_mp_shape, MP_RIGHT, center_right)
                    pts7_l = eye_points_7(full_mp_shape, MP_LEFT, center_left)
                    yaw_r, pitch_r = predict_gaze_deg(pts7_r)
                    yaw_l, pitch_l = predict_gaze_deg(pts7_l)
                    norm_right = np.linalg.norm(full_mp_shape[133] - full_mp_shape[33])  # vẽ mũi tên
                    norm_left = np.linalg.norm(full_mp_shape[362] - full_mp_shape[263])

                    # Góc lệch trung bình 2 mắt (độ)
                    avg_yaw = (yaw_r + yaw_l) / 2
                    avg_pitch = (pitch_r + pitch_l) / 2

                    # 3. Vẽ vector hướng nhìn (đường xanh lá) — đổi góc (độ) sang hướng pixel.
                    #    Lưu ý dấu: nếu hướng vẽ ngược, đổi dấu sin(pitch)/sin(yaw) cho khớp camera.
                    GAZE_LEN = 2.5  # hệ số độ dài mũi tên (theo bề rộng mắt)
                    end_r = (int(center_right[0] + math.sin(math.radians(yaw_r)) * norm_right * GAZE_LEN),
                             int(center_right[1] + math.sin(math.radians(pitch_r)) * norm_right * GAZE_LEN))
                    end_l = (int(center_left[0] + math.sin(math.radians(yaw_l)) * norm_left * GAZE_LEN),
                             int(center_left[1] + math.sin(math.radians(pitch_l)) * norm_left * GAZE_LEN))
                    cv2.line(img, center_right, end_r, (0, 255, 0), 2)
                    cv2.line(img, center_left, end_l, (0, 255, 0), 2)

                    # 4. Làm mượt EMA (theo độ)
                    smooth_yaw = alpha * avg_yaw + (1 - alpha) * smooth_yaw
                    smooth_pitch = alpha * avg_pitch + (1 - alpha) * smooth_pitch

                    # 5. Vùng "biển quảng cáo" tính bằng ĐỘ (hiệu chỉnh bằng cách nhìn 4 góc màn hình)
                    YAW_MIN, YAW_MAX = -20.0, 20.0      # góc lệch ngang cho phép (độ)
                    PITCH_MIN, PITCH_MAX = -15.0, 25.0  # góc lệch dọc cho phép (độ)
                    is_looking_at_screen = (YAW_MIN < smooth_yaw < YAW_MAX) and \
                                           (PITCH_MIN < smooth_pitch < PITCH_MAX)

                    # 6. Hiển thị kết quả
                    if is_looking_at_screen:
                        status_text = "ENGAGED: LOOKING AT BILLBOARD"
                        color = (0, 255, 0)  # Xanh lá - đang tương tác
                    else:
                        # Độ lệch so với tâm vùng màn hình (độ)
                        cx_deg = (YAW_MIN + YAW_MAX) / 2
                        cy_deg = (PITCH_MIN + PITCH_MAX) / 2
                        angle_off = math.sqrt((smooth_yaw - cx_deg) ** 2 + (smooth_pitch - cy_deg) ** 2)
                        status_text = f"NOT LOOKING (Off by {angle_off:.1f} deg)"
                        color = (0, 0, 255)
                    cv2.putText(img, status_text, (x_min, y_min - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    print(f"yaw: {smooth_yaw:.1f} deg, pitch: {smooth_pitch:.1f} deg")
            except Exception as e:
                print(f"Error: {e}")

    cv2.imshow('Gaze Tracking', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()