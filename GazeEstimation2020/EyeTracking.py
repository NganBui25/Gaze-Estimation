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
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_x = joblib.load('models/model_x.pkl') # model for predicting x-coordinate of gaze vector
model_y = joblib.load('models/model_y.pkl') # model for predicting y-coordinate of gaze vector
model = PupilNet_v2()                # model for predicting pupil center
model.load_state_dict(torch.load('models/pupilnet_v5.pt', map_location=device))

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")python EyeTracking.py
model = model.to(device)
print(device)

FACE_LANDMARKER_MODEL_PATH = os.path.join("models", "face_landmarker.task")
if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
    raise FileNotFoundError(
        f"Missing MediaPipe face landmark model: {FACE_LANDMARKER_MODEL_PATH}"
    )

face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH),
        num_faces=5,
        running_mode=vision.RunningMode.IMAGE,
    )
)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_mediapipe_landmarks(mesh_landmarks, w, h):
    landmarks = getattr(mesh_landmarks, "landmark", mesh_landmarks)
    coords = np.zeros((468, 2), dtype=int)
    for i, landmark in enumerate(landmarks[:468]):
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

cap = cv2.VideoCapture("tcp://192.168.137.183:5000") # initializing webcam
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
smooth_x, smooth_y = 0.0, 0.0
alpha = 0.2
while True:
    ret, img = cap.read()
    if not ret: break
    h, w, _ = img.shape
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
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

                    # 2. Logic dự đoán hướng nhìn (Gaze Prediction)
                    # Tính toán định mức (normalization)
                    norm_right = np.linalg.norm(shape[36] - shape[39])
                    norm_left = np.linalg.norm(shape[42] - shape[45])

                    # Chuẩn hóa dữ liệu đầu vào cho model_x, model_y
                    # Mắt phải
                    ldmks_right = (np.vstack([shape[36:42], center_right]) - shape[36]) / norm_right
                    feat_r = ldmks_right.reshape(1, -1) # Ép về 2D (1 hàng, nhiều cột)
                    look_x_r = model_x.predict(feat_r)[0]
                    

                    #look_x_r = model_x.predict(ldmks_right.reshape(1, -1)[0])
                    #look_y_r = model_y.predict(np.append(ldmks_right.reshape(
                    # ]1, -1), look_x_r).reshape(1, -1)[0])
                    feat_y_r = np.append(feat_r.flatten(), look_x_r).reshape(1, -1)
                    look_y_r = model_y.predict(feat_y_r)[0]

                    # Mắt trái
                    ldmks_left = (np.vstack([shape[42:48], center_left]) - shape[42]) / norm_left
                    #look_x_l = model_x.predict(ldmks_left.reshape(1, -1)[0])
                    #look_y_l = model_y.predict(np.append(ldmks_left.reshape(1, -1), look_x_l).reshape(1, -1)[0])
                    feat_l = ldmks_left.reshape(1, -1)
                    look_x_l = model_x.predict(feat_l)[0]
                    feat_y_l = np.append(feat_l.flatten(), look_x_l).reshape(1, -1)
                    look_y_l = model_y.predict(feat_y_l)[0]

                    # 3. Vẽ vector hướng nhìn (Đường xanh lá)
                    end_r = (int(look_x_r * norm_right * 1.5 + shape[36][0]), int(look_y_r * norm_right + shape[36][1]))
                    end_l = (int(look_x_l * norm_left * 1.5 + shape[42][0]), int(look_y_l * norm_left + shape[42][1]))
                    cv2.line(img, center_right, end_r, (0, 255, 0), 2)
                    cv2.line(img, center_left, end_l, (0, 255, 0), 2)

                    avg_look_x = (look_x_r + look_x_l) / 2
                    avg_look_y = (look_y_r + look_y_l) / 2
                    
                    deviation = math.sqrt(avg_look_x**2 + avg_look_y**2)
                    angle_deg = deviation * 45 # Hệ số 45 này Ngân có thể căn chỉnh lại

                    # Ngưỡng nhìn thẳng (ví dụ dưới 15 độ là đang nhìn cam)
                    """
                    if angle_deg < 15.0:
                        status_text = "STATUS: LOOKING AT CAMERA"
                        color = (0, 255, 0) # Xanh lá
                    else:
                        status_text = f"STATUS: LOOKING AWAY ({angle_deg:.1f} deg)"
                        color = (0, 0, 255) # Đỏ
                    cv2.putText(img, status_text, (x_min, y_min - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    """
                    # 1. Áp dụng bộ lọc làm mượt EMA
                    avg_raw_x = (look_x_r + look_x_l) / 2
                    avg_raw_y = (look_y_r + look_y_l) / 2

                    smooth_x = alpha * avg_raw_x + (1 - alpha) * smooth_x
                    smooth_y = alpha * avg_raw_y + (1 - alpha) * smooth_y

                    # 2. ĐỊNH NGHĨA KHÔNG GIAN BIỂN QUẢNG CÁO (Calibration)
                    # Ngân hãy ngồi nhìn vào 4 góc màn hình để tìm ra các con số này nhé:
                    SCREEN_X_MIN, SCREEN_X_MAX = -0.5, 0.5  # Ví dụ vùng nhìn chiều ngang
                    SCREEN_Y_MIN, SCREEN_Y_MAX = -0.15, 0.8  # Ví dụ vùng nhìn chiều dọc (thường lệch xuống dưới)

                    # Kiểm tra xem có đang nhìn vào "biển quảng cáo" không
                    is_looking_at_screen = (SCREEN_X_MIN < smooth_x < SCREEN_X_MAX) and \
                                        (SCREEN_Y_MIN < smooth_y < SCREEN_Y_MAX)

                    # 3. HIỂN THỊ KẾT QUẢ
                    if is_looking_at_screen:
                        status_text = "ENGAGED: LOOKING AT BILLBOARD"
                        color = (0, 255, 0) # Xanh lá - Đang tương tác
                    else:
                        # Tính độ lệch so với tâm màn hình (ví dụ tâm là 0, 0.3)
                        error_x = smooth_x - 0
                        error_y = smooth_y - 0.3
                        angle_off = math.sqrt(error_x**2 + error_y**2) * 30
                        status_text = f"NOT LOOKING (Off by {angle_off:.1f} deg)"
                        color = (0, 0, 255)
                    cv2.putText(img, status_text, (x_min, y_min - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    print(f"X: {smooth_x:.2f}, Y: {smooth_y:.2f}")
            except Exception as e:
                print(f"Error: {e}")

    cv2.imshow('Gaze Tracking', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()