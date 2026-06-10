import cv2
import math
import numpy as np
import torch
import tensorflow as tf  # type: ignore
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models.PupilNet import PupilNet_v2
from utils.eye_prediction import EyePrediction
from utils.eye_sample import EyeSample

from .config import (
    AGE_RANGES,
    AGE_RANGE_NAMES,
    AUDIENCE_SEGMENT_IDS,
    GENDER_THRESHOLD,
    IMG_SIZE,
)


def gender_from_prob(prob):
    return "Female" if prob >= GENDER_THRESHOLD else "Male"


def age_distribution_to_range_probs(age_probs):
    age_probs = np.asarray(age_probs, dtype=np.float32).reshape(-1)
    range_probs = np.array(
        [age_probs[low:high + 1].sum() for low, high in AGE_RANGES],
        dtype=np.float32,
    )
    total = float(range_probs.sum())
    if total > 0:
        range_probs /= total
    return range_probs


def audience_segment_id_for_prediction(gender, age_range):
    normalized_gender = str(gender).strip().lower()
    return AUDIENCE_SEGMENT_IDS.get(normalized_gender, {}).get(age_range)


def get_mediapipe_landmarks(mesh_landmarks, w, h):
    landmarks = getattr(mesh_landmarks, "landmark", mesh_landmarks)
    coords = np.zeros((468, 2), dtype=int)
    for i, landmark in enumerate(landmarks[:468]):
        coords[i] = [int(landmark.x * w), int(landmark.y * h)]
    return coords


def map_to_dlib_style(mp_shape):
    fake_shape = np.zeros((68, 2), dtype=float)
    right_eye_idx = [33, 160, 158, 133, 153, 144]
    left_eye_idx = [362, 385, 387, 263, 373, 380]

    for i, idx in enumerate(right_eye_idx):
        fake_shape[36 + i] = mp_shape[idx]
    for i, idx in enumerate(left_eye_idx):
        fake_shape[42 + i] = mp_shape[idx]
    return fake_shape


def preprocess_face_for_inference(face_bgr, img_size=IMG_SIZE):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    height, width = face_rgb.shape[:2]
    size = max(height, width)
    padded = np.zeros((size, size, 3), dtype=face_rgb.dtype)
    top = (size - height) // 2
    left = (size - width) // 2
    padded[top:top + height, left:left + width] = face_rgb
    face_rgb = cv2.resize(padded, (img_size, img_size), interpolation=cv2.INTER_AREA)
    face_rgb = face_rgb.astype(np.float32) / 255.0
    face_rgb = np.expand_dims(face_rgb, axis=0)
    return face_rgb


def predict_age_gender(model, face_bgr):
    input_img = preprocess_face_for_inference(face_bgr, IMG_SIZE)
    preds = model.predict(input_img, verbose=0)

    if not isinstance(preds, dict):
        raise ValueError("Age/gender model must return named output tensors")

    pred_gender = preds["gender_output"]
    age_distribution = preds["age_distribution_output"].reshape(-1)
    gender_prob = float(pred_gender[0][0])
    gender_label = gender_from_prob(gender_prob)
    range_probs = age_distribution_to_range_probs(age_distribution)
    age_range_index = int(np.argmax(range_probs))
    age_range = AGE_RANGE_NAMES[age_range_index]
    age_range_confidence = float(range_probs[age_range_index])
    audience_segment_id = audience_segment_id_for_prediction(gender_label, age_range)
    return (
        gender_label,
        gender_prob,
        age_range,
        age_range_confidence,
        audience_segment_id,
    )


# ---- Ánh xạ MediaPipe -> 7 điểm theo ĐÚNG thứ tự unityeyes_processed_14d.csv ----
# (kiểm chứng bằng PupilNet/check_feature.py, đối chiếu interior_margin của UnityEyes):
#   point_1 = khóe mắt TRONG (gốc (0,0))         point_2 = khóe mắt NGOÀI (|mag|=1)
#   point_3..6 = 4 điểm mí (dưới ×2, trên ×2)    point_7 = tâm đồng tử (từ PupilNet)
MP_RIGHT = {"inner": 133, "outer": 33, "lids": [144, 154, 160, 161]}   # mắt phải (ảnh bên trái)
MP_LEFT = {"inner": 362, "outer": 263, "lids": [373, 381, 387, 388]}   # mắt trái (ảnh bên phải)


def eye_points_7(mp_shape, spec, pupil):
    """Lấy 7 điểm (2D) theo thứ tự CSV [trong, ngoài, mí×4, đồng tử] từ mediapipe landmarks."""
    pts = [mp_shape[spec["inner"]], mp_shape[spec["outer"]]]
    pts += [mp_shape[i] for i in spec["lids"]]
    pts.append(np.asarray(pupil, dtype=np.float32))
    return np.asarray(pts, dtype=np.float32)


def build_feature_14d(pts7):
    """Dựng vector 14 chiều từ 7 điểm theo thứ tự CSV [trong, ngoài, mí×4, đồng tử].
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


def predict_gaze_deg(gaze_model, pts7):
    """Dự đoán đồng thời (yaw, pitch) theo ĐỘ. Tự lật dấu yaw nếu mắt bị mirror về canonical."""
    feat, mirrored = build_feature_14d(pts7)
    yaw, pitch = gaze_model.predict(feat)[0]
    if mirrored:
        yaw = -yaw                     # đưa yaw về hệ thực của ảnh
    return float(yaw), float(pitch)


def head_angles_from_matrix(matrix):
    """Tính (yaw, pitch) của ĐẦU theo độ từ facial transformation matrix của MediaPipe.

    Hệ camera của MediaPipe theo chuẩn OpenGL (+X phải, +Y lên, +Z hướng về camera);
    cột thứ 3 của ma trận xoay là trục +Z của khuôn mặt (hướng mặt) trong hệ camera.
    Quy ước dấu khớp với góc mắt khi vẽ lên ảnh: yaw dương = quay về bên phải ảnh,
    pitch dương = cúi xuống (phía dưới ảnh).
    """
    m = np.asarray(matrix, dtype=np.float32)
    fwd = m[:3, 2]
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
    pitch = math.degrees(math.atan2(-fwd[1], math.hypot(float(fwd[0]), float(fwd[2]))))
    return yaw, pitch


def face_bearing_deg(face_center, frame_w, frame_h, hfov_deg):
    """Góc phương vị (yaw, pitch) theo độ của khuôn mặt so với trục quang camera,
    suy từ vị trí tâm mặt trong khung hình và góc mở ngang của camera.
    Dấu theo hướng ảnh: phải ảnh = yaw dương, dưới ảnh = pitch dương."""
    fx = (0.5 * frame_w) / math.tan(math.radians(hfov_deg) * 0.5)
    bearing_yaw = math.degrees(math.atan2(face_center[0] - 0.5 * frame_w, fx))
    bearing_pitch = math.degrees(math.atan2(face_center[1] - 0.5 * frame_h, fx))
    return bearing_yaw, bearing_pitch


def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []

    for corner1, corner2, is_left in [(42, 45, True), (36, 39, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        estimated_radius = 0.5 * eye_width * scale

        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = inv_translate_mat * inv_scale_mat * inv_center_mat

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)

        if is_left:
            eye_image = np.fliplr(eye_image)

        eyes.append(
            EyeSample(
                # Machine A only needs the eye crop, transform, and side flag later.
                # Avoid copying the full-resolution gray frame twice per detected face.
                orig_img=eye_image,
                img=eye_image,
                transform_inv=inv_transform_mat,
                is_left=is_left,
                estimated_radius=estimated_radius,
            )
        )
    return eyes


def predict_pupil(pupil_model, device, eyes, ow=160, oh=96):
    result = []
    for eye in eyes:
        with torch.no_grad():
            x = torch.tensor([eye.img / 255.0], dtype=torch.float32).to(device)
            pupil = pupil_model(x.view(1, 1, 96, 160))
            pupil = np.asarray(pupil.cpu().numpy())
            if pupil.shape != (1, 2):
                continue

            tmp = pupil[0][0]
            pupil[0][0] = pupil[0][1] / 2
            pupil[0][1] = tmp / 2
            pupil = pupil * np.array([oh / 48, ow / 80])

            temp = np.zeros((1, 3))
            if eye.is_left:
                temp[:, 0] = ow - pupil[:, 1]
            else:
                temp[:, 0] = pupil[:, 1]
            temp[:, 1] = pupil[:, 0]
            temp[:, 2] = 1.0
            pupil = temp

            pupil = np.asarray(np.matmul(pupil, eye.transform_inv.T))[:, :2]
            result.append(EyePrediction(eye_sample=eye, landmarks=pupil, gaze=None))

    return result


def build_face_bbox(full_mp_shape, frame_width, frame_height, pad_ratio=0.30):
    x_min, y_min = np.min(full_mp_shape, axis=0)
    x_max, y_max = np.max(full_mp_shape, axis=0)

    pad_x = int((x_max - x_min) * pad_ratio)
    pad_y = int((y_max - y_min) * pad_ratio)

    x1 = max(0, int(x_min) - pad_x)
    y1 = max(0, int(y_min) - pad_y)
    x2 = min(frame_width, int(x_max) + pad_x)
    y2 = min(frame_height, int(y_max) + pad_y)
    return x1, y1, x2, y2
