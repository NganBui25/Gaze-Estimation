import math
import os
import threading
import time

import cv2  # type: ignore
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf  # type: ignore
import torch

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models.PupilNet import PupilNet_v2
from utils.eye_prediction import EyePrediction
from utils.eye_sample import EyeSample


# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "tcp://192.168.137.183:5000")
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_BUFFERSIZE = 1
AGE_GENDER_EVERY_N_FRAMES = 10

GAZE_MODEL_X_PATH = os.path.join(BASE_DIR, "models", "model_x.pkl")
GAZE_MODEL_Y_PATH = os.path.join(BASE_DIR, "models", "model_y.pkl")
PUPIL_MODEL_PATH = os.path.join(BASE_DIR, "models", "pupilnet_v5.pt")
FACE_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
AGE_GENDER_MODEL_PATH = os.path.join(ROOT_DIR, "AgeAndGender", "ResNet50_128_phase2_new.keras")

IMG_SIZE = 128
MAX_AGE = 116
GENDER_THRESHOLD = 0.5


# =========================
# CUSTOM OBJECTS FOR AGE/GENDER MODEL
# =========================
@tf.keras.utils.register_keras_serializable(package="agender")
class BinaryF1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1.0 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1.0 - y_pred)))

    def result(self):
        numerator = 2.0 * self.tp
        denominator = numerator + self.fp + self.fn
        return tf.math.divide_no_nan(numerator, denominator)

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


@tf.keras.utils.register_keras_serializable(package="agender")
class CoralLoss(tf.keras.losses.Loss):
    def __init__(self, name="coral_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return tf.reduce_sum(losses, axis=-1)


@tf.keras.utils.register_keras_serializable(package="agender")
class CoralMAE(tf.keras.metrics.Metric):
    def __init__(self, name="mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_age = tf.reduce_sum(tf.cast(y_true, tf.float32), axis=-1)
        y_pred_age = tf.reduce_sum(tf.nn.sigmoid(y_pred), axis=-1)
        errors = tf.abs(y_true_age - y_pred_age)
        self.total.assign_add(tf.reduce_sum(errors))
        self.count.assign_add(tf.cast(tf.size(errors), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0.0)


# =========================
# HELPERS
# =========================
def gender_from_prob(prob):
    return "Female" if prob >= GENDER_THRESHOLD else "Male"


def coral_logits_to_age_np(logits):
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.sum(axis=-1)


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
    face_rgb = cv2.resize(face_rgb, (img_size, img_size))
    face_rgb = face_rgb.astype(np.float32)
    face_rgb = np.expand_dims(face_rgb, axis=0)
    return face_rgb


def predict_age_gender(model, face_bgr):
    input_img = preprocess_face_for_inference(face_bgr, IMG_SIZE)
    preds = model.predict(input_img, verbose=0)

    if isinstance(preds, dict):
        pred_gender = preds["gender_output"]
        pred_age = preds["age_output"]
    else:
        pred_gender, pred_age = preds

    gender_prob = float(pred_gender[0][0])
    age_pred = float(coral_logits_to_age_np(pred_age)[0])
    age_pred = max(0.0, min(age_pred, float(MAX_AGE)))
    gender_label = gender_from_prob(gender_prob)
    return gender_label, gender_prob, age_pred


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
                orig_img=frame.copy(),
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


def build_face_bbox(full_mp_shape, frame_width, frame_height, pad_ratio=0.15):
    x_min, y_min = np.min(full_mp_shape, axis=0)
    x_max, y_max = np.max(full_mp_shape, axis=0)

    pad_x = int((x_max - x_min) * pad_ratio)
    pad_y = int((y_max - y_min) * pad_ratio)

    x1 = max(0, int(x_min) - pad_x)
    y1 = max(0, int(y_min) - pad_y)
    x2 = min(frame_width, int(x_max) + pad_x)
    y2 = min(frame_height, int(y_max) + pad_y)
    return x1, y1, x2, y2


class LatestFrameGrabber:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, FRAME_BUFFERSIZE)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.latest_frame = None
        self.latest_ok = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped.is_set():
            if not self.cap.isOpened():
                self.latest_ok = False
                break

            ok, frame = self.cap.read()
            with self.lock:
                self.latest_ok = ok
                if ok:
                    self.latest_frame = frame

            if not ok:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return self.latest_ok, self.latest_frame.copy()

    def release(self):
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


# =========================
# LOAD MODELS
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_x = joblib.load(GAZE_MODEL_X_PATH)
model_y = joblib.load(GAZE_MODEL_Y_PATH)

pupil_model = PupilNet_v2()
pupil_model.load_state_dict(torch.load(PUPIL_MODEL_PATH, map_location=device))
pupil_model = pupil_model.to(device)
pupil_model.eval()

custom_objects = {
    "BinaryF1Score": BinaryF1Score,
    "CoralLoss": CoralLoss,
    "CoralMAE": CoralMAE,
}

age_gender_model = tf.keras.models.load_model(
    AGE_GENDER_MODEL_PATH,
    custom_objects=custom_objects,
    compile=False,
)

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

print(f"Using device: {device}")
print("Loaded gaze, pupil, and age/gender models successfully.")


# =========================
# MAIN LOOP
# =========================
grabber = LatestFrameGrabber(VIDEO_SOURCE).start()

if not grabber.cap.isOpened():
    raise RuntimeError(
        f"Không mở được webcam/stream từ {VIDEO_SOURCE}. "
        "Hãy kiểm tra IP Raspberry Pi hoặc đặt biến môi trường VIDEO_SOURCE."
    )

print("Nhấn Q để thoát...")

smooth_x, smooth_y = 0.0, 0.0
alpha = 0.2
frame_index = 0
last_age_gender = None

while True:
    ret, frame = grabber.read()
    if not ret:
        time.sleep(0.05)
        continue

    display_frame = frame.copy()
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            full_mp_shape = get_mediapipe_landmarks(face_landmarks, w, h)
            shape = map_to_dlib_style(full_mp_shape)
            x1, y1, x2, y2 = build_face_bbox(full_mp_shape, w, h)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0 and frame_index % AGE_GENDER_EVERY_N_FRAMES == 0:
                try:
                    last_age_gender = predict_age_gender(
                        age_gender_model, face_crop
                    )
                except Exception as exc:
                    last_age_gender = None
                    print("Age/Gender prediction error:", exc)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            if last_age_gender is not None:
                gender_label, gender_prob, age_pred = last_age_gender
                cv2.putText(
                    display_frame,
                    f"{gender_label} ({gender_prob:.2f}) | Age: {age_pred:.1f}",
                    (x1, max(20, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    display_frame,
                    "Age/Gender: stale or unavailable",
                    (x1, max(20, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eye_samples = segment_eyes(gray, shape)
                pupil_predicts = predict_pupil(pupil_model, device, eye_samples)

                left_eyes = [x for x in pupil_predicts if x.eye_sample.is_left]
                right_eyes = [x for x in pupil_predicts if not x.eye_sample.is_left]

                if left_eyes and right_eyes:
                    center_left = tuple(left_eyes[0].landmarks[0].astype(int))
                    center_right = tuple(right_eyes[0].landmarks[0].astype(int))

                    cv2.circle(display_frame, center_left, 2, (0, 0, 255), -1)
                    cv2.circle(display_frame, center_right, 2, (0, 0, 255), -1)
                    for (x, y) in shape[36:48].astype(int):
                        cv2.circle(display_frame, (x, y), 1, (255, 0, 0), -1)

                    norm_right = np.linalg.norm(shape[36] - shape[39])
                    norm_left = np.linalg.norm(shape[42] - shape[45])

                    if norm_right > 0 and norm_left > 0:
                        ldmks_right = (np.vstack([shape[36:42], center_right]) - shape[36]) / norm_right
                        feat_r = ldmks_right.reshape(1, -1)
                        look_x_r = model_x.predict(feat_r)[0]
                        feat_y_r = np.append(feat_r.flatten(), look_x_r).reshape(1, -1)
                        look_y_r = model_y.predict(feat_y_r)[0]

                        ldmks_left = (np.vstack([shape[42:48], center_left]) - shape[42]) / norm_left
                        feat_l = ldmks_left.reshape(1, -1)
                        look_x_l = model_x.predict(feat_l)[0]
                        feat_y_l = np.append(feat_l.flatten(), look_x_l).reshape(1, -1)
                        look_y_l = model_y.predict(feat_y_l)[0]

                        end_r = (
                            int(look_x_r * norm_right * 1.5 + shape[36][0]),
                            int(look_y_r * norm_right + shape[36][1]),
                        )
                        end_l = (
                            int(look_x_l * norm_left * 1.5 + shape[42][0]),
                            int(look_y_l * norm_left + shape[42][1]),
                        )
                        cv2.line(display_frame, center_right, end_r, (0, 255, 0), 2)
                        cv2.line(display_frame, center_left, end_l, (0, 255, 0), 2)

                        avg_raw_x = (float(look_x_r) + float(look_x_l)) / 2.0
                        avg_raw_y = (float(look_y_r) + float(look_y_l)) / 2.0

                        smooth_x = alpha * avg_raw_x + (1 - alpha) * smooth_x
                        smooth_y = alpha * avg_raw_y + (1 - alpha) * smooth_y

                        deviation = math.sqrt(smooth_x**2 + smooth_y**2)
                        angle_deg = deviation * 45

                        SCREEN_X_MIN, SCREEN_X_MAX = -0.5, 0.5
                        SCREEN_Y_MIN, SCREEN_Y_MAX = -0.15, 0.8
                        is_looking_at_screen = (
                            SCREEN_X_MIN < smooth_x < SCREEN_X_MAX
                        ) and (SCREEN_Y_MIN < smooth_y < SCREEN_Y_MAX)

                        if is_looking_at_screen:
                            status_text = "ENGAGED: LOOKING AT BILLBOARD"
                            color = (0, 255, 0)
                        else:
                            error_x = smooth_x
                            error_y = smooth_y - 0.3
                            angle_off = math.sqrt(error_x**2 + error_y**2) * 30
                            status_text = f"NOT LOOKING ({angle_off:.1f} deg)"
                            color = (0, 0, 255)

                        cv2.putText(
                            display_frame,
                            f"Gaze: x={smooth_x:.2f}, y={smooth_y:.2f}, angle={angle_deg:.1f}",
                            (x1, min(h - 10, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )
                        cv2.putText(
                            display_frame,
                            status_text,
                            (x1, min(h - 10, y2 + 40)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )
            except Exception as exc:
                print(f"Gaze prediction error: {exc}")

    cv2.imshow("Combined Gaze + Age/Gender", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_index += 1

grabber.release()
cv2.destroyAllWindows()