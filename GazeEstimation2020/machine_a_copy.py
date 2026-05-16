import math
import os
import csv
import socket
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import cv2  # type: ignore
import joblib
import mediapipe as mp
import numpy as np
import requests
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

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "udp://0.0.0.0:5000")
SERVER_IP = os.getenv("MAY_B_IP", os.getenv("SERVER_IP", "127.0.0.1"))
SERVER_PORT = int(os.getenv("MAY_B_PORT", os.getenv("SERVER_PORT", "5000")))
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_BUFFERSIZE = 1
AGE_GENDER_EVERY_N_FRAMES = 10
AUDIENCE_WINDOW_SECONDS = float(os.getenv("AUDIENCE_WINDOW_SECONDS", "4.0"))
AD_MISSING_FRAME_TIMEOUT = float(os.getenv("AD_MISSING_FRAME_TIMEOUT", "1.5"))
SENSOR_POLL_INTERVAL = float(os.getenv("SENSOR_POLL_INTERVAL", "0.2"))
SENSOR_SERIAL_PORT = os.getenv("SENSOR_SERIAL_PORT")
SENSOR_SOCKET_HOST = os.getenv("SENSOR_SOCKET_HOST")
SENSOR_SOCKET_PORT = int(os.getenv("SENSOR_SOCKET_PORT", "5001"))
SENSOR_BAUD_RATE = int(os.getenv("SENSOR_BAUD_RATE", "115200"))
AD_MEDIA_ROOT = os.getenv("AD_MEDIA_ROOT", BASE_DIR)
SELECT_AD_URL = os.getenv(
    "SELECT_AD_URL",
    f"http://{SERVER_IP}:{SERVER_PORT}/api/advertisements/select",
)
REPORT_AD_URL = os.getenv(
    "REPORT_AD_URL",
    f"http://{SERVER_IP}:{SERVER_PORT}/api/ad-play-logs/report",
)
REQUEST_AD_URLS = [SELECT_AD_URL]
FINAL_STATS_URLS = [REPORT_AD_URL]
WINDOW_NAME = os.getenv("BILLBOARD_WINDOW_NAME", "Smart Billboard")

GAZE_MODEL_X_PATH = os.path.join(BASE_DIR, "models", "model_x.pkl")
GAZE_MODEL_Y_PATH = os.path.join(BASE_DIR, "models", "model_y.pkl")
PUPIL_MODEL_PATH = os.path.join(BASE_DIR, "models", "pupilnet_v5.pt")
FACE_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
AGE_GENDER_MODEL_PATH = os.path.join(ROOT_DIR, "AgeAndGender", "ResNet50_128_phase2_new.keras")
TRACKING_TEST_CSV = os.path.join(BASE_DIR, "tracking_test.csv")
SENSOR_DEFAULT_STATE = os.getenv("SENSOR_DEFAULT_STATE", "Light")

IMG_SIZE = 128
MAX_AGE = 116
GENDER_THRESHOLD = 0.5
TRACK_MATCH_IOU_THRESHOLD = 0.3


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


def post_json_async(url, payload):
    def _worker():
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as exc:
            print(f"HTTP POST failed for {url}: {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def post_json_async_many(urls, payload):
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        post_json_async(url, payload)


def append_tracking_report(report):
    file_exists = os.path.exists(TRACKING_TEST_CSV)
    with open(TRACKING_TEST_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["gender", "age", "duration", "type"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(report)


def resolve_age_gender_for_report(last_age_gender, session_gender, session_age):
    if last_age_gender is not None:
        gender_label, _, age_pred = last_age_gender
        return gender_label, age_pred
    return session_gender, session_age


def finalize_attention_session(last_age_gender, session_gender, session_age, start_time, last_seen_time):
    end_time = last_seen_time
    duration = round(end_time - start_time, 3)
    final_gender, final_age = resolve_age_gender_for_report(
        last_age_gender,
        session_gender,
        session_age,
    )
    final_report = {
        "gender": final_gender,
        "age": final_age,
        "duration": duration,
        "type": "final",
    }
    post_json_async_many(FINAL_STATS_URLS, final_report)
    append_tracking_report(final_report)
    print("Tracking final report:", final_report)
    return final_report


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


def iso_now(ts=None):
    if ts is None:
        return datetime.now().isoformat(timespec="seconds")
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def normalize_sensor_state(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip().lower()
    if not value:
        return None
    if value in {"light", "bright", "on", "1", "true", "open"}:
        return "Light"
    if value in {"dark", "off", "0", "false", "closed"}:
        return "Dark"
    if "light" in value:
        return "Light"
    if "dark" in value:
        return "Dark"
    return None


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter_area / max(area_a + area_b - inter_area, 1.0)


def majority_vote(values, default="Unknown"):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return default
    return Counter(filtered).most_common(1)[0][0]


def normalize_gender_for_api(gender):
    if gender is None:
        return "unknown"
    normalized_gender = str(gender).strip().lower()
    if normalized_gender in {"male", "female", "unknown"}:
        return normalized_gender
    return "unknown"


def mean_or_none(values):
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def resolve_media_path(media_filename):
    if not media_filename:
        return None
    candidate_paths = []
    if os.path.isabs(media_filename):
        candidate_paths.append(media_filename)
    candidate_paths.extend(
        [
            os.path.join(AD_MEDIA_ROOT, media_filename),
            os.path.join(BASE_DIR, media_filename),
            os.path.join(ROOT_DIR, media_filename),
            os.path.join(os.getcwd(), media_filename),
        ]
    )
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return candidate_paths[0] if candidate_paths else media_filename


def extract_ad_selection(payload):
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected advertisement payload: {payload!r}")

    nested_payload = payload.get("data")
    if isinstance(nested_payload, dict):
        payload = nested_payload

    advertisement_payload = payload.get("advertisement")
    if isinstance(advertisement_payload, dict):
        payload = advertisement_payload

    ad_id = payload.get("ad_id") or payload.get("id")
    media_filename = (
        payload.get("media_filename")
        or payload.get("mediaFileName")
        or payload.get("filename")
        or payload.get("media")
    )
    duration_seconds = payload.get("duration_seconds")
    if duration_seconds is None:
        duration_seconds = payload.get("duration") or payload.get("seconds")

    if ad_id is None or media_filename is None or duration_seconds is None:
        raise ValueError(f"Missing ad fields in response: {payload!r}")

    return {
        "ad_id": ad_id,
        "media_filename": media_filename,
        "duration_seconds": float(duration_seconds),
    }


@dataclass
class ViewerTrack:
    track_id: int
    bbox: tuple
    first_seen_ts: float
    last_seen_ts: float
    age_samples: list = field(default_factory=list)
    gender_samples: list = field(default_factory=list)
    looking_samples: list = field(default_factory=list)
    looking_duration_total: float = 0.0
    frames_seen: int = 0

    def update(self, bbox, age_value, gender_value, looking_value, timestamp):
        previous_seen_ts = self.last_seen_ts
        self.bbox = bbox
        self.last_seen_ts = timestamp
        self.frames_seen += 1
        if age_value is not None:
            self.age_samples.append(float(age_value))
        if gender_value is not None:
            self.gender_samples.append(gender_value)
        if looking_value is not None:
            self.looking_samples.append(bool(looking_value))
            if looking_value and previous_seen_ts is not None:
                self.looking_duration_total += max(0.0, timestamp - previous_seen_ts)

    def summary(self, session_end_ts=None):
        end_ts = self.last_seen_ts if session_end_ts is None else session_end_ts
        estimated_age = mean_or_none(self.age_samples)
        gender = majority_vote(self.gender_samples)
        watch_duration = self.looking_duration_total
        if (
            session_end_ts is not None
            and self.looking_samples
            and self.looking_samples[-1]
            and end_ts > self.last_seen_ts
        ):
            watch_duration += max(0.0, end_ts - self.last_seen_ts)
        return {
            "viewer_id": self.track_id,
            "estimated_age": None if estimated_age is None else int(round(estimated_age)),
            "gender": str(gender).strip().lower() if gender is not None else "unknown",
            "start_time": iso_now(self.first_seen_ts),
            "end_time": iso_now(end_ts),
            "watch_duration": round(watch_duration, 3),
        }


class ViewerTrackManager:
    def __init__(self, iou_threshold=TRACK_MATCH_IOU_THRESHOLD):
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_track_id = 1

    def reset(self):
        self.tracks = []
        self.next_track_id = 1

    def update(self, detections, timestamp):
        unmatched_track_indexes = set(range(len(self.tracks)))

        for detection in detections:
            best_index = None
            best_score = 0.0
            for track_index in unmatched_track_indexes:
                score = bbox_iou(detection["bbox"], self.tracks[track_index].bbox)
                if score > best_score:
                    best_score = score
                    best_index = track_index

            if best_index is not None and best_score >= self.iou_threshold:
                self.tracks[best_index].update(
                    detection["bbox"],
                    detection.get("estimated_age"),
                    detection.get("gender"),
                    detection.get("looking"),
                    timestamp,
                )
                unmatched_track_indexes.discard(best_index)
            else:
                new_track = ViewerTrack(
                    track_id=self.next_track_id,
                    bbox=detection["bbox"],
                    first_seen_ts=timestamp,
                    last_seen_ts=timestamp,
                )
                new_track.update(
                    detection["bbox"],
                    detection.get("estimated_age"),
                    detection.get("gender"),
                    detection.get("looking"),
                    timestamp,
                )
                self.tracks.append(new_track)
                self.next_track_id += 1

    def active_summaries(self, session_end_ts=None):
        return [
            track.summary(session_end_ts=session_end_ts)
            for track in self.tracks
            if track.frames_seen > 0
        ]


class SensorMonitor:
    def __init__(self, default_state=SENSOR_DEFAULT_STATE):
        self._state = normalize_sensor_state(default_state) or "Light"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_state(self):
        with self._lock:
            return self._state

    def _set_state(self, value):
        normalized_value = normalize_sensor_state(value)
        if normalized_value is None:
            return
        with self._lock:
            self._state = normalized_value

    def _run(self):
        if SENSOR_SERIAL_PORT:
            if self._run_serial_loop():
                return
        if SENSOR_SOCKET_HOST:
            if self._run_socket_loop():
                return
        while not self._stop_event.is_set():
            time.sleep(SENSOR_POLL_INTERVAL)

    def _run_serial_loop(self):
        try:
            import serial  # type: ignore
        except Exception as exc:
            print(f"Serial sensor unavailable: {exc}")
            return False

        while not self._stop_event.is_set():
            try:
                with serial.Serial(SENSOR_SERIAL_PORT, SENSOR_BAUD_RATE, timeout=1) as ser:
                    print(f"Connected to sensor serial port {SENSOR_SERIAL_PORT}")
                    while not self._stop_event.is_set():
                        raw_value = ser.readline().decode("utf-8", errors="ignore").strip()
                        self._set_state(raw_value)
            except Exception as exc:
                print(f"Serial sensor read failed: {exc}")
                time.sleep(1.0)
        return True

    def _run_socket_loop(self):
        while not self._stop_event.is_set():
            try:
                with socket.create_connection((SENSOR_SOCKET_HOST, SENSOR_SOCKET_PORT), timeout=5.0) as sock:
                    print(f"Connected to sensor socket {SENSOR_SOCKET_HOST}:{SENSOR_SOCKET_PORT}")
                    sock.settimeout(1.0)
                    buffer = ""
                    while not self._stop_event.is_set():
                        try:
                            data = sock.recv(1024)
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        buffer += data.decode("utf-8", errors="ignore")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            self._set_state(line)
            except Exception as exc:
                print(f"Socket sensor read failed: {exc}")
                time.sleep(1.0)
        return True


class AdSelectionRequest:
    def __init__(self, url, payload, generation_id):
        self.url = url
        self.payload = payload
        self.generation_id = generation_id
        self.event = threading.Event()
        self.response = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            response = requests.post(self.url, json=self.payload, timeout=8)
            response.raise_for_status()
            self.response = response.json()
        except Exception as exc:
            self.error = exc
        finally:
            self.event.set()


def build_selection_payload(tracks, window_start_ts, window_end_ts):
    viewer_summaries = [track.summary(session_end_ts=window_end_ts) for track in tracks if track.frames_seen > 0]
    viewer_count = len(viewer_summaries)
    avg_age = mean_or_none([viewer["estimated_age"] for viewer in viewer_summaries])
    majority_gender = normalize_gender_for_api(
        majority_vote([viewer["gender"] for viewer in viewer_summaries], default="unknown")
    )

    return {
        "viewer_count": viewer_count,
        "timestamp": iso_now(window_end_ts),
        "avg_age": 0 if avg_age is None else int(round(avg_age)),
        "majority_gender": majority_gender,
        "window_start": iso_now(window_start_ts),
        "window_end": iso_now(window_end_ts),
    }


def build_report_payload(ad_id, tracks, start_ts, end_ts):
    viewers = [track.summary(session_end_ts=end_ts) for track in tracks if track.frames_seen > 0]
    return {
        "ad_id": ad_id,
        "start_time": iso_now(start_ts),
        "end_time": iso_now(end_ts),
        "total_viewers": len(viewers),
        "viewers": viewers,
    }


def create_black_frame(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    return np.zeros((height, width, 3), dtype=np.uint8)


class LatestFrameGrabber:
    def __init__(self, source):
        self.source = source
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
                time.sleep(0.5)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source)
                continue

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


def collect_viewer_detections(frame, frame_index, gaze_state, annotate=True):
    display_frame = frame.copy()
    detections = []
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not results.face_landmarks:
        return display_frame, detections

    for face_landmarks in results.face_landmarks:
        full_mp_shape = get_mediapipe_landmarks(face_landmarks, w, h)
        shape = map_to_dlib_style(full_mp_shape)
        x1, y1, x2, y2 = build_face_bbox(full_mp_shape, w, h)

        face_crop = frame[y1:y2, x1:x2]
        estimated_age = None
        gender_label = None

        if face_crop.size > 0 and frame_index % AGE_GENDER_EVERY_N_FRAMES == 0:
            try:
                gender_label, _, estimated_age = predict_age_gender(age_gender_model, face_crop)
            except Exception as exc:
                print("Age/Gender prediction error:", exc)

        is_looking_at_screen = False
        try:
            eye_samples = segment_eyes(gray_frame, shape)
            pupil_predicts = predict_pupil(pupil_model, device, eye_samples)

            left_eyes = [item for item in pupil_predicts if item.eye_sample.is_left]
            right_eyes = [item for item in pupil_predicts if not item.eye_sample.is_left]

            if left_eyes and right_eyes:
                center_left = tuple(left_eyes[0].landmarks[0].astype(int))
                center_right = tuple(right_eyes[0].landmarks[0].astype(int))

                norm_right = np.linalg.norm(shape[36] - shape[39])
                norm_left = np.linalg.norm(shape[42] - shape[45])

                if norm_right > 0 and norm_left > 0:
                    look_x_r = model_x.predict(
                        ((np.vstack([shape[36:42], center_right]) - shape[36]) / norm_right).reshape(1, -1)
                    )[0]
                    look_y_r = model_y.predict(
                        np.append(
                            ((np.vstack([shape[36:42], center_right]) - shape[36]) / norm_right).reshape(1, -1).flatten(),
                            look_x_r,
                        ).reshape(1, -1)
                    )[0]

                    look_x_l = model_x.predict(
                        ((np.vstack([shape[42:48], center_left]) - shape[42]) / norm_left).reshape(1, -1)
                    )[0]
                    look_y_l = model_y.predict(
                        np.append(
                            ((np.vstack([shape[42:48], center_left]) - shape[42]) / norm_left).reshape(1, -1).flatten(),
                            look_x_l,
                        ).reshape(1, -1)
                    )[0]

                    avg_raw_x = (float(look_x_r) + float(look_x_l)) / 2.0
                    avg_raw_y = (float(look_y_r) + float(look_y_l)) / 2.0

                    gaze_state["smooth_x"] = 0.2 * avg_raw_x + 0.8 * gaze_state["smooth_x"]
                    gaze_state["smooth_y"] = 0.2 * avg_raw_y + 0.8 * gaze_state["smooth_y"]

                    screen_x_min, screen_x_max = -0.5, 0.5
                    screen_y_min, screen_y_max = -0.15, 0.8
                    is_looking_at_screen = (
                        screen_x_min < gaze_state["smooth_x"] < screen_x_max
                    ) and (screen_y_min < gaze_state["smooth_y"] < screen_y_max)

                if annotate:
                    cv2.circle(display_frame, center_left, 2, (0, 0, 255), -1)
                    cv2.circle(display_frame, center_right, 2, (0, 0, 255), -1)
        except Exception as exc:
            print(f"Gaze prediction error: {exc}")

        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "estimated_age": estimated_age,
                "gender": gender_label,
                "looking": is_looking_at_screen,
            }
        )

        if annotate:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            if gender_label is not None and estimated_age is not None:
                cv2.putText(
                    display_frame,
                    f"{gender_label} | Age: {estimated_age:.1f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
            status_text = "Looking" if is_looking_at_screen else "Not looking"
            status_color = (0, 255, 0) if is_looking_at_screen else (0, 0, 255)
            cv2.putText(
                display_frame,
                status_text,
                (x1, min(h - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
            )

    return display_frame, detections


def finalize_ad_session(ad_info, ad_manager, ad_start_ts, ad_end_ts):
    report_payload = build_report_payload(
        ad_info["ad_id"],
        ad_manager.tracks,
        ad_start_ts,
        ad_end_ts,
    )
    post_json_async(REPORT_AD_URL, report_payload)
    print("Ad report queued:", report_payload)


def main():
    sensor_monitor = SensorMonitor().start()
    grabber = LatestFrameGrabber(VIDEO_SOURCE).start()

    deadline = time.time() + 10.0
    while time.time() < deadline:
        ret, _ = grabber.read()
        if ret:
            break
        time.sleep(0.2)
    else:
        sensor_monitor.stop()
        grabber.release()
        raise RuntimeError(
            f"Không nhận được frame từ {VIDEO_SOURCE} trong 10 giây. "
            "Hãy kiểm tra IP Raspberry Pi, port UDP, hoặc biến môi trường VIDEO_SOURCE."
        )

    print("Nhấn Q để thoát...")

    frame_index = 0
    gaze_state = {"smooth_x": 0.0, "smooth_y": 0.0}
    selection_manager = ViewerTrackManager()
    ad_manager = ViewerTrackManager()

    state = "idle"
    selection_start_ts = None
    selection_request = None
    selection_generation_id = 0
    selected_ad = None
    selected_ad_path = None
    ad_capture = None
    ad_start_ts = None
    last_ad_frame_ts = None
    last_ad_display_frame = None

    try:
        while True:
            sensor_state = sensor_monitor.get_state()
            ret, frame = grabber.read()
            now_ts = time.time()

            if state == "ad":
                display_frame = (
                    last_ad_display_frame.copy()
                    if last_ad_display_frame is not None
                    else create_black_frame()
                )

                if ret:
                    _, detections = collect_viewer_detections(frame, frame_index, gaze_state, annotate=False)
                    ad_manager.update(detections, now_ts)

                if ad_capture is not None and (last_ad_frame_ts is None or now_ts - last_ad_frame_ts >= 0.03):
                    ok, ad_frame = ad_capture.read()
                    if ok:
                        last_ad_display_frame = cv2.resize(ad_frame, (FRAME_WIDTH, FRAME_HEIGHT))
                        display_frame = last_ad_display_frame.copy()
                        last_ad_frame_ts = now_ts
                    else:
                        if selected_ad_path is not None:
                            ad_capture.release()
                            ad_capture = cv2.VideoCapture(selected_ad_path)
                            ok, ad_frame = ad_capture.read()
                            if ok:
                                last_ad_display_frame = cv2.resize(ad_frame, (FRAME_WIDTH, FRAME_HEIGHT))
                                display_frame = last_ad_display_frame.copy()
                                last_ad_frame_ts = now_ts

                if selected_ad is not None and ad_start_ts is not None:
                    elapsed = now_ts - ad_start_ts
                    if elapsed >= selected_ad["duration_seconds"]:
                        finalize_ad_session(selected_ad, ad_manager, ad_start_ts, now_ts)
                        state = "idle"
                        selection_start_ts = None
                        selected_ad = None
                        selected_ad_path = None
                        ad_start_ts = None
                        last_ad_frame_ts = None
                        last_ad_display_frame = None
                        ad_manager.reset()
                        if ad_capture is not None:
                            ad_capture.release()
                            ad_capture = None

                cv2.imshow(WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_index += 1
                continue

            if sensor_state == "Dark":
                display_frame = create_black_frame()
                cv2.putText(
                    display_frame,
                    "Display off - sensor dark",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_index += 1
                continue

            if not ret:
                display_frame = create_black_frame()
                cv2.imshow(WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_index += 1
                time.sleep(0.02)
                continue

            display_frame, detections = collect_viewer_detections(frame, frame_index, gaze_state, annotate=True)
            has_viewers = len(detections) > 0

            if not has_viewers:
                selection_generation_id += 1
                selection_start_ts = None
                selection_manager.reset()
                if selection_request is not None and not selection_request.event.is_set():
                    selection_request = None

            if has_viewers and selection_start_ts is None:
                selection_start_ts = now_ts
                selection_manager.reset()

            if has_viewers and selection_request is None:
                selection_manager.update(detections, now_ts)

                elapsed = now_ts - selection_start_ts
                if elapsed >= AUDIENCE_WINDOW_SECONDS:
                    selection_payload = build_selection_payload(
                        selection_manager.tracks,
                        selection_start_ts,
                        now_ts,
                    )
                    if selection_payload["viewer_count"] > 0:
                        selection_request = AdSelectionRequest(
                            SELECT_AD_URL,
                            selection_payload,
                            selection_generation_id,
                        ).start()
                        cv2.putText(
                            display_frame,
                            "Selecting next ad...",
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )
                    else:
                        selection_start_ts = now_ts
                        selection_manager.reset()

            elif selection_request is not None and selection_request.event.is_set():
                if selection_request.generation_id != selection_generation_id:
                    selection_request = None
                    selection_start_ts = None
                    selection_manager.reset()
                elif not has_viewers:
                    selection_request = None
                    selection_start_ts = None
                    selection_manager.reset()
                elif selection_request.error is not None:
                    print(f"Ad selection failed: {selection_request.error}")
                    selection_request = None
                    selection_start_ts = now_ts
                    selection_manager.reset()
                else:
                    try:
                        selected_ad = extract_ad_selection(selection_request.response)
                        selected_ad_path = resolve_media_path(selected_ad["media_filename"])
                        ad_capture = cv2.VideoCapture(selected_ad_path)
                        if not ad_capture.isOpened():
                            raise RuntimeError(f"Cannot open ad media: {selected_ad_path}")
                        ad_manager.reset()
                        state = "ad"
                        ad_start_ts = now_ts
                        last_ad_frame_ts = None
                        last_ad_display_frame = None
                        selection_manager.reset()
                        print(f"Selected ad: {selected_ad}")
                    except Exception as exc:
                        print(f"Invalid ad response: {exc}")
                        selected_ad = None
                        selected_ad_path = None
                        last_ad_display_frame = None
                        selection_request = None
                        selection_start_ts = now_ts
                        selection_manager.reset()
                selection_request = None

            remaining = 0.0 if selection_start_ts is None else max(0.0, AUDIENCE_WINDOW_SECONDS - (now_ts - selection_start_ts))
            cv2.putText(
                display_frame,
                f"Sensor: {sensor_state} | Sampling: {remaining:.1f}s",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1

    finally:
        if selected_ad is not None and ad_start_ts is not None:
            finalize_ad_session(selected_ad, ad_manager, ad_start_ts, time.time())
        if ad_capture is not None:
            ad_capture.release()
        grabber.release()
        sensor_monitor.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()