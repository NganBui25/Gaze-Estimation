import os

import joblib
import tensorflow as tf  # type: ignore
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from models.PupilNet import PupilNet_v2

from .config import (
    AGE_GENDER_MODEL_PATH,
    FACE_LANDMARKER_MODEL_PATH,
    GAZE_MODEL_X_PATH,
    GAZE_MODEL_Y_PATH,
    PUPIL_MODEL_PATH,
)


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


def load_models():
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
    return device, model_x, model_y, pupil_model, age_gender_model, face_landmarker
