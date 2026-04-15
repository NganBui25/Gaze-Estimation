import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Users\ADMIN\Downloads\ResNet50_128_phase2_new.keras"
IMAGE_PATH = r"C:\path\to\your\image.jpg"
IMG_SIZE = 128


# =========================
# CUSTOM OBJECTS
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


def coral_logits_to_age_np(logits):
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.sum(axis=-1)


# =========================
# FACE DETECTION
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_largest_face(img_bgr, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    return max(faces, key=lambda f: f[2] * f[3])


def crop_face_with_margin(img_bgr, face_box, margin=0.25):
    x, y, w, h = face_box
    h_img, w_img = img_bgr.shape[:2]

    dx = int(w * margin)
    dy = int(h * margin)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(w_img, x + w + dx)
    y2 = min(h_img, y + h + dy)

    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def preprocess_face_for_inference(face_bgr, img_size=IMG_SIZE):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (img_size, img_size))
    face_rgb = face_rgb.astype(np.float32)
    face_rgb = np.expand_dims(face_rgb, axis=0)
    return face_rgb


# =========================
# LOAD MODEL
# =========================
custom_objects = {
    "BinaryF1Score": BinaryF1Score,
    "CoralLoss": CoralLoss,
    "CoralMAE": CoralMAE,
}

model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)


# =========================
# PREDICT
# =========================
def predict_image(model, image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Khong doc duoc anh: {image_path}")

    face_box = detect_largest_face(img_bgr)
    if face_box is None:
        raise ValueError("Khong phat hien duoc khuon mat.")

    face_crop, (x1, y1, x2, y2) = crop_face_with_margin(img_bgr, face_box, margin=0.25)
    input_img = preprocess_face_for_inference(face_crop, IMG_SIZE)

    preds = model.predict(input_img, verbose=0)

    gender_prob = float(preds["gender_output"][0][0])
    predicted_gender = "Female" if gender_prob >= 0.5 else "Male"
    predicted_age = int(np.round(coral_logits_to_age_np(preds["age_output"])[0]))

    img_show = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_show)
    plt.title("Original image + detected face")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    plt.title(f"Gender: {predicted_gender} ({gender_prob:.3f}) | Age: {predicted_age}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Predicted gender :", predicted_gender)
    print("Gender prob      :", round(gender_prob, 4))
    print("Predicted age    :", predicted_age)


predict_image(model, IMAGE_PATH)
