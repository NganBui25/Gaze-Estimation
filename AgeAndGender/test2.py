import cv2 # type: ignore
import numpy as np
import tensorflow as tf  # type: ignore

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Users\ADMIN\Downloads\ResNet50_128_phase2_new.keras"
IMG_SIZE = 128
MAX_AGE = 116
GENDER_THRESHOLD = 0.5

# =========================
# CUSTOM OBJECTS FOR CORAL MODEL
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

# =========================
# LOAD MODEL
# =========================
custom_objects = {
    "BinaryF1Score": BinaryF1Score,
    "CoralLoss": CoralLoss,
    "CoralMAE": CoralMAE,
}

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False
)

print("Loaded model successfully!")
print("Output names:", model.output_names)

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# PREPROCESS
# =========================
def preprocess_face_for_inference(face_bgr, img_size=IMG_SIZE):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (img_size, img_size))
    face_rgb = face_rgb.astype(np.float32)
    face_rgb = np.expand_dims(face_rgb, axis=0)
    return face_rgb

# =========================
# PREDICT 1 FACE
# =========================
def predict_face(model, face_bgr):
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

# =========================
# WEBCAM LOOP
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Không mở được webcam.")

print("Nhấn Q để thoát...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame từ webcam.")
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        pad = int(0.15 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        try:
            gender_label, gender_prob, age_pred = predict_face(model, face_crop)

            text1 = f"{gender_label} ({gender_prob:.2f})"
            text2 = f"Age: {age_pred:.1f}"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, text1, (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, text2, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            cv2.putText(display_frame, "Predict error", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("Prediction error:", e)

    cv2.imshow("Age Gender Prediction", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
