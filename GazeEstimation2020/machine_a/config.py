import os


def _get_int(name, default):
    return int(os.getenv(name, str(default)))


PERFORMANCE_MODE = os.getenv("PERFORMANCE_MODE", "balanced").strip().lower()

if PERFORMANCE_MODE == "light":
    CAPTURE_WIDTH = _get_int("CAPTURE_WIDTH", 640)
    CAPTURE_HEIGHT = _get_int("CAPTURE_HEIGHT", 480)
    PROCESS_EVERY_N_FRAMES = _get_int("PROCESS_EVERY_N_FRAMES", 1)
    AGE_GENDER_EVERY_N_FRAMES = _get_int("AGE_GENDER_EVERY_N_FRAMES", 10)
elif PERFORMANCE_MODE == "aggressive":
    CAPTURE_WIDTH = _get_int("CAPTURE_WIDTH", 426)
    CAPTURE_HEIGHT = _get_int("CAPTURE_HEIGHT", 240)
    PROCESS_EVERY_N_FRAMES = _get_int("PROCESS_EVERY_N_FRAMES", 3)
    AGE_GENDER_EVERY_N_FRAMES = _get_int("AGE_GENDER_EVERY_N_FRAMES", 30)
else:
    PERFORMANCE_MODE = "balanced"
    CAPTURE_WIDTH = _get_int("CAPTURE_WIDTH", 640)
    CAPTURE_HEIGHT = _get_int("CAPTURE_HEIGHT", 360)
    PROCESS_EVERY_N_FRAMES = _get_int("PROCESS_EVERY_N_FRAMES", 2)
    AGE_GENDER_EVERY_N_FRAMES = _get_int("AGE_GENDER_EVERY_N_FRAMES", 20)


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PACKAGE_DIR)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, os.pardir))

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "udp://0.0.0.0:5000")
SERVER_IP = os.getenv("MAY_B_IP", os.getenv("SERVER_IP", "127.0.0.1"))
SERVER_PORT = int(os.getenv("MAY_B_PORT", os.getenv("SERVER_PORT", "5000")))
FRAME_WIDTH = CAPTURE_WIDTH
FRAME_HEIGHT = CAPTURE_HEIGHT
FRAME_BUFFERSIZE = 1
FRAME_STALE_TIMEOUT = float(os.getenv("FRAME_STALE_TIMEOUT", "0.75"))
VIDEO_RECONNECT_FAILED_READS = _get_int("VIDEO_RECONNECT_FAILED_READS", 3)
VIDEO_IDLE_SLEEP_SECONDS = float(os.getenv("VIDEO_IDLE_SLEEP_SECONDS", "0.005"))
# FFmpeg UDP FIFO is measured in 188-byte packets. This default is about 25 MB.
VIDEO_UDP_FIFO_SIZE = _get_int("VIDEO_UDP_FIFO_SIZE", 131_072)
VIDEO_UDP_BUFFER_SIZE = _get_int("VIDEO_UDP_BUFFER_SIZE", 4_194_304)
VIDEO_UDP_TIMEOUT_US = _get_int("VIDEO_UDP_TIMEOUT_US", 2_000_000)
VISION_PROCESS_WIDTH = _get_int("VISION_PROCESS_WIDTH", 640)
DEMOGRAPHIC_REFRESH_SECONDS = float(os.getenv("DEMOGRAPHIC_REFRESH_SECONDS", "1.0"))
DEMOGRAPHIC_CACHE_TTL_SECONDS = float(os.getenv("DEMOGRAPHIC_CACHE_TTL_SECONDS", "3.0"))
AUDIENCE_WINDOW_SECONDS = float(os.getenv("AUDIENCE_WINDOW_SECONDS", "4.0"))
NO_VIEWER_SIGNAL_DELAY_SECONDS = float(os.getenv("NO_VIEWER_SIGNAL_DELAY_SECONDS", "5.0"))
AD_MISSING_FRAME_TIMEOUT = float(os.getenv("AD_MISSING_FRAME_TIMEOUT", "1.5"))
SENSOR_POLL_INTERVAL = float(os.getenv("SENSOR_POLL_INTERVAL", "0.2"))
SENSOR_SERIAL_PORT = os.getenv("SENSOR_SERIAL_PORT")
SENSOR_SOCKET_HOST = os.getenv("SENSOR_SOCKET_HOST")
SENSOR_SOCKET_PORT = int(os.getenv("SENSOR_SOCKET_PORT", "5001"))
SENSOR_BAUD_RATE = int(os.getenv("SENSOR_BAUD_RATE", "115200"))
AD_MEDIA_ROOT = os.getenv("AD_MEDIA_ROOT", os.path.join(APP_DIR, "ads"))
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
TRACKING_WINDOW_NAME = os.getenv("TRACKING_WINDOW_NAME", "Tracking View")
AD_WINDOW_NAME = os.getenv("AD_WINDOW_NAME", "Ad Display")
TRACKING_WINDOW_WIDTH = int(os.getenv("TRACKING_WINDOW_WIDTH", "640"))
TRACKING_WINDOW_HEIGHT = int(os.getenv("TRACKING_WINDOW_HEIGHT", "480"))
TRACKING_WINDOW_X = int(os.getenv("TRACKING_WINDOW_X", "20"))
TRACKING_WINDOW_Y = int(os.getenv("TRACKING_WINDOW_Y", "20"))
AD_WINDOW_WIDTH = int(os.getenv("AD_WINDOW_WIDTH", "1180"))
AD_WINDOW_HEIGHT = int(os.getenv("AD_WINDOW_HEIGHT", "800"))
AD_WINDOW_X = int(os.getenv("AD_WINDOW_X", "700"))
AD_WINDOW_Y = int(os.getenv("AD_WINDOW_Y", "20"))
AD_WINDOW_FULLSCREEN = os.getenv("AD_WINDOW_FULLSCREEN", "0").strip().lower() in {"1", "true", "yes", "on"}

# best_model.joblib: Pipeline(StandardScaler + MLP) dự đoán ĐỒNG THỜI (yaw, pitch) theo độ
# từ vector đặc trưng 14 chiều (7 điểm mắt). Thay cho model_x.pkl + model_y.pkl cũ.
GAZE_MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.joblib")
PUPIL_MODEL_PATH = os.path.join(APP_DIR, "models", "pupilnet_v5.pt")

# Vùng "đang nhìn màn hình" tính theo ĐỘ (hiệu chỉnh bằng cách nhìn 4 góc màn hình).
GAZE_YAW_MIN = float(os.getenv("GAZE_YAW_MIN", "-20.0"))
GAZE_YAW_MAX = float(os.getenv("GAZE_YAW_MAX", "20.0"))
GAZE_PITCH_MIN = float(os.getenv("GAZE_PITCH_MIN", "-15.0"))
GAZE_PITCH_MAX = float(os.getenv("GAZE_PITCH_MAX", "25.0"))
# Hệ số làm mượt EMA cho (yaw, pitch) — áp dụng THEO TỪNG khuôn mặt.
GAZE_SMOOTH_ALPHA = float(os.getenv("GAZE_SMOOTH_ALPHA", "0.3"))
# Trạng thái làm mượt của một khuôn mặt bị xóa nếu không thấy lại trong khoảng này (giây).
GAZE_STATE_TTL_SECONDS = float(os.getenv("GAZE_STATE_TTL_SECONDS", "1.0"))

# Bù góc quay đầu (head pose): góc nhìn thực = góc mắt-trong-đầu + GAZE_HEAD_WEIGHT × góc đầu.
# Đặt 0 để tắt (quay về hành vi chỉ dùng góc mắt).
GAZE_HEAD_WEIGHT = float(os.getenv("GAZE_HEAD_WEIGHT", "1.0"))
# Bù góc phương vị: người đứng lệch khỏi trục camera muốn nhìn màn hình (đặt cạnh camera)
# phải liếc về phía camera; cộng góc phương vị của khuôn mặt để quy "nhìn màn hình" về ~0 độ.
GAZE_BEARING_CORRECTION = os.getenv("GAZE_BEARING_CORRECTION", "1").strip().lower() in {"1", "true", "yes", "on"}
# Góc mở ngang của camera (độ) — dùng để đổi vị trí khuôn mặt trong khung hình sang góc phương vị.
CAMERA_HFOV_DEG = float(os.getenv("CAMERA_HFOV_DEG", "60.0"))
FACE_LANDMARKER_MODEL_PATH = os.path.join(APP_DIR, "models", "face_landmarker.task")
# Mặc định nằm TRONG repo (GazeEstimation2020/models/) để clone từ git là chạy được;
# đường dẫn cũ trỏ ra ../TrainModelAgeAndGender bên ngoài repo nên sau khi push sẽ gãy.
# Cần đặt file .keras vào models/ hoặc trỏ qua biến môi trường AGE_GENDER_MODEL_PATH.
AGE_GENDER_MODEL_PATH = os.getenv(
    "AGE_GENDER_MODEL_PATH",
    os.path.join(APP_DIR, "models", "age_gender_range_efficientnetv2s_v1.keras"),
)
TRACKING_TEST_CSV = os.path.join(APP_DIR, "tracking_test.csv")
SENSOR_DEFAULT_STATE = os.getenv("SENSOR_DEFAULT_STATE", "Light")

IMG_SIZE = 256
MAX_AGE = 116
GENDER_THRESHOLD = 0.5
TRACK_MATCH_IOU_THRESHOLD = 0.3

AGE_RANGES = [
    (0, 17),
    (18, 25),
    (26, 35),
    (36, 45),
    (46, 54),
    (55, 65),
    (66, MAX_AGE),
]
AGE_RANGE_NAMES = [
    "0-17",
    "18-25",
    "26-35",
    "36-45",
    "46-54",
    "55-65",
    "66+",
]

# These IDs match the audience_segments rows currently stored in Machine B.
AUDIENCE_SEGMENT_IDS = {
    "male": {
        "0-17": 1,
        "18-25": 2,
        "26-35": 3,
        "36-45": 4,
        "46-54": 5,
        "55-65": 6,
        "66+": 7,
    },
    "female": {
        "0-17": 8,
        "18-25": 9,
        "26-35": 10,
        "36-45": 11,
        "46-54": 12,
        "55-65": 13,
        "66+": 14,
    },
}
