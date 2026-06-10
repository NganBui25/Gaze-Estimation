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
VIDEO_RECONNECT_FAILED_READS = _get_int("VIDEO_RECONNECT_FAILED_READS", 8)
VIDEO_IDLE_SLEEP_SECONDS = float(os.getenv("VIDEO_IDLE_SLEEP_SECONDS", "0.005"))
VIDEO_UDP_FIFO_SIZE = _get_int("VIDEO_UDP_FIFO_SIZE", 1_000_000)
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

GAZE_MODEL_PATH = os.path.join(APP_DIR, "models", "best_model.joblib")
GAZE_MODEL_META_PATH = os.path.join(APP_DIR, "models", "best_model_meta.json")
PUPIL_MODEL_PATH = os.path.join(APP_DIR, "models", "pupilnet_v5.pt")
FACE_LANDMARKER_MODEL_PATH = os.path.join(APP_DIR, "models", "face_landmarker.task")
AGE_GENDER_MODEL_PATH = os.getenv(
    "AGE_GENDER_MODEL_PATH",
    os.path.abspath(
        os.path.join(
            ROOT_DIR,
            "..",
            "TrainModelAgeAndGender",
            "age_gender_range_efficientnetv2s_v1.keras",
        )
    ),
)
TRACKING_TEST_CSV = os.path.join(APP_DIR, "tracking_test.csv")
SENSOR_DEFAULT_STATE = os.getenv("SENSOR_DEFAULT_STATE", "Light")

IMG_SIZE = 256
MAX_AGE = 116
GENDER_THRESHOLD = 0.5
TRACK_MATCH_IOU_THRESHOLD = 0.3
GAZE_STATE_IOU_THRESHOLD = float(os.getenv("GAZE_STATE_IOU_THRESHOLD", "0.3"))
GAZE_STATE_TTL_SECONDS = float(os.getenv("GAZE_STATE_TTL_SECONDS", "1.5"))
GAZE_EMA_ALPHA = float(os.getenv("GAZE_EMA_ALPHA", "0.2"))
GAZE_YAW_MIN = float(os.getenv("GAZE_YAW_MIN", "-20.0"))
GAZE_YAW_MAX = float(os.getenv("GAZE_YAW_MAX", "20.0"))
GAZE_PITCH_MIN = float(os.getenv("GAZE_PITCH_MIN", "-15.0"))
GAZE_PITCH_MAX = float(os.getenv("GAZE_PITCH_MAX", "25.0"))
GAZE_MAX_PITCH_DISAGREEMENT_DEG = float(os.getenv("GAZE_MAX_PITCH_DISAGREEMENT_DEG", "20.0"))
GAZE_MAX_ABS_YAW_DEG = float(os.getenv("GAZE_MAX_ABS_YAW_DEG", "60.0"))
GAZE_MAX_ABS_PITCH_DEG = float(os.getenv("GAZE_MAX_ABS_PITCH_DEG", "45.0"))
GAZE_MIN_EYE_WIDTH_PX = float(os.getenv("GAZE_MIN_EYE_WIDTH_PX", "8.0"))

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
