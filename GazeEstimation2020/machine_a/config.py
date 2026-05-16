import os


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(PACKAGE_DIR)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, os.pardir))

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
AD_MEDIA_ROOT = os.getenv("AD_MEDIA_ROOT", APP_DIR)
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
AD_WINDOW_WIDTH = int(os.getenv("AD_WINDOW_WIDTH", "1180"))
AD_WINDOW_HEIGHT = int(os.getenv("AD_WINDOW_HEIGHT", "800"))
AD_WINDOW_FULLSCREEN = os.getenv("AD_WINDOW_FULLSCREEN", "0").strip().lower() in {"1", "true", "yes", "on"}

GAZE_MODEL_X_PATH = os.path.join(APP_DIR, "models", "model_x.pkl")
GAZE_MODEL_Y_PATH = os.path.join(APP_DIR, "models", "model_y.pkl")
PUPIL_MODEL_PATH = os.path.join(APP_DIR, "models", "pupilnet_v5.pt")
FACE_LANDMARKER_MODEL_PATH = os.path.join(APP_DIR, "models", "face_landmarker.task")
AGE_GENDER_MODEL_PATH = os.path.join(APP_DIR, "..", "AgeAndGender", "ResNet50_128_phase2_new.keras")
TRACKING_TEST_CSV = os.path.join(APP_DIR, "tracking_test.csv")
SENSOR_DEFAULT_STATE = os.getenv("SENSOR_DEFAULT_STATE", "Light")

IMG_SIZE = 128
MAX_AGE = 116
GENDER_THRESHOLD = 0.5
TRACK_MATCH_IOU_THRESHOLD = 0.3
