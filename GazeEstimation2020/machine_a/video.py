import threading
import time

import cv2
import numpy as np

from .config import AD_WINDOW_HEIGHT, AD_WINDOW_WIDTH, FRAME_BUFFERSIZE, FRAME_HEIGHT, FRAME_WIDTH


def create_black_frame(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    return np.zeros((height, width, 3), dtype=np.uint8)


def resize_canvas(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


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
