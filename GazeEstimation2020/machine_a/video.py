import threading
import time

import cv2
import numpy as np

from .config import AD_WINDOW_HEIGHT, AD_WINDOW_WIDTH, FRAME_BUFFERSIZE, FRAME_HEIGHT, FRAME_STALE_TIMEOUT, FRAME_WIDTH


def create_black_frame(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    return np.zeros((height, width, 3), dtype=np.uint8)


def resize_canvas(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class LatestFrameGrabber:
    def __init__(self, source):
        self.source = source
        self.cap = self._open_capture()
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.latest_frame = None
        self.last_frame_ts = None
        self.failed_reads = 0
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped.is_set():
            if not self.cap.isOpened():
                time.sleep(0.5)
                self.cap.release()
                self.cap = self._open_capture()
                continue

            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.latest_frame = frame
                    self.last_frame_ts = time.time()
                self.failed_reads = 0
            else:
                self.failed_reads += 1
                if self.failed_reads >= 30:
                    self.cap.release()
                    self.failed_reads = 0

            if not ok:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            is_fresh = (
                self.last_frame_ts is not None
                and time.time() - self.last_frame_ts <= FRAME_STALE_TIMEOUT
            )
            return is_fresh, self.latest_frame.copy()

    def _open_capture(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, FRAME_BUFFERSIZE)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 15)
        return cap

    def release(self):
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
