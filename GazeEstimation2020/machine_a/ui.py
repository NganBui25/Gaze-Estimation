import cv2
import numpy as np

from .config import (
    AD_WINDOW_FULLSCREEN,
    AD_WINDOW_HEIGHT,
    AD_WINDOW_NAME,
    AD_WINDOW_WIDTH,
    TRACKING_WINDOW_HEIGHT,
    TRACKING_WINDOW_NAME,
    TRACKING_WINDOW_WIDTH,
)


def create_black_frame(width: int, height: int):
    return np.zeros((height, width, 3), dtype=np.uint8)


def resize_canvas(frame, width: int, height: int):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def setup_windows():
    cv2.namedWindow(TRACKING_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(AD_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TRACKING_WINDOW_NAME, TRACKING_WINDOW_WIDTH, TRACKING_WINDOW_HEIGHT)
    cv2.resizeWindow(AD_WINDOW_NAME, AD_WINDOW_WIDTH, AD_WINDOW_HEIGHT)
    if AD_WINDOW_FULLSCREEN:
        cv2.setWindowProperty(AD_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
