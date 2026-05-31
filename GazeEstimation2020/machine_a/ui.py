import cv2
import numpy as np

try:
    import ctypes
    from ctypes import wintypes
except Exception:  # pragma: no cover - best effort fallback for non-Windows runtimes
    ctypes = None
    wintypes = None

from .config import (
    AD_WINDOW_FULLSCREEN,
    AD_WINDOW_HEIGHT,
    AD_WINDOW_NAME,
    AD_WINDOW_X,
    AD_WINDOW_Y,
    AD_WINDOW_WIDTH,
    TRACKING_WINDOW_HEIGHT,
    TRACKING_WINDOW_NAME,
    TRACKING_WINDOW_X,
    TRACKING_WINDOW_Y,
    TRACKING_WINDOW_WIDTH,
)


def create_black_frame(width: int, height: int):
    return np.zeros((height, width, 3), dtype=np.uint8)


def resize_canvas(frame, width: int, height: int):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _get_windows_monitors():
    if ctypes is None or wintypes is None or not hasattr(ctypes, "windll"):
        return []

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    monitors = []
    user32 = ctypes.windll.user32

    monitor_enum_proc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    def _enum_monitor(hmonitor, _hdc, _lprc, _lparam):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            rect = info.rcMonitor
            monitors.append(
                {
                    "left": int(rect.left),
                    "top": int(rect.top),
                    "right": int(rect.right),
                    "bottom": int(rect.bottom),
                    "width": int(rect.right - rect.left),
                    "height": int(rect.bottom - rect.top),
                    "primary": bool(info.dwFlags & 1),
                }
            )
        return True

    callback = monitor_enum_proc(_enum_monitor)
    try:
        user32.EnumDisplayMonitors(0, 0, callback, 0)
    except Exception:
        return []
    return monitors


def _pick_monitor_layout():
    monitors = _get_windows_monitors()
    if len(monitors) < 2:
        return None

    primary_monitor = next((monitor for monitor in monitors if monitor["primary"]), monitors[0])
    secondary_monitor = next((monitor for monitor in monitors if monitor is not primary_monitor), None)
    if secondary_monitor is None:
        return None

    tracking_width = min(TRACKING_WINDOW_WIDTH, max(320, primary_monitor["width"] - 40))
    tracking_height = min(TRACKING_WINDOW_HEIGHT, max(240, primary_monitor["height"] - 80))

    return {
        "tracking": {
            "x": primary_monitor["left"] + 20,
            "y": primary_monitor["top"] + 40,
            "width": tracking_width,
            "height": tracking_height,
        },
        "ad": {
            "x": secondary_monitor["left"],
            "y": secondary_monitor["top"],
            "width": secondary_monitor["width"],
            "height": secondary_monitor["height"],
            "fullscreen": True,
        },
    }


def setup_windows():
    cv2.namedWindow(TRACKING_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(AD_WINDOW_NAME, cv2.WINDOW_NORMAL)

    layout = _pick_monitor_layout()
    if layout is None:
        cv2.resizeWindow(TRACKING_WINDOW_NAME, TRACKING_WINDOW_WIDTH, TRACKING_WINDOW_HEIGHT)
        cv2.resizeWindow(AD_WINDOW_NAME, AD_WINDOW_WIDTH, AD_WINDOW_HEIGHT)
        cv2.moveWindow(TRACKING_WINDOW_NAME, TRACKING_WINDOW_X, TRACKING_WINDOW_Y)
        cv2.moveWindow(AD_WINDOW_NAME, AD_WINDOW_X, AD_WINDOW_Y)
        fullscreen = AD_WINDOW_FULLSCREEN
    else:
        tracking = layout["tracking"]
        ad = layout["ad"]
        cv2.resizeWindow(TRACKING_WINDOW_NAME, tracking["width"], tracking["height"])
        cv2.moveWindow(TRACKING_WINDOW_NAME, tracking["x"], tracking["y"])
        cv2.resizeWindow(AD_WINDOW_NAME, ad["width"], ad["height"])
        cv2.moveWindow(AD_WINDOW_NAME, ad["x"], ad["y"])
        fullscreen = AD_WINDOW_FULLSCREEN or ad["fullscreen"]

    if fullscreen:
        cv2.setWindowProperty(AD_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def set_ad_fullscreen(enabled: bool):
    if enabled:
        cv2.setWindowProperty(AD_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(AD_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
