import socket
import threading
import time

from .config import (
    SENSOR_BAUD_RATE,
    SENSOR_DEFAULT_STATE,
    SENSOR_POLL_INTERVAL,
    SENSOR_SERIAL_PORT,
    SENSOR_SOCKET_HOST,
    SENSOR_SOCKET_PORT,
)


def normalize_sensor_state(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip().lower()
    if not value:
        return None
    if value in {"light", "bright", "on", "1", "true", "open"}:
        return "Light"
    if value in {"dark", "off", "0", "false", "closed"}:
        return "Dark"
    if "light" in value:
        return "Light"
    if "dark" in value:
        return "Dark"
    return None


class SensorMonitor:
    def __init__(self, default_state=SENSOR_DEFAULT_STATE):
        self._state = normalize_sensor_state(default_state) or "Light"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_state(self):
        with self._lock:
            return self._state

    def _set_state(self, value):
        normalized_value = normalize_sensor_state(value)
        if normalized_value is None:
            return
        with self._lock:
            self._state = normalized_value

    def _run(self):
        if SENSOR_SERIAL_PORT:
            if self._run_serial_loop():
                return
        if SENSOR_SOCKET_HOST:
            if self._run_socket_loop():
                return
        while not self._stop_event.is_set():
            time.sleep(SENSOR_POLL_INTERVAL)

    def _run_serial_loop(self):
        try:
            import serial  # type: ignore
        except Exception as exc:
            print(f"Serial sensor unavailable: {exc}")
            return False

        while not self._stop_event.is_set():
            try:
                with serial.Serial(SENSOR_SERIAL_PORT, SENSOR_BAUD_RATE, timeout=1) as ser:
                    print(f"Connected to sensor serial port {SENSOR_SERIAL_PORT}")
                    while not self._stop_event.is_set():
                        raw_value = ser.readline().decode("utf-8", errors="ignore").strip()
                        self._set_state(raw_value)
            except Exception as exc:
                print(f"Serial sensor read failed: {exc}")
                time.sleep(1.0)
        return True

    def _run_socket_loop(self):
        while not self._stop_event.is_set():
            try:
                with socket.create_connection((SENSOR_SOCKET_HOST, SENSOR_SOCKET_PORT), timeout=5.0) as sock:
                    print(f"Connected to sensor socket {SENSOR_SOCKET_HOST}:{SENSOR_SOCKET_PORT}")
                    sock.settimeout(1.0)
                    buffer = ""
                    while not self._stop_event.is_set():
                        try:
                            data = sock.recv(1024)
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        buffer += data.decode("utf-8", errors="ignore")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            self._set_state(line)
            except Exception as exc:
                print(f"Socket sensor read failed: {exc}")
                time.sleep(1.0)
        return True
