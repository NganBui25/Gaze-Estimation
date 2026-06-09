import queue
import threading
import time

from .tracking import bbox_iou
from .vision_utils import predict_age_gender


class DemographicPredictor:
    def __init__(
        self,
        model,
        *,
        refresh_seconds=1.0,
        cache_ttl_seconds=3.0,
        match_iou_threshold=0.25,
        queue_size=4,
    ):
        self.model = model
        self.refresh_seconds = refresh_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.match_iou_threshold = match_iou_threshold
        self.tasks = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.cache = []
        self.pending = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def get_prediction(self, bbox, now_ts=None):
        now_ts = time.time() if now_ts is None else now_ts
        with self.lock:
            self._remove_expired(now_ts)
            match = self._find_best_match(self.cache, bbox)
            return None if match is None else match["prediction"]

    def submit(self, bbox, face_crop, now_ts=None):
        if face_crop.size == 0:
            return

        now_ts = time.time() if now_ts is None else now_ts
        with self.lock:
            self._remove_expired(now_ts)
            cached = self._find_best_match(self.cache, bbox)
            if cached is not None and now_ts - cached["updated_at"] < self.refresh_seconds:
                return

            pending = self._find_best_match(self.pending, bbox)
            if pending is not None and now_ts - pending["queued_at"] < self.refresh_seconds:
                return

            task = {
                "bbox": tuple(bbox),
                "face_crop": face_crop.copy(),
                "queued_at": now_ts,
            }
            try:
                self.tasks.put_nowait(task)
            except queue.Full:
                return
            self.pending.append(task)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                task = self.tasks.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                prediction = predict_age_gender(self.model, task["face_crop"])
                now_ts = time.time()
                with self.lock:
                    pending = self._find_best_match(self.pending, task["bbox"])
                    if pending is not None:
                        self.pending.remove(pending)

                    cached = self._find_best_match(self.cache, task["bbox"])
                    entry = {
                        "bbox": task["bbox"],
                        "prediction": prediction,
                        "updated_at": now_ts,
                    }
                    if cached is None:
                        self.cache.append(entry)
                    else:
                        cached.update(entry)
            except Exception as exc:
                print(f"Age/Gender prediction error: {exc}")
                with self.lock:
                    pending = self._find_best_match(self.pending, task["bbox"])
                    if pending is not None:
                        self.pending.remove(pending)
            finally:
                self.tasks.task_done()

    def _remove_expired(self, now_ts):
        self.cache = [
            entry
            for entry in self.cache
            if now_ts - entry["updated_at"] <= self.cache_ttl_seconds
        ]
        self.pending = [
            entry
            for entry in self.pending
            if now_ts - entry["queued_at"] <= self.cache_ttl_seconds
        ]

    def _find_best_match(self, entries, bbox):
        best_entry = None
        best_score = 0.0
        for entry in entries:
            score = bbox_iou(entry["bbox"], bbox)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score < self.match_iou_threshold:
            return None
        return best_entry
