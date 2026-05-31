import os
import threading

from .config import AD_MEDIA_ROOT, APP_DIR, ROOT_DIR
from .reporting import post_json_async


class AdSelectionRequest:
    def __init__(self, url, payload, generation_id):
        self.url = url
        self.payload = payload
        self.generation_id = generation_id
        self.event = threading.Event()
        self.response = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            import requests

            response = requests.post(self.url, json=self.payload, timeout=8)
            response.raise_for_status()
            self.response = response.json()
        except Exception as exc:
            self.error = exc
        finally:
            self.event.set()


def resolve_media_path(media_filename):
    if not media_filename:
        return None
    candidate_paths = []
    if os.path.isabs(media_filename):
        candidate_paths.append(media_filename)
    candidate_paths.extend(
        [
            os.path.join(AD_MEDIA_ROOT, media_filename),
            os.path.join(APP_DIR, media_filename),
            os.path.join(ROOT_DIR, media_filename),
            os.path.join(os.getcwd(), media_filename),
        ]
    )
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return candidate_paths[0] if candidate_paths else media_filename


def extract_ad_selection(payload):
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected advertisement payload: {payload!r}")

    nested_payload = payload.get("data")
    if isinstance(nested_payload, dict):
        payload = nested_payload

    advertisement_payload = payload.get("advertisement")
    if isinstance(advertisement_payload, dict):
        payload = advertisement_payload

    ad_id = payload.get("ad_id") or payload.get("id")
    media_filename = (
        payload.get("media_filename")
        or payload.get("mediaFileName")
        or payload.get("filename")
        or payload.get("media")
    )
    duration_seconds = payload.get("duration_seconds")
    if duration_seconds is None:
        duration_seconds = payload.get("duration") or payload.get("seconds")

    if ad_id is None or media_filename is None or duration_seconds is None:
        raise ValueError(f"Missing ad fields in response: {payload!r}")

    return {
        "ad_id": ad_id,
        "media_filename": media_filename,
        "duration_seconds": float(duration_seconds),
    }
