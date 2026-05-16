import csv
import os
from datetime import datetime

import numpy as np

from .config import FINAL_STATS_URLS, TRACKING_TEST_CSV


def iso_now(ts=None):
    if ts is None:
        return datetime.now().isoformat(timespec="seconds")
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def mean_or_none(values):
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def majority_vote(values, default="Unknown"):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return default
    from collections import Counter

    return Counter(filtered).most_common(1)[0][0]


def normalize_gender_for_api(gender):
    if gender is None:
        return "unknown"
    normalized_gender = str(gender).strip().lower()
    if normalized_gender in {"male", "female", "unknown"}:
        return normalized_gender
    return "unknown"


def post_json_async(url, payload):
    import threading
    import requests

    def _worker():
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
        except Exception as exc:
            print(f"HTTP POST failed for {url}: {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def post_json_async_many(urls, payload):
    seen = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        post_json_async(url, payload)


def append_tracking_report(report):
    file_exists = os.path.exists(TRACKING_TEST_CSV)
    with open(TRACKING_TEST_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["gender", "age", "duration", "type"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(report)


def resolve_age_gender_for_report(last_age_gender, session_gender, session_age):
    if last_age_gender is not None:
        gender_label, _, age_pred = last_age_gender
        return gender_label, age_pred
    return session_gender, session_age


def finalize_attention_session(last_age_gender, session_gender, session_age, start_time, last_seen_time):
    end_time = last_seen_time
    duration = round(end_time - start_time, 3)
    final_gender, final_age = resolve_age_gender_for_report(
        last_age_gender,
        session_gender,
        session_age,
    )
    final_report = {
        "gender": final_gender,
        "age": final_age,
        "duration": duration,
        "type": "final",
    }
    post_json_async_many(FINAL_STATS_URLS, final_report)
    append_tracking_report(final_report)
    print("Tracking final report:", final_report)
    return final_report
