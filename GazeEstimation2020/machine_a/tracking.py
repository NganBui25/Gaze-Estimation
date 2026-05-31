from dataclasses import dataclass, field

from .config import TRACK_MATCH_IOU_THRESHOLD
from .reporting import iso_now, mean_or_none, majority_vote


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter_area / max(area_a + area_b - inter_area, 1.0)


@dataclass
class ViewerTrack:
    track_id: int
    bbox: tuple
    first_seen_ts: float
    last_seen_ts: float
    age_samples: list = field(default_factory=list)
    gender_samples: list = field(default_factory=list)
    looking_samples: list = field(default_factory=list)
    looking_duration_total: float = 0.0
    frames_seen: int = 0

    def update(self, bbox, age_value, gender_value, looking_value, timestamp):
        previous_seen_ts = self.last_seen_ts
        self.bbox = bbox
        self.last_seen_ts = timestamp
        self.frames_seen += 1
        if age_value is not None:
            self.age_samples.append(float(age_value))
        if gender_value is not None:
            self.gender_samples.append(gender_value)
        if looking_value is not None:
            self.looking_samples.append(bool(looking_value))
            if looking_value and previous_seen_ts is not None:
                self.looking_duration_total += max(0.0, timestamp - previous_seen_ts)

    def summary(self, session_end_ts=None):
        end_ts = self.last_seen_ts if session_end_ts is None else session_end_ts
        estimated_age = mean_or_none(self.age_samples)
        gender = majority_vote(self.gender_samples)
        watch_duration = self.looking_duration_total
        if (
            session_end_ts is not None
            and self.looking_samples
            and self.looking_samples[-1]
            and end_ts > self.last_seen_ts
        ):
            watch_duration += max(0.0, end_ts - self.last_seen_ts)
        return {
            "viewer_id": self.track_id,
            "estimated_age": None if estimated_age is None else int(estimated_age),
            "gender": str(gender).strip().lower() if gender is not None else "unknown",
            "start_time": iso_now(self.first_seen_ts),
            "end_time": iso_now(end_ts),
            "watch_duration": round(watch_duration, 3),
        }


class ViewerTrackManager:
    def __init__(self, iou_threshold=TRACK_MATCH_IOU_THRESHOLD):
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_track_id = 1

    def reset(self):
        self.tracks = []
        self.next_track_id = 1

    def update(self, detections, timestamp):
        unmatched_track_indexes = set(range(len(self.tracks)))

        for detection in detections:
            best_index = None
            best_score = 0.0
            for track_index in unmatched_track_indexes:
                score = bbox_iou(detection["bbox"], self.tracks[track_index].bbox)
                if score > best_score:
                    best_score = score
                    best_index = track_index

            if best_index is not None and best_score >= self.iou_threshold:
                self.tracks[best_index].update(
                    detection["bbox"],
                    detection.get("estimated_age"),
                    detection.get("gender"),
                    detection.get("looking"),
                    timestamp,
                )
                unmatched_track_indexes.discard(best_index)
            else:
                new_track = ViewerTrack(
                    track_id=self.next_track_id,
                    bbox=detection["bbox"],
                    first_seen_ts=timestamp,
                    last_seen_ts=timestamp,
                )
                new_track.update(
                    detection["bbox"],
                    detection.get("estimated_age"),
                    detection.get("gender"),
                    detection.get("looking"),
                    timestamp,
                )
                self.tracks.append(new_track)
                self.next_track_id += 1

    def active_summaries(self, session_end_ts=None):
        return [
            track.summary(session_end_ts=session_end_ts)
            for track in self.tracks
            if track.frames_seen > 0
        ]
