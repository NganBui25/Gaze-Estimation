from dataclasses import dataclass, field

from .config import (
    GAZE_EMA_ALPHA,
    GAZE_STATE_IOU_THRESHOLD,
    GAZE_STATE_TTL_SECONDS,
    TRACK_MATCH_IOU_THRESHOLD,
)
from .reporting import iso_now, majority_vote


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


class GazeStateManager:
    def __init__(
        self,
        *,
        iou_threshold=GAZE_STATE_IOU_THRESHOLD,
        ttl_seconds=GAZE_STATE_TTL_SECONDS,
        alpha=GAZE_EMA_ALPHA,
    ):
        self.iou_threshold = iou_threshold
        self.ttl_seconds = ttl_seconds
        self.alpha = alpha
        self.entries = []

    def update(self, bbox, yaw, pitch, timestamp):
        self.entries = [
            entry
            for entry in self.entries
            if timestamp - entry["updated_at"] <= self.ttl_seconds
        ]
        entry = self._find_best_match(bbox)
        if entry is None:
            entry = {
                "bbox": tuple(bbox),
                "smooth_yaw": float(yaw),
                "smooth_pitch": float(pitch),
                "updated_at": timestamp,
            }
            self.entries.append(entry)
        else:
            entry["bbox"] = tuple(bbox)
            entry["smooth_yaw"] = self.alpha * yaw + (1.0 - self.alpha) * entry["smooth_yaw"]
            entry["smooth_pitch"] = self.alpha * pitch + (1.0 - self.alpha) * entry["smooth_pitch"]
            entry["updated_at"] = timestamp
        return entry["smooth_yaw"], entry["smooth_pitch"]

    def _find_best_match(self, bbox):
        best_entry = None
        best_score = 0.0
        for entry in self.entries:
            score = bbox_iou(entry["bbox"], bbox)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score < self.iou_threshold:
            return None
        return best_entry


@dataclass
class ViewerTrack:
    track_id: int
    bbox: tuple
    first_seen_ts: float
    last_seen_ts: float
    audience_segment_samples: list = field(default_factory=list)
    looking_samples: list = field(default_factory=list)
    last_looking_value: object = None
    looking_duration_total: float = 0.0
    frames_seen: int = 0

    def update(self, bbox, audience_segment_id, looking_value, timestamp):
        previous_seen_ts = self.last_seen_ts
        self.bbox = bbox
        self.last_seen_ts = timestamp
        self.frames_seen += 1
        if audience_segment_id is not None:
            self.audience_segment_samples.append(int(audience_segment_id))
        if looking_value is not None:
            self.looking_samples.append(bool(looking_value))
            if looking_value and self.last_looking_value is True and previous_seen_ts is not None:
                self.looking_duration_total += max(0.0, timestamp - previous_seen_ts)
        self.last_looking_value = looking_value

    def summary(self, session_end_ts=None):
        end_ts = self.last_seen_ts if session_end_ts is None else session_end_ts
        audience_segment_id = majority_vote(self.audience_segment_samples, default=None)
        watch_duration = self.looking_duration_total
        if (
            session_end_ts is not None
            and self.last_looking_value is True
            and end_ts > self.last_seen_ts
        ):
            watch_duration += max(0.0, end_ts - self.last_seen_ts)
        return {
            "viewer_id": self.track_id,
            "audience_segment_id": audience_segment_id,
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
                    detection.get("audience_segment_id"),
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
                    detection.get("audience_segment_id"),
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
