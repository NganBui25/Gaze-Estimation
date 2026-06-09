from datetime import datetime
from sqlalchemy import delete
from sqlalchemy.orm import Session
from app.models.ad_play_log import AdPlayLog

class AdPlayLogRepo:
    def __init__(self, db: Session):
        self.db = db

    def create(self, advertisement_id: int, played_at: datetime, total_viewers: int,
        avg_look_duration: float, dominant_audience_segment_id: int | None,
    ):
        ad_play_log = AdPlayLog(
            advertisement_id=advertisement_id,
            played_at=played_at,
            total_viewers=total_viewers,
            avg_look_duration=avg_look_duration,
            dominant_audience_segment_id=dominant_audience_segment_id,
        )

        self.db.add(ad_play_log)
        self.db.flush()
        return ad_play_log

    def delete_older_than(self, cutoff: datetime) -> int:
        result = self.db.execute(
            delete(AdPlayLog).where(AdPlayLog.played_at < cutoff)
        )
        return int(result.rowcount or 0)
