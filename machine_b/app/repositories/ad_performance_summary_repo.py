from datetime import date, datetime
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.ad_performance_summary import AdPerformanceSummary


class AdPerformanceSummaryRepo:
    def __init__(self, db: Session):
        self.db = db

    def get_by_unique_key(
        self,
        advertisement_id: int,
        audience_segment_id: int,
        stats_date: date,
    ) -> AdPerformanceSummary | None:
        stmt = (
            select(AdPerformanceSummary)
            .where(
                AdPerformanceSummary.advertisement_id == advertisement_id,
                AdPerformanceSummary.audience_segment_id == audience_segment_id,
                AdPerformanceSummary.stats_date == stats_date,
            )
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def create(
        self,
        advertisement_id: int,
        audience_segment_id: int,
        stats_date: date,
        play_count: int,
        viewer_count: int,
        avg_look_duration: float,
    ) -> AdPerformanceSummary:
        summary = AdPerformanceSummary(
            advertisement_id=advertisement_id,
            audience_segment_id=audience_segment_id,
            stats_date=stats_date,
            play_count=play_count,
            viewer_count=viewer_count,
            avg_look_duration=avg_look_duration,
            created_at=datetime.utcnow(),
        )

        self.db.add(summary)
        self.db.flush()
        return summary
