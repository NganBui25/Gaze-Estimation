from datetime import date

from sqlalchemy.orm import Session

from app.models.ad_performance_summary import AdPerformanceSummary
from app.repositories.ad_performance_summary_repo import AdPerformanceSummaryRepo


class AdPerformanceSummaryService:
    def __init__(self, db: Session):
        self.db = db
        self.ad_performance_summary_repo = AdPerformanceSummaryRepo(db)

    def update_summary(self, advertisement_id: int, audience_segment_id: int,
                       stats_date: date, viewer_count_increment: int, total_watch_duration_increment: float,):
        if viewer_count_increment <= 0:
            raise ValueError("viewer_count must be greater than 0")
        if total_watch_duration_increment < 0:
            raise ValueError("total_watch_duration_increment must be greater than or equal to 0")
        
        summary = self.ad_performance_summary_repo.get_by_unique_key(
            advertisement_id=advertisement_id,
            audience_segment_id = audience_segment_id,
            stats_date = stats_date,
        )
        new_avg_look_duration = total_watch_duration_increment/viewer_count_increment

        if summary is None:
            return self.ad_performance_summary_repo.create(
                advertisement_id=advertisement_id,
                audience_segment_id=audience_segment_id,
                stats_date=stats_date,
                play_count=1,
                viewer_count=viewer_count_increment,
                avg_look_duration=new_avg_look_duration,
            )
        
        old_total_watch_duration = summary.avg_look_duration*summary.viewer_count
        new_total_view_count = summary.viewer_count + viewer_count_increment
        new_total_watch_duration = old_total_watch_duration + total_watch_duration_increment

        summary.play_count += 1
        summary.viewer_count = new_total_view_count
        summary.avg_look_duration = new_total_watch_duration/new_total_view_count

        self.db.flush()
        return summary