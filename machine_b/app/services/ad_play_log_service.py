from collections import defaultdict

from sqlalchemy.orm import Session

from app.DTO.ad_report import AdReportRequest, AdReportResponse
from app.repositories.ad_play_log_repo import AdPlayLogRepo
from app.repositories.advertisement_repo import AdvertisementRepo
from app.services.ad_performance_summary_service import AdPerformanceSummaryService
from app.services.audience_segment_service import AudienceSegmentService
from app.services.category_audience_score_service import CategoryAudienceScoreService

class AdPlayLogService:
    def __init__(self, db: Session):
        self.db = db
        self.ad_play_log_repo = AdPlayLogRepo(db)
        self.advertisement_repo = AdvertisementRepo(db)
        self.audience_segment_service = AudienceSegmentService(db)
        self.ad_performance_summary_service = AdPerformanceSummaryService(db)
        self.category_audience_score_service = CategoryAudienceScoreService(db)
    
    def create_report(self, request: AdReportRequest):
        advertisement = self.advertisement_repo.get_by_id(request.ad_id)
        if advertisement is None:
            raise LookupError("advertisement not found")

        if request.end_time <= request.start_time:
            raise ValueError("end_time must be greater than start_time")

        if request.total_viewers != len(request.viewers):
            raise ValueError("total_viewers must be equal to the number of items in viewers")
        
        total_watch_duration = 0.0
        grouped_stats: dict[int, dict[str, float | int]] = defaultdict(
            lambda:{
                "viewer_count": 0,
                "total_watch_duration": 0.0,
            }
        )

        try: 
            for viewer in request.viewers:
                audience_segment = self.audience_segment_service.get_segment_by_age_and_gender(
                    avg_age = viewer.estimated_age,
                    gender = viewer.gender,
                )
                grouped_stats[audience_segment.id]["viewer_count"] += 1
                grouped_stats[audience_segment.id]["total_watch_duration"] += viewer.watch_duration
                total_watch_duration += viewer.watch_duration

            avg_look_duration = 0.0
            if request.total_viewers > 0:
                avg_look_duration = total_watch_duration/request.total_viewers
            dominant_audience_segment_id = self._get_dominant_audience_segment_id(grouped_stats)
            stats_date = request.start_time.date()
            ad_play_log = self.ad_play_log_repo.create(
                advertisement_id=request.ad_id,
                played_at = request.start_time,
                total_viewers=request.total_viewers,
                avg_look_duration=avg_look_duration,
                dominant_audience_segment_id=dominant_audience_segment_id,
            )
            stats_date = request.start_time.date()

            for audience_segment_id, segment_stat in grouped_stats.items():
                viewer_count_increment = int(segment_stat["viewer_count"])
                total_watch_duration_increment = float(segment_stat["total_watch_duration"])
                self.category_audience_score_service.update_current_score(
                    category_id=advertisement.category_id,
                    audience_segment_id=audience_segment_id,
                    viewer_count=viewer_count_increment,
                    total_watch_duration=total_watch_duration_increment,
                    ad_duration_seconds=advertisement.duration_seconds,
                )

                self.ad_performance_summary_service.update_summary(
                    advertisement_id=request.ad_id,
                    audience_segment_id=audience_segment_id,
                    stats_date=stats_date,
                    viewer_count_increment=viewer_count_increment,
                    total_watch_duration_increment=total_watch_duration_increment,
                )

            self.db.commit()
            self.db.refresh(ad_play_log)

            return AdReportResponse(
                ad_play_log_id=ad_play_log.id,
                advertisement_id=request.ad_id,
                total_viewers=request.total_viewers,
                avg_look_duration=avg_look_duration,
                dominant_audience_segment_id=dominant_audience_segment_id,
                stats_date=stats_date,
            )
        except Exception:
            self.db.rollback()
            raise


    def _get_dominant_audience_segment_id(
        self,
        grouped_stats: dict[int, dict[str, float | int]],
    ) -> int | None:
        if not grouped_stats:
            return None

        dominant_segment_id = max(
            grouped_stats.items(),
            key=lambda item: (
                int(item[1]["viewer_count"]),
                float(item[1]["total_watch_duration"]),
                -item[0],
            ),
        )[0]

        return dominant_segment_id
                
