from datetime import date, timedelta

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.models.advertisement import Advertisement
from app.models.ad_performance_summary import AdPerformanceSummary
from app.models.ad_play_log import AdPlayLog
from app.models.audience_segment import AudienceSegment
from app.models.category import Category
from app.models.category_audience_score import CategoryAudienceScore


class DashboardRepo:
    def __init__(self, db: Session):
        self.db = db

    def get_total_plays(self) -> int:
        stmt = select(func.count(AdPlayLog.id))
        result = self.db.execute(stmt).scalar_one()
        return int(result or 0)

    def get_total_viewers(self) -> int:
        stmt = select(func.coalesce(func.sum(AdPlayLog.total_viewers), 0))
        result = self.db.execute(stmt).scalar_one()
        return int(result or 0)

    def get_avg_look_duration(self) -> float:
        stmt = select(func.coalesce(func.avg(AdPlayLog.avg_look_duration), 0.0))
        result = self.db.execute(stmt).scalar_one()
        return float(result or 0.0)

    def get_active_ads_count(self) -> int:
        stmt = select(func.count(Advertisement.id)).where(
            Advertisement.is_active.is_(True)
        )
        result = self.db.execute(stmt).scalar_one()
        return int(result or 0)

    def get_daily_play_trend(self, days: int = 7) -> list[dict]:
        start_date = date.today() - timedelta(days=days - 1)

        stmt = (
            select(
                AdPerformanceSummary.stats_date,
                func.coalesce(
                    func.sum(AdPerformanceSummary.play_count), 0
                ).label("play_count"),
            )
            .where(AdPerformanceSummary.stats_date >= start_date)
            .group_by(AdPerformanceSummary.stats_date)
            .order_by(AdPerformanceSummary.stats_date.asc())
        )

        rows = self.db.execute(stmt).all()

        return [
            {
                "date": row.stats_date.isoformat(),
                "play_count": int(row.play_count),
            }
            for row in rows
        ]

    def get_daily_viewer_trend(self, days: int = 7) -> list[dict]:
        start_date = date.today() - timedelta(days=days - 1)

        stmt = (
            select(
                AdPerformanceSummary.stats_date,
                func.coalesce(
                    func.sum(AdPerformanceSummary.viewer_count), 0
                ).label("viewer_count"),
            )
            .where(AdPerformanceSummary.stats_date >= start_date)
            .group_by(AdPerformanceSummary.stats_date)
            .order_by(AdPerformanceSummary.stats_date.asc())
        )

        rows = self.db.execute(stmt).all()

        return [
            {
                "date": row.stats_date.isoformat(),
                "viewer_count": int(row.viewer_count),
            }
            for row in rows
        ]

    def get_gender_distribution(self) -> list[dict]:
        stmt = (
            select(
                AudienceSegment.gender,
                func.coalesce(
                    func.sum(AdPerformanceSummary.viewer_count), 0
                ).label("viewer_count"),
            )
            .join(
                AudienceSegment,
                AudienceSegment.id == AdPerformanceSummary.audience_segment_id,
            )
            .group_by(AudienceSegment.gender)
            .order_by(AudienceSegment.gender.asc())
        )

        rows = self.db.execute(stmt).all()

        return [
            {
                "gender": row.gender,
                "viewer_count": int(row.viewer_count),
            }
            for row in rows
        ]

    def get_age_distribution(self) -> list[dict]:
        stmt = (
            select(
                AudienceSegment.age_group,
                AudienceSegment.age_min,
                func.coalesce(
                    func.sum(AdPerformanceSummary.viewer_count), 0
                ).label("viewer_count"),
            )
            .join(
                AudienceSegment,
                AudienceSegment.id == AdPerformanceSummary.audience_segment_id,
            )
            .group_by(AudienceSegment.age_group, AudienceSegment.age_min)
            .order_by(AudienceSegment.age_min.asc())
        )

        rows = self.db.execute(stmt).all()

        return [
            {
                "age_group": row.age_group,
                "viewer_count": int(row.viewer_count),
            }
            for row in rows
        ]

    def get_top_ads(self, limit: int = 5) -> list[dict]:
        stmt = (
            select(
                Advertisement.id.label("ad_id"),
                Advertisement.title,
                func.coalesce(
                    func.sum(AdPerformanceSummary.play_count), 0
                ).label("total_plays"),
                func.coalesce(
                    func.sum(AdPerformanceSummary.viewer_count), 0
                ).label("total_viewers"),
                func.coalesce(
                    func.avg(AdPerformanceSummary.avg_look_duration), 0.0
                ).label("avg_look_duration"),
            )
            .join(
                AdPerformanceSummary,
                AdPerformanceSummary.advertisement_id == Advertisement.id,
            )
            .group_by(Advertisement.id, Advertisement.title)
            .order_by(
                desc("avg_look_duration"),
                desc("total_viewers"),
                desc("total_plays"),
            )
            .limit(limit)
        )

        rows = self.db.execute(stmt).all()

        return [
            {
                "ad_id": row.ad_id,
                "title": row.title,
                "total_plays": int(row.total_plays),
                "total_viewers": int(row.total_viewers),
                "avg_look_duration": float(row.avg_look_duration),
            }
            for row in rows
        ]

    def get_recent_logs(self, limit: int = 10) -> list[dict]:
        stmt = (
            select(
                AdPlayLog.id.label("log_id"),
                AdPlayLog.played_at,
                AdPlayLog.total_viewers,
                AdPlayLog.avg_look_duration,
                Advertisement.title.label("ad_title"),
                AudienceSegment.gender,
                AudienceSegment.age_group,
            )
            .join(
                Advertisement,
                Advertisement.id == AdPlayLog.advertisement_id,
            )
            .outerjoin(
                AudienceSegment,
                AudienceSegment.id == AdPlayLog.dominant_audience_segment_id,
            )
            .order_by(AdPlayLog.played_at.desc())
            .limit(limit)
        )

        rows = self.db.execute(stmt).all()

        return [
            {
                "log_id": row.log_id,
                "played_at": row.played_at,
                "ad_title": row.ad_title,
                "total_viewers": int(row.total_viewers),
                "avg_look_duration": float(row.avg_look_duration),
                "dominant_segment": (
                    f"{row.gender} {row.age_group}"
                    if row.gender and row.age_group
                    else "Unknown"
                ),
            }
            for row in rows
        ]

    def get_top_categories(self, limit: int = 5) -> list[dict]:
        stmt = (
            select(
                Category.id.label("category_id"),
                Category.name.label("category_name"),
                func.coalesce(func.count(AdPlayLog.id), 0).label("total_plays"),
                func.coalesce(func.sum(AdPlayLog.total_viewers), 0).label(
                    "total_viewers"
                ),
                func.coalesce(func.avg(AdPlayLog.avg_look_duration), 0.0).label(
                    "avg_look_duration"
                ),
            )
            .join(Advertisement, Advertisement.category_id == Category.id)
            .join(AdPlayLog, AdPlayLog.advertisement_id == Advertisement.id)
            .group_by(Category.id, Category.name)
            .order_by(
                desc("avg_look_duration"),
                desc("total_viewers"),
                desc("total_plays"),
            )
            .limit(limit)
        )

        rows = self.db.execute(stmt).all()

        return [
            {
                "category_id": row.category_id,
                "category_name": row.category_name,
                "total_plays": int(row.total_plays),
                "total_viewers": int(row.total_viewers),
                "avg_look_duration": float(row.avg_look_duration),
            }
            for row in rows
        ]

    def get_all_categories(self) -> list[dict]:
        stmt = select(Category.id, Category.name).order_by(Category.name.asc())
        rows = self.db.execute(stmt).all()

        return [{"id": row.id, "name": row.name} for row in rows]

    def get_all_advertisements(self) -> list[dict]:
        stmt = select(Advertisement.id, Advertisement.title).order_by(
            Advertisement.title.asc(),
            Advertisement.id.asc(),
        )
        rows = self.db.execute(stmt).all()

        return [{"id": row.id, "title": row.title} for row in rows]

    def get_advertisements_with_metrics(
        self,
        search_term: str | None = None,
        category_id: int | None = None,
        status_filter: str | None = None,
    ) -> list[dict]:
        stmt = (
            select(
                Advertisement.id,
                Advertisement.title,
                Advertisement.description,
                Advertisement.media_filename,
                Advertisement.duration_seconds,
                Advertisement.is_active,
                Category.name.label("category_name"),
                func.coalesce(func.count(AdPlayLog.id), 0).label("total_plays"),
                func.coalesce(func.sum(AdPlayLog.total_viewers), 0).label(
                    "total_viewers"
                ),
                func.coalesce(func.avg(AdPlayLog.avg_look_duration), 0.0).label(
                    "avg_look_duration"
                ),
            )
            .join(Category, Category.id == Advertisement.category_id)
            .outerjoin(AdPlayLog, AdPlayLog.advertisement_id == Advertisement.id)
        )

        if search_term:
            keyword = f"%{search_term.strip()}%"
            stmt = stmt.where(
                Advertisement.title.ilike(keyword)
                | Advertisement.media_filename.ilike(keyword)
            )

        if category_id is not None:
            stmt = stmt.where(Advertisement.category_id == category_id)

        if status_filter == "active":
            stmt = stmt.where(Advertisement.is_active.is_(True))
        elif status_filter == "inactive":
            stmt = stmt.where(Advertisement.is_active.is_(False))

        stmt = stmt.group_by(
            Advertisement.id,
            Advertisement.title,
            Advertisement.description,
            Advertisement.media_filename,
            Advertisement.duration_seconds,
            Advertisement.is_active,
            Category.name,
        ).order_by(Advertisement.id.asc())

        rows = self.db.execute(stmt).all()

        return [
            {
                "id": row.id,
                "title": row.title,
                "description": row.description,
                "media_filename": row.media_filename,
                "duration_seconds": row.duration_seconds,
                "is_active": bool(row.is_active),
                "category_name": row.category_name,
                "total_plays": int(row.total_plays or 0),
                "total_viewers": int(row.total_viewers or 0),
                "avg_look_duration": float(row.avg_look_duration or 0.0),
            }
            for row in rows
        ]

    def get_categories_with_metrics(self) -> list[dict]:
        category_stmt = (
            select(
                Category.id,
                Category.name,
                func.coalesce(func.count(func.distinct(Advertisement.id)), 0).label(
                    "ad_count"
                ),
                func.coalesce(func.count(AdPlayLog.id), 0).label("total_plays"),
                func.coalesce(func.sum(AdPlayLog.total_viewers), 0).label(
                    "total_viewers"
                ),
                func.coalesce(func.avg(AdPlayLog.avg_look_duration), 0.0).label(
                    "avg_look_duration"
                ),
            )
            .outerjoin(Advertisement, Advertisement.category_id == Category.id)
            .outerjoin(AdPlayLog, AdPlayLog.advertisement_id == Advertisement.id)
            .group_by(Category.id, Category.name)
            .order_by(Category.name.asc())
        )

        category_rows = self.db.execute(category_stmt).all()

        score_stmt = (
            select(
                CategoryAudienceScore.category_id,
                AudienceSegment.gender,
                AudienceSegment.age_group,
                CategoryAudienceScore.current_score,
            )
            .join(
                AudienceSegment,
                AudienceSegment.id == CategoryAudienceScore.audience_segment_id,
            )
            .order_by(
                CategoryAudienceScore.category_id.asc(),
                CategoryAudienceScore.current_score.desc(),
            )
        )

        score_rows = self.db.execute(score_stmt).all()

        segment_map: dict[int, list[dict]] = {}
        for row in score_rows:
            segment_map.setdefault(row.category_id, []).append(
                {
                    "gender": row.gender,
                    "age_group": row.age_group,
                    "current_score": float(row.current_score),
                }
            )

        results = []
        for row in category_rows:
            results.append(
                {
                    "id": row.id,
                    "name": row.name,
                    "ad_count": int(row.ad_count or 0),
                    "total_plays": int(row.total_plays or 0),
                    "total_viewers": int(row.total_viewers or 0),
                    "avg_look_duration": float(row.avg_look_duration or 0.0),
                    "top_segments": segment_map.get(row.id, [])[:3],
                }
            )

        return results

    def get_play_logs_with_details(self, limit: int = 50) -> list[dict]:
        stmt = (
            select(
                AdPlayLog.id.label("log_id"),
                AdPlayLog.played_at,
                AdPlayLog.total_viewers,
                AdPlayLog.avg_look_duration,
                Advertisement.title.label("ad_title"),
                Category.name.label("category_name"),
                AudienceSegment.gender,
                AudienceSegment.age_group,
            )
            .join(
                Advertisement,
                Advertisement.id == AdPlayLog.advertisement_id,
            )
            .join(
                Category,
                Category.id == Advertisement.category_id,
            )
            .outerjoin(
                AudienceSegment,
                AudienceSegment.id == AdPlayLog.dominant_audience_segment_id,
            )
            .order_by(AdPlayLog.played_at.desc())
            .limit(limit)
        )

        rows = self.db.execute(stmt).all()

        return [
            {
                "log_id": row.log_id,
                "played_at": row.played_at,
                "ad_title": row.ad_title,
                "category_name": row.category_name,
                "total_viewers": int(row.total_viewers),
                "avg_look_duration": float(row.avg_look_duration),
                "dominant_segment": (
                    f"{row.gender} {row.age_group}"
                    if row.gender and row.age_group
                    else "Unknown"
                ),
            }
            for row in rows
        ]

    def get_report_summary(
        self,
        period: str = "daily",
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> dict:
        summary_stmt = select(
            func.coalesce(func.sum(AdPerformanceSummary.play_count), 0).label(
                "total_plays"
            ),
            func.coalesce(func.sum(AdPerformanceSummary.viewer_count), 0).label(
                "total_viewers"
            ),
            func.coalesce(func.avg(AdPerformanceSummary.avg_look_duration), 0.0).label(
                "avg_look_duration"
            ),
        )

        summary_stmt = self._apply_report_filters(
            stmt=summary_stmt,
            period=period,
            date_from=date_from,
            date_to=date_to,
        )

        summary_row = self.db.execute(summary_stmt).one()

        best_ad_stmt = (
            select(
                Advertisement.title.label("title"),
                func.coalesce(
                    func.avg(AdPerformanceSummary.avg_look_duration), 0.0
                ).label("avg_look_duration"),
                func.coalesce(func.sum(AdPerformanceSummary.viewer_count), 0).label(
                    "total_viewers"
                ),
            )
            .join(
                AdPerformanceSummary,
                AdPerformanceSummary.advertisement_id == Advertisement.id,
            )
            .group_by(Advertisement.id, Advertisement.title)
            .order_by(
                desc("avg_look_duration"),
                desc("total_viewers"),
            )
            .limit(1)
        )

        best_ad_stmt = self._apply_report_filters(
            stmt=best_ad_stmt,
            period=period,
            date_from=date_from,
            date_to=date_to,
        )

        best_ad_row = self.db.execute(best_ad_stmt).one_or_none()

        return {
            "total_plays": int(summary_row.total_plays or 0),
            "total_viewers": int(summary_row.total_viewers or 0),
            "avg_look_duration": float(summary_row.avg_look_duration or 0.0),
            "best_ad_title": best_ad_row.title if best_ad_row else None,
        }

    def get_report_rows(
        self,
        limit: int = 100,
        period: str = "daily",
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[dict]:
        stmt = (
            select(
                AdPerformanceSummary.stats_date,
                Advertisement.title.label("ad_title"),
                Category.name.label("category_name"),
                AudienceSegment.gender,
                AudienceSegment.age_group,
                AdPerformanceSummary.play_count,
                AdPerformanceSummary.viewer_count,
                AdPerformanceSummary.avg_look_duration,
            )
            .join(
                Advertisement,
                Advertisement.id == AdPerformanceSummary.advertisement_id,
            )
            .join(
                Category,
                Category.id == Advertisement.category_id,
            )
            .join(
                AudienceSegment,
                AudienceSegment.id == AdPerformanceSummary.audience_segment_id,
            )
        )

        stmt = self._apply_report_filters(
            stmt=stmt,
            period=period,
            date_from=date_from,
            date_to=date_to,
        ).order_by(
            AdPerformanceSummary.stats_date.desc(),
            Advertisement.title.asc(),
        ).limit(limit)

        rows = self.db.execute(stmt).all()

        return [
            {
                "stats_date": row.stats_date.isoformat(),
                "ad_title": row.ad_title,
                "category_name": row.category_name,
                "segment_label": f"{row.gender} {row.age_group}",
                "play_count": int(row.play_count),
                "viewer_count": int(row.viewer_count),
                "avg_look_duration": float(row.avg_look_duration),
            }
            for row in rows
        ]

    def _apply_report_filters(
        self,
        stmt,
        period: str,
        date_from: date | None,
        date_to: date | None,
    ):
        if date_from is not None:
            stmt = stmt.where(AdPerformanceSummary.stats_date >= date_from)

        if date_to is not None:
            stmt = stmt.where(AdPerformanceSummary.stats_date <= date_to)

        if period == "weekly":
            start_date = date.today() - timedelta(days=6)
            stmt = stmt.where(AdPerformanceSummary.stats_date >= start_date)
        elif period == "monthly":
            start_date = date.today() - timedelta(days=29)
            stmt = stmt.where(AdPerformanceSummary.stats_date >= start_date)

        return stmt
