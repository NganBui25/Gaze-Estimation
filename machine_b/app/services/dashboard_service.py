from datetime import date

from sqlalchemy.orm import Session

from app.repositories.advertisement_repo import AdvertisementRepo
from app.repositories.category_repo import CategoryRepo
from app.repositories.dashboard_repo import DashboardRepo


class DashboardService:
    def __init__(self, db: Session):
        self.db = db
        self.dashboard_repo = DashboardRepo(db)
        self.advertisement_repo = AdvertisementRepo(db)
        self.category_repo = CategoryRepo(db)

    def get_overview(self, trend_days: int = 7) -> dict:
        total_plays = self.dashboard_repo.get_total_plays()
        total_viewers = self.dashboard_repo.get_total_viewers()
        avg_look_duration = self.dashboard_repo.get_avg_look_duration()
        active_ads = self.dashboard_repo.get_active_ads_count()

        daily_play_trend = self.dashboard_repo.get_daily_play_trend(days=trend_days)
        daily_viewer_trend = self.dashboard_repo.get_daily_viewer_trend(
            days=trend_days
        )
        gender_distribution = self.dashboard_repo.get_gender_distribution()
        age_distribution = self.dashboard_repo.get_age_distribution()
        top_ads = self.dashboard_repo.get_top_ads(limit=5)
        recent_logs = self.dashboard_repo.get_recent_logs(limit=10)
        top_categories = self.dashboard_repo.get_top_categories(limit=5)

        return {
            "kpis": {
                "total_plays": total_plays,
                "total_viewers": total_viewers,
                "avg_look_duration": round(avg_look_duration, 2),
                "active_ads": active_ads,
            },
            "daily_play_trend": daily_play_trend,
            "daily_viewer_trend": daily_viewer_trend,
            "gender_distribution": gender_distribution,
            "age_distribution": age_distribution,
            "top_ads": self._format_top_ads(top_ads),
            "recent_logs": self._format_recent_logs(recent_logs),
            "top_categories": self._format_top_categories(top_categories),
        }

    def get_advertisements_page_data(
        self,
        search_term: str | None = None,
        category_id: int | None = None,
        status_filter: str | None = None,
    ) -> list[dict]:
        advertisements = self.dashboard_repo.get_advertisements_with_metrics(
            search_term=search_term,
            category_id=category_id,
            status_filter=status_filter,
        )

        return [
            {
                "id": ad["id"],
                "title": ad["title"],
                "description": ad["description"],
                "media_filename": ad["media_filename"],
                "duration_seconds": ad["duration_seconds"],
                "is_active": ad["is_active"],
                "category_name": ad["category_name"],
                "total_plays": ad["total_plays"],
                "total_viewers": ad["total_viewers"],
                "avg_look_duration": round(ad["avg_look_duration"], 2),
            }
            for ad in advertisements
        ]

    def create_advertisement(
        self,
        title: str,
        description: str | None,
        media_filename: str,
        duration_seconds: int,
        category_id: int,
    ) -> int:
        payload = self._validate_advertisement_payload(
            title=title,
            description=description,
            media_filename=media_filename,
            duration_seconds=duration_seconds,
            category_id=category_id,
        )

        advertisement = self.advertisement_repo.create(**payload)
        self.db.commit()
        return advertisement.id

    def update_advertisement(
        self,
        advertisement_id: int,
        title: str,
        description: str | None,
        media_filename: str,
        duration_seconds: int,
        category_id: int,
    ) -> None:
        advertisement = self.advertisement_repo.get_by_id(advertisement_id)
        if advertisement is None:
            raise LookupError("advertisement not found")

        payload = self._validate_advertisement_payload(
            title=title,
            description=description,
            media_filename=media_filename,
            duration_seconds=duration_seconds,
            category_id=category_id,
        )

        advertisement.title = payload["title"]
        advertisement.description = payload["description"]
        advertisement.media_filename = payload["media_filename"]
        advertisement.duration_seconds = payload["duration_seconds"]
        advertisement.category_id = payload["category_id"]

        self.db.commit()

    def toggle_advertisement(self, advertisement_id: int) -> bool:
        advertisement = self.advertisement_repo.get_by_id(advertisement_id)
        if advertisement is None:
            raise LookupError("advertisement not found")

        advertisement.is_active = not advertisement.is_active
        self.db.commit()
        return advertisement.is_active

    def delete_advertisement(self, advertisement_id: int) -> None:
        advertisement = self.advertisement_repo.get_by_id(advertisement_id)
        if advertisement is None:
            raise LookupError("advertisement not found")

        if self.advertisement_repo.has_related_activity(advertisement_id):
            raise ValueError(
                "cannot delete advertisement with existing performance data"
            )

        self.advertisement_repo.delete(advertisement)
        self.db.commit()

    def get_category_options(self) -> list[dict]:
        categories = self.dashboard_repo.get_all_categories()
        return [{"id": category["id"], "name": category["name"]} for category in categories]

    def get_advertisement_options(self) -> list[dict]:
        advertisements = self.dashboard_repo.get_all_advertisements()
        return [{"id": ad["id"], "title": ad["title"]} for ad in advertisements]

    def get_categories_page_data(self) -> list[dict]:
        categories = self.dashboard_repo.get_categories_with_metrics()

        return [
            {
                "id": category["id"],
                "name": category["name"],
                "ad_count": category["ad_count"],
                "total_plays": category["total_plays"],
                "total_viewers": category["total_viewers"],
                "avg_look_duration": round(category["avg_look_duration"], 2),
                "top_segments": self._format_top_segments(category["top_segments"]),
            }
            for category in categories
        ]

    def get_play_logs_page_data(self, limit: int = 50) -> list[dict]:
        play_logs = self.dashboard_repo.get_play_logs_with_details(limit=limit)

        formatted_logs = []
        for log in play_logs:
            played_at = log["played_at"]
            played_at_text = (
                played_at.strftime("%Y-%m-%d %H:%M:%S")
                if played_at is not None
                else ""
            )

            formatted_logs.append(
                {
                    "log_id": log["log_id"],
                    "played_at": played_at,
                    "played_at_text": played_at_text,
                    "ad_title": log["ad_title"],
                    "category_name": log["category_name"],
                    "total_viewers": log["total_viewers"],
                    "avg_look_duration": round(log["avg_look_duration"], 2),
                    "dominant_segment": log["dominant_segment"],
                }
            )

        return formatted_logs

    def get_report_summary(self) -> dict:
        summary = self.dashboard_repo.get_report_summary()

        return {
            "total_plays": summary["total_plays"],
            "total_viewers": summary["total_viewers"],
            "avg_look_duration": round(summary["avg_look_duration"], 2),
            "best_ad_title": summary["best_ad_title"],
        }

    def get_filtered_report_summary(
        self,
        period: str = "daily",
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> dict:
        summary = self.dashboard_repo.get_report_summary(
            period=period,
            date_from=date_from,
            date_to=date_to,
        )

        return {
            "total_plays": summary["total_plays"],
            "total_viewers": summary["total_viewers"],
            "avg_look_duration": round(summary["avg_look_duration"], 2),
            "best_ad_title": summary["best_ad_title"],
        }

    def get_report_rows(
        self,
        limit: int = 100,
        period: str = "daily",
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[dict]:
        report_rows = self.dashboard_repo.get_report_rows(
            limit=limit,
            period=period,
            date_from=date_from,
            date_to=date_to,
        )

        return [
            {
                "stats_date": row["stats_date"],
                "ad_title": row["ad_title"],
                "category_name": row["category_name"],
                "segment_label": row["segment_label"],
                "play_count": row["play_count"],
                "viewer_count": row["viewer_count"],
                "avg_look_duration": round(row["avg_look_duration"], 2),
            }
            for row in report_rows
        ]

    def _format_top_ads(self, top_ads: list[dict]) -> list[dict]:
        return [
            {
                "ad_id": ad["ad_id"],
                "title": ad["title"],
                "total_plays": ad["total_plays"],
                "total_viewers": ad["total_viewers"],
                "avg_look_duration": round(ad["avg_look_duration"], 2),
            }
            for ad in top_ads
        ]

    def _format_recent_logs(self, recent_logs: list[dict]) -> list[dict]:
        formatted_logs = []

        for log in recent_logs:
            played_at = log["played_at"]
            played_at_text = (
                played_at.strftime("%Y-%m-%d %H:%M:%S")
                if played_at is not None
                else ""
            )

            formatted_logs.append(
                {
                    "log_id": log["log_id"],
                    "played_at": played_at,
                    "played_at_text": played_at_text,
                    "ad_title": log["ad_title"],
                    "total_viewers": log["total_viewers"],
                    "avg_look_duration": round(log["avg_look_duration"], 2),
                    "dominant_segment": log["dominant_segment"],
                }
            )

        return formatted_logs

    def _format_top_categories(self, top_categories: list[dict]) -> list[dict]:
        return [
            {
                "category_id": category["category_id"],
                "category_name": category["category_name"],
                "total_plays": category["total_plays"],
                "total_viewers": category["total_viewers"],
                "avg_look_duration": round(category["avg_look_duration"], 2),
            }
            for category in top_categories
        ]

    def _format_top_segments(self, top_segments: list[dict]) -> list[dict]:
        return [
            {
                "gender": segment["gender"],
                "age_group": segment["age_group"],
                "current_score": round(segment["current_score"], 4),
            }
            for segment in top_segments
        ]

    def _validate_advertisement_payload(
        self,
        title: str,
        description: str | None,
        media_filename: str,
        duration_seconds: int,
        category_id: int,
    ) -> dict:
        normalized_title = title.strip()
        normalized_filename = media_filename.strip()
        normalized_description = description.strip() if description else None

        if not normalized_title:
            raise ValueError("title is required")

        if not normalized_filename:
            raise ValueError("media_filename is required")

        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than 0")

        category = self.category_repo.get_by_id(category_id)
        if category is None:
            raise ValueError("category not found")

        return {
            "title": normalized_title,
            "description": normalized_description,
            "media_filename": normalized_filename,
            "duration_seconds": duration_seconds,
            "category_id": category.id,
        }
