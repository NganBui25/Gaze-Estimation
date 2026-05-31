from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.advertisement import Advertisement
from app.models.ad_performance_summary import AdPerformanceSummary
from app.models.ad_play_log import AdPlayLog

class AdvertisementRepo:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, ad_id: int):
        stmt = select(Advertisement).where(Advertisement.id == ad_id)
        return self.db.execute(stmt).scalar_one_or_none()
    
    def find_active_by_category_id(self, category_id: int) -> list[Advertisement]:
        stmt = (
            select(Advertisement)
            .where(
                Advertisement.category_id == category_id,
                Advertisement.is_active.is_(True),
            )
            .order_by(Advertisement.id.asc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def find_active(self) -> list[Advertisement]:
        stmt = (
            select(Advertisement)
            .where(Advertisement.is_active.is_(True))
            .order_by(Advertisement.id.asc())
        )
        return list(self.db.execute(stmt).scalars().all())
    
    def find_active_by_category_id_and_media_filename(
        self,
        category_id: int,
        media_filename: str,
    ) -> Advertisement | None:
        stmt = (
            select(Advertisement)
            .where(
                Advertisement.category_id == category_id,
                Advertisement.media_filename == media_filename,
                Advertisement.is_active.is_(True),
            )
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def create(
        self,
        title: str,
        description: str | None,
        media_filename: str,
        duration_seconds: int,
        category_id: int,
        is_active: bool = True,
    ) -> Advertisement:
        advertisement = Advertisement(
            title=title,
            description=description,
            media_filename=media_filename,
            duration_seconds=duration_seconds,
            category_id=category_id,
            is_active=is_active,
        )
        self.db.add(advertisement)
        self.db.flush()
        return advertisement

    def delete(self, advertisement: Advertisement) -> None:
        self.db.delete(advertisement)
        self.db.flush()

    def has_related_activity(self, advertisement_id: int) -> bool:
        play_stmt = select(func.count(AdPlayLog.id)).where(
            AdPlayLog.advertisement_id == advertisement_id
        )
        summary_stmt = select(func.count(AdPerformanceSummary.id)).where(
            AdPerformanceSummary.advertisement_id == advertisement_id
        )

        play_count = self.db.execute(play_stmt).scalar_one()
        summary_count = self.db.execute(summary_stmt).scalar_one()

        return bool(play_count or summary_count)
