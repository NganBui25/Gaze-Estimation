from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.advertisement import Advertisement

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
