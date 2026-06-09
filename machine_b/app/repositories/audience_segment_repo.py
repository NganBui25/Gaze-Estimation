from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.audience_segment import AudienceSegment

class AudienceSegmentRepo:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, segment_id: int):
        stmt = select(AudienceSegment).where(AudienceSegment.id == segment_id)
        return self.db.execute(stmt).scalar_one_or_none()
    
    def find_all_by_gender(self, gender: str) -> list[AudienceSegment]:
        stmt = (
            select(AudienceSegment)
            .where(AudienceSegment.gender == gender)
            .order_by(AudienceSegment.age_min.asc())
        )
        return list(self.db.execute(stmt).scalars().all())

    def find_all(self) -> list[AudienceSegment]:
        stmt = select(AudienceSegment).order_by(
            AudienceSegment.gender.asc(),
            AudienceSegment.age_min.asc(),
        )
        return list(self.db.execute(stmt).scalars().all())
