from sqlalchemy.orm import Session

from app.models.audience_segment import AudienceSegment
from app.repositories.audience_segment_repo import AudienceSegmentRepo

class AudienceSegmentService:
    def __init__(self, db: Session):
        self.db = db
        self.audience_segment_repo = AudienceSegmentRepo(db)

    def get_by_id(self, segment_id: int) -> AudienceSegment | None:
        return self.audience_segment_repo.get_by_id(segment_id)

    def get_required_by_id(self, segment_id: int) -> AudienceSegment:
        segment = self.get_by_id(segment_id)
        if segment is None:
            raise LookupError("audience segment not found")
        return segment
