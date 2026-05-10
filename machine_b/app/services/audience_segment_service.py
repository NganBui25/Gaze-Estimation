from sqlalchemy.orm import Session

from app.models.audience_segment import AudienceSegment
from app.repositories.audience_segment_repo import AudienceSegmentRepo

class AudienceSegmentService:
    ALLOWED_GENDERS = {"male", "female", "unknown"}
    def __init__(self, db: Session):
        self.db = db
        self.audience_segment_repo = AudienceSegmentRepo(db)

    def normalize_gender(self, gender: str):
        nomalized_gender = gender.strip().lower()
        if nomalized_gender not in self.ALLOWED_GENDERS:
            raise ValueError("gender is invalid")
        return nomalized_gender
    
    def get_by_id(self, segment_id: int) -> AudienceSegment | None:
        return self.audience_segment_repo.get_by_id(segment_id)
    
    def get_segment_by_age_and_gender( self, avg_age: int, gender: str):
        if avg_age < 0:
            raise ValueError("avg_age must be greater than or equal to 0")
        
        normalized_gender = self.normalize_gender(gender)

        segment = self.audience_segment_repo.find_by_age_and_gender(avg_age=avg_age, gender=normalized_gender,)
        if segment is None:
            raise LookupError("audience segment not found")

        return segment
    
    def get_segment_id_by_age_and_gender(
        self,
        avg_age: int,
        gender: str,
    ) -> int:
        segment = self.get_segment_by_age_and_gender(
            avg_age=avg_age,
            gender=gender,
        )
        return segment.id
