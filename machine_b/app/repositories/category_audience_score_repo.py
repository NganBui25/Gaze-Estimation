from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.category_audience_score import CategoryAudienceScore

class CategoryAudienceScoreRepo:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, score_id: int) -> CategoryAudienceScore | None:
        stmt = select(CategoryAudienceScore).where(CategoryAudienceScore.id == score_id)
        return self.db.execute(stmt).scalar_one_or_none()
    
    def get_by_category_id_and_audience_segment_id(self, category_id: int, audience_segment_id: int,):
        stmt = (
            select(CategoryAudienceScore)
            .where(
                CategoryAudienceScore.category_id == category_id,
                CategoryAudienceScore.audience_segment_id == audience_segment_id,
            )
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()
    def find_all_by_audience_segment_id(self, audience_segment_id: int) -> list[CategoryAudienceScore]:
        stmt = (
            select(CategoryAudienceScore)
            .where(CategoryAudienceScore.audience_segment_id == audience_segment_id)
            .order_by(
                CategoryAudienceScore.current_score.desc(),
                CategoryAudienceScore.id.asc(),
            )
        )
        return list(self.db.execute(stmt).scalars().all())
    
    def find_best_by_audience_segment_id(self, audience_segment_id: int) -> CategoryAudienceScore | None:
        stmt = (
            select(CategoryAudienceScore)
            .where(CategoryAudienceScore.audience_segment_id == audience_segment_id)
            .order_by(
                CategoryAudienceScore.current_score.desc(),
                CategoryAudienceScore.id.asc(),
            )
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()
    
    def create(self, category_id: int, audience_segment_id: int, initial_score: float, current_score: float):
        category_audience_score = CategoryAudienceScore(
            category_id = category_id,
            audience_segment_id = audience_segment_id,
            initial_score = initial_score,
            current_score = current_score,
        )

        self.db.add(category_audience_score)
        self.db.flush()

        return category_audience_score
