from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.ad_performance_summary import AdPerformanceSummary
from app.models.advertisement import Advertisement
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

    def find_lowest_average_score_active_category_id(self) -> int | None:
        stmt = (
            select(
                CategoryAudienceScore.category_id,
                func.avg(CategoryAudienceScore.current_score).label("avg_score"),
            )
            .join(
                Advertisement,
                Advertisement.category_id == CategoryAudienceScore.category_id,
            )
            .where(Advertisement.is_active.is_(True))
            .group_by(CategoryAudienceScore.category_id)
            .order_by(
                func.avg(CategoryAudienceScore.current_score).asc(),
                CategoryAudienceScore.category_id.asc(),
            )
            .limit(1)
        )
        row = self.db.execute(stmt).first()
        return None if row is None else int(row.category_id)

    def find_average_scores_for_active_categories(self) -> list[tuple[int, float]]:
        stmt = (
            select(
                CategoryAudienceScore.category_id,
                func.avg(CategoryAudienceScore.current_score).label("avg_score"),
            )
            .join(
                Advertisement,
                Advertisement.category_id == CategoryAudienceScore.category_id,
            )
            .where(Advertisement.is_active.is_(True))
            .group_by(CategoryAudienceScore.category_id)
            .order_by(CategoryAudienceScore.category_id.asc())
        )
        rows = self.db.execute(stmt).all()
        return [
            (int(row.category_id), float(row.avg_score or 0.0))
            for row in rows
        ]

    def get_historical_viewer_count(
        self,
        category_id: int,
        audience_segment_id: int,
    ) -> int:
        stmt = (
            select(func.coalesce(func.sum(AdPerformanceSummary.viewer_count), 0))
            .join(
                Advertisement,
                Advertisement.id == AdPerformanceSummary.advertisement_id,
            )
            .where(
                Advertisement.category_id == category_id,
                AdPerformanceSummary.audience_segment_id == audience_segment_id,
            )
        )
        return int(self.db.execute(stmt).scalar_one() or 0)
    
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
