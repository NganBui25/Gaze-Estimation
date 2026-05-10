from sqlalchemy.orm import Session

from app.models.category_audience_score import CategoryAudienceScore
from app.repositories.category_audience_score_repo import CategoryAudienceScoreRepo


class CategoryAudienceScoreService:
    PRIOR_WEIGHT = 10.0
    MAX_VIEWER_WEIGHT = 5.0

    def __init__(self, db: Session):
        self.db = db
        self.category_audience_score_repo = CategoryAudienceScoreRepo(db)

    def get_all_by_audience_segment_id(
        self,
        audience_segment_id: int,
    ) -> list[CategoryAudienceScore]:
        return self.category_audience_score_repo.find_all_by_audience_segment_id(
            audience_segment_id=audience_segment_id,
        )

    def get_best_score_by_audience_segment_id(
        self,
        audience_segment_id: int,
    ) -> CategoryAudienceScore:
        best_score = self.category_audience_score_repo.find_best_by_audience_segment_id(
            audience_segment_id=audience_segment_id,
        )

        if best_score is None:
            raise LookupError("category audience score not found")

        return best_score

    def get_best_category_id_by_audience_segment_id(
        self,
        audience_segment_id: int,
    ) -> int:
        best_score = self.get_best_score_by_audience_segment_id(
            audience_segment_id=audience_segment_id,
        )
        return best_score.category_id
    
    def calculator_actual_score(self, viewer_count: int, total_watch_duration: int, ad_duration_seconds: int,):
        if viewer_count <= 0:
            raise ValueError("viewer_count must be greater than 0")

        if total_watch_duration < 0:
            raise ValueError("total_watch_duration must be greater than or equal to 0")

        if ad_duration_seconds <= 0:
            raise ValueError("ad_duration_seconds must be greater than 0")
 
        avg_watch_duration = total_watch_duration/viewer_count
        watch_ratio = min(avg_watch_duration/ad_duration_seconds, 1.0)

        return round(watch_ratio, 4)
    
    def update_current_score(self, category_id: int, audience_segment_id: int, viewer_count: int, total_watch_duration: float, ad_duration_seconds: int,):
        actual_score = self.calculator_actual_score(viewer_count=viewer_count, total_watch_duration=total_watch_duration, ad_duration_seconds=ad_duration_seconds,)
        category_audience_score = (
            self.category_audience_score_repo.get_by_category_id_and_audience_segment_id(
                category_id=category_id,
                audience_segment_id=audience_segment_id,
            )
        )

        if category_audience_score is None:
            return self.category_audience_score_repo.create(
                category_id=category_id,
                audience_segment_id=audience_segment_id,
                initial_score=0.0,
                current_score=actual_score,
            )
        
        viewer_weight = min(float(viewer_count), self.MAX_VIEWER_WEIGHT)

        new_current_score = (
             (category_audience_score.current_score * self.PRIOR_WEIGHT)
            + (actual_score * viewer_weight)
        ) / (self.PRIOR_WEIGHT + viewer_weight)

        category_audience_score.current_score = round(new_current_score, 4)
        self.db.flush()

        return category_audience_score