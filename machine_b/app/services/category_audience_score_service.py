from sqlalchemy.orm import Session

from app.models.category_audience_score import CategoryAudienceScore
from app.repositories.category_audience_score_repo import CategoryAudienceScoreRepo


class CategoryAudienceScoreService:
    MAX_PRIOR_WEIGHT = 100.0
    MAX_VIEWER_WEIGHT = 20.0

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

    def get_lowest_average_score_active_category_id(self) -> int:
        category_id = self.category_audience_score_repo.find_lowest_average_score_active_category_id()
        if category_id is None:
            raise LookupError("no active category audience score found")
        return category_id
    
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

        historical_viewer_count = self.category_audience_score_repo.get_historical_viewer_count(
            category_id=category_id,
            audience_segment_id=audience_segment_id,
        )
        prior_weight = min(float(historical_viewer_count), self.MAX_PRIOR_WEIGHT)
        viewer_weight = min(float(viewer_count), self.MAX_VIEWER_WEIGHT)

        if prior_weight <= 0:
            new_current_score = actual_score
        else:
            new_current_score = (
                 (category_audience_score.current_score * prior_weight)
                + (actual_score * viewer_weight)
            ) / (prior_weight + viewer_weight)

        category_audience_score.current_score = round(new_current_score, 4)
        self.db.flush()

        return category_audience_score
