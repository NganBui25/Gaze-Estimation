import random

from sqlalchemy.orm import Session

from app.models.advertisement import Advertisement
from app.repositories.advertisement_repo import AdvertisementRepo
from app.services.audience_segment_service import AudienceSegmentService
from app.services.category_audience_score_service import CategoryAudienceScoreService


class AdvertisementService:
    def __init__(self, db: Session):
        self.db = db
        self.advertisement_repo = AdvertisementRepo(db)
        self.audience_segment_service = AudienceSegmentService(db)
        self.category_audience_score_service = CategoryAudienceScoreService(db)

    def get_by_id(self, ad_id: int) -> Advertisement | None:
        return self.advertisement_repo.get_by_id(ad_id)

    def get_active_ads_by_category_id(self, category_id: int) -> list[Advertisement]:
        return self.advertisement_repo.find_active_by_category_id(category_id)

    def get_random_active_ad_by_category_id(self, category_id: int) -> Advertisement:
        advertisements = self.get_active_ads_by_category_id(category_id)

        if not advertisements:
            raise LookupError("no active advertisement found for this category")

        return random.choice(advertisements)

    def get_random_active_ad(self) -> Advertisement:
        advertisements = self.advertisement_repo.find_active()

        if not advertisements:
            raise LookupError("no active advertisement found")

        return random.choice(advertisements)

    def select_ad(self, viewer_count: int, avg_age: int, majority_gender: str) -> Advertisement:
        if viewer_count <= 0:
            try:
                category_id = self.category_audience_score_service.get_lowest_average_score_active_category_id()
            except LookupError:
                return self.get_random_active_ad()

            return self.get_random_active_ad_by_category_id(category_id)

        audience_segment = self.audience_segment_service.get_segment_by_age_and_gender(
            avg_age=avg_age,
            gender=majority_gender,
        )

        category_id = self.category_audience_score_service.get_best_category_id_by_audience_segment_id(
            audience_segment_id=audience_segment.id,
        )

        advertisement = self.get_random_active_ad_by_category_id(
            category_id=category_id,
        )

        return advertisement
