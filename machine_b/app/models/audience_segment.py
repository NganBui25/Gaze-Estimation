from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.category_audience_score import CategoryAudienceScore
    from app.models.ad_play_log import AdPlayLog
    from app.models.ad_performance_summary import AdPerformanceSummary


class AudienceSegment(Base):
    __tablename__ = "audience_segments"

    __table_args__ = (
        UniqueConstraint(
            "gender",
            "age_min",
            "age_max",
            name="uq_audience_segments_gender_age_range",
        ),
        CheckConstraint("age_min >= 0", name="ck_audience_segments_age_min_non_negative"),
        CheckConstraint("age_max >= age_min", name="ck_audience_segments_age_range_valid"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    gender: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    age_group: Mapped[str] = mapped_column(String(50), nullable=False)
    age_min: Mapped[int] = mapped_column(Integer, nullable=False)
    age_max: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    category_audience_scores: Mapped[List["CategoryAudienceScore"]] = relationship(
        "CategoryAudienceScore",
        back_populates="audience_segment",
    )

    ad_play_logs: Mapped[List["AdPlayLog"]] = relationship(
        "AdPlayLog",
        back_populates="dominant_audience_segment",
    )

    ad_performance_summaries: Mapped[List["AdPerformanceSummary"]] = relationship(
        "AdPerformanceSummary",
        back_populates="audience_segment",
    )
