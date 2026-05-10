from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Date, Float, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.advertisement import Advertisement
    from app.models.audience_segment import AudienceSegment


class AdPerformanceSummary(Base):
    __tablename__ = "ad_performance_summary"

    __table_args__ = (
        UniqueConstraint(
            "advertisement_id",
            "audience_segment_id",
            "stats_date",
            name="uq_ad_performance_summary_ad_segment_date",
        ),
        CheckConstraint("play_count >= 0", name="ck_ad_performance_summary_play_count_non_negative"),
        CheckConstraint("viewer_count >= 0", name="ck_ad_performance_summary_viewer_count_non_negative"),
        CheckConstraint("avg_look_duration >= 0", name="ck_ad_performance_summary_avg_look_duration_non_negative"),
    )
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    advertisement_id: Mapped[int] = mapped_column(
        ForeignKey("advertisements.id"),
        nullable=False,
        index=True,
    )
    audience_segment_id: Mapped[int] = mapped_column(
        ForeignKey("audience_segments.id"),
        nullable=False,
        index=True,
    )
    stats_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    play_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    viewer_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_look_duration: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    created_at: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        server_default=func.current_date(),
    )

    advertisement: Mapped["Advertisement"] = relationship(
        "Advertisement",
        back_populates="ad_performance_summaries",
    )
    audience_segment: Mapped["AudienceSegment"] = relationship(
        "AudienceSegment",
        back_populates="ad_performance_summaries",
    )
