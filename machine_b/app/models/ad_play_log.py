from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.advertisement import Advertisement
    from app.models.audience_segment import AudienceSegment


LOG_ID_TYPE = BigInteger().with_variant(Integer, "sqlite")


class AdPlayLog(Base):
    __tablename__ = "ad_play_logs"

    __table_args__ = (
        CheckConstraint("total_viewers >= 0", name="ck_ad_play_logs_total_viewers_non_negative"),
        CheckConstraint("avg_look_duration >= 0", name="ck_ad_play_logs_avg_look_duration_non_negative"),
    )
    
    id: Mapped[int] = mapped_column(
        LOG_ID_TYPE,
        primary_key=True,
        autoincrement=True,
    )
    advertisement_id: Mapped[int] = mapped_column(
        ForeignKey("advertisements.id"),
        nullable=False,
        index=True,
    )
    played_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    total_viewers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_look_duration: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    dominant_audience_segment_id: Mapped[int | None] = mapped_column(
    ForeignKey("audience_segments.id"),
    nullable=True,
    index=True,
)

    advertisement: Mapped["Advertisement"] = relationship(
        "Advertisement",
        back_populates="ad_play_logs",
    )
    dominant_audience_segment: Mapped["AudienceSegment | None"] = relationship(
        "AudienceSegment",
        back_populates="ad_play_logs",
    )
