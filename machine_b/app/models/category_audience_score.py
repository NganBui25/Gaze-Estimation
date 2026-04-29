from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.category import Category
    from app.models.audience_segment import AudienceSegment


class CategoryAudienceScore(Base):
    __tablename__ = "category_audience_scores"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id"),
        nullable=False,
        index=True,
    )
    audience_segment_id: Mapped[int] = mapped_column(
        ForeignKey("audience_segments.id"),
        nullable=False,
        index=True,
    )
    initial_score: Mapped[float] = mapped_column(Float, nullable=False)
    current_score: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    category: Mapped["Category"] = relationship(
        "Category",
        back_populates="category_audience_scores",
    )
    audience_segment: Mapped["AudienceSegment"] = relationship(
        "AudienceSegment",
        back_populates="category_audience_scores",
    )
