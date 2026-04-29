from datetime import datetime
from typing import TYPE_CHECKING, List, List

from sqlalchemy import Boolean, CheckConstraint, CheckConstraint, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.category import Category
    from app.models.ad_play_log import AdPlayLog
    from app.models.ad_performance_summary import AdPerformanceSummary


class Advertisement(Base):
    __tablename__ = "advertisements"

    __table_args__ = (
        CheckConstraint("duration_seconds > 0", name="ck_advertisements_duration_seconds_positive"),
    )
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    category: Mapped["Category"] = relationship(
        "Category",
        back_populates="advertisements",
    )

    ad_play_logs: Mapped[List["AdPlayLog"]] = relationship(
        "AdPlayLog",
        back_populates="advertisement",
    )

    ad_performance_summaries: Mapped[List["AdPerformanceSummary"]] = relationship(
        "AdPerformanceSummary",
        back_populates="advertisement",
    )
