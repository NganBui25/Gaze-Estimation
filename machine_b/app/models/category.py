from datetime import datetime
from typing import List, TYPE_CHECKING

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.advertisement import Advertisement
    from app.models.category_audience_score import CategoryAudienceScore

class Category(Base):
    __tablename__ = "categories"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    #ĐỊnh nghĩa quan hệ 1-nhiều
    advertisements: Mapped[List["Advertisement"]] = relationship(
        "Advertisement",
        back_populates="category",
    )
    category_audience_scores: Mapped[List["CategoryAudienceScore"]] = relationship(
        "CategoryAudienceScore",
        back_populates="category",
    )