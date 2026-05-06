from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.category import Category

class CategoryRepo:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, category_id: int):
        stmt = select(Category).where(Category.id == category_id)
        return self.db.execute(stmt).scalar_one_or_none()
    
    def find_all(self) -> list[Category]:
        stmt = select(Category).order_by(Category.id.asc())
        return list(self.db.execute(stmt).scalars().all())
    
    def get_by_name(self, name: str) -> Category | None:
        stmt = select(Category).where(Category.name == name)
        return self.db.execute(stmt).scalar_one_or_none()
    
    