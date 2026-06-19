from sqlalchemy import String, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    course: Mapped[str] = mapped_column(String(255))
    topic: Mapped[str] = mapped_column(String(255))
    question_data: Mapped[dict] = mapped_column(JSON)
