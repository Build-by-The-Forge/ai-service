from sqlalchemy import Boolean, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AnswerSheet(Base):
    __tablename__ = "answer_sheets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    school_id: Mapped[int] = mapped_column(Integer, ForeignKey("schools.id"), index=True)
    session_id: Mapped[int] = mapped_column(Integer, index=True)
    question_id: Mapped[int] = mapped_column(Integer)
    selected_answer: Mapped[str] = mapped_column(String(10))
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
