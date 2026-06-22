from sqlalchemy import String, Integer, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class ExamSession(Base):
    __tablename__ = "exam_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    school_id: Mapped[int] = mapped_column(Integer, ForeignKey("schools.id"), index=True)
    exam_id: Mapped[int] = mapped_column(Integer, ForeignKey("exams.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(50), default="active")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    current_index: Mapped[int] = mapped_column(Integer, default=0)
