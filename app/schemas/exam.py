from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ExamBase(BaseModel):
    title: str
    course: str
    duration_minutes: int
    is_active: bool = True


class ExamCreate(ExamBase):
    pass


class ExamUpdate(BaseModel):
    title: Optional[str] = None
    course: Optional[str] = None
    duration_minutes: Optional[int] = None
    is_active: Optional[bool] = None


class ExamResponse(ExamBase):
    id: int
    school_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
