from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ExamBase(BaseModel):
    title: str
    course: str
    duration_minutes: int
    is_active: bool = True


class ExamCreate(ExamBase):
    pass


class ExamUpdate(BaseModel):
    title: str | None = None
    course: str | None = None
    duration_minutes: int | None = None
    is_active: bool | None = None


class ExamResponse(ExamBase):
    id: int
    school_id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
