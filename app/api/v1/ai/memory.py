"""Learning memory (M5 design §3, ruling R3).

Rows are the signed user's in the signed tenant; the body cannot name
either. The backend reports graded outcomes after an attempt is submitted
(it checked assessment.attempt.submit) and lets a member read their own
mastery (ai.use).
"""

from fastapi import APIRouter, Depends

from app.core.service_call import signed_user_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.db.models.learning_memory import LearningMemory
from app.schemas.ai import (
    MemoryData,
    MemoryItem,
    MemoryQueryRequest,
    MemoryResponse,
    MemoryUpsertRequest,
)
from app.services.ai.dependencies import get_db_session
from app.services.memory.memory_service import MemoryService

router = APIRouter()

MEMORY_RECORD_PERMISSION = "assessment.attempt.submit"


async def get_memory_service(db=Depends(get_db_session)) -> MemoryService:
    return MemoryService(db)


def _data(rows: list[LearningMemory]) -> MemoryData:
    return MemoryData(
        items=[
            MemoryItem(topic=r.topic, mastery_score=round(r.mastery_score, 4), attempts=r.attempts)
            for r in rows
        ]
    )


@router.post("/memory/query", response_model=MemoryResponse)
async def query_memory(
    body: MemoryQueryRequest,
    call: VerifiedCall = Depends(signed_user_call_with("ai.use")),
    memory: MemoryService = Depends(get_memory_service),
) -> MemoryResponse:
    rows = await memory.query(call.context.user_id or "", body.topics)
    return MemoryResponse(data=_data(rows), message="Memory retrieved")


@router.post("/memory/upsert", response_model=MemoryResponse)
async def record_memory(
    body: MemoryUpsertRequest,
    call: VerifiedCall = Depends(signed_user_call_with(MEMORY_RECORD_PERMISSION)),
    memory: MemoryService = Depends(get_memory_service),
) -> MemoryResponse:
    rows = await memory.record(
        call.context.user_id or "", [(o.topic, o.correct) for o in body.observations]
    )
    return MemoryResponse(data=_data(rows), message="Memory updated")
