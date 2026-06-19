import asyncio
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.tenant import set_school_id
from app.services.exam.redis_session_store import RedisSessionStore
from app.services.exam.redis_timer import RedisExamTimer
from app.services.exam.anti_cheat import AntiCheatEngine
from app.services.realtime.sync_bridge import SyncBridge

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send(self, session_id: str, message: dict):
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_text(json.dumps(message))

    async def broadcast(self, message: dict):
        for ws in list(self.active_connections.values()):
            await ws.send_text(json.dumps(message))

manager = ConnectionManager()
store = RedisSessionStore()
timer = RedisExamTimer()
anti_cheat = AntiCheatEngine()
sync_bridge = SyncBridge()

async def websocket_sync_handler(session_id: str, payload: dict):
    event = payload.get("event")
    if event == "answer_sync":
        await manager.send(session_id, {
            "event": "answer_sync",
            "data": payload.get("data"),
        })

@router.websocket("/ws/exam/{session_id}")
async def exam_websocket(websocket: WebSocket, session_id: str):
    school_id = websocket.headers.get("x-school-id")
    if not school_id:
        await websocket.close(code=1008)
        return

    set_school_id(int(school_id))
    scoped_session_id = f"{school_id}:{session_id}"

    await manager.connect(websocket, scoped_session_id)

    async def subscribe_to_session():
        await sync_bridge.pubsub.subscribe(
            channel=f"exam:{school_id}:{session_id}",
            handler=lambda message: websocket_sync_handler(scoped_session_id, message),
        )

    sub_task = asyncio.create_task(subscribe_to_session())

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            event = payload.get("event")

            if event == "init_session":
                await store.create_session(scoped_session_id, {
                    "user_id": payload.get("user_id"),
                    "answers": {},
                    "start_time": time.time(),
                })
                await timer.start(scoped_session_id, 30)

                await manager.send(scoped_session_id, {
                    "event": "session_initialized",
                    "session_id": session_id,
                    "school_id": int(school_id),
                })

            elif event == "answer":
                await store.update_answer(
                    session_id=scoped_session_id,
                    question_id=payload.get("question_id"),
                    answer=payload.get("answer"),
                )
                await sync_bridge.broadcast_answer(scoped_session_id, {
                    "question_id": payload.get("question_id"),
                    "answer": payload.get("answer"),
                    "user_id": payload.get("user_id"),
                    "timestamp": time.time(),
                })

                await manager.send(scoped_session_id, {
                    "event": "answer_saved",
                    "question_id": payload.get("question_id"),
                })

            elif event == "start_timer":
                await timer.start(scoped_session_id, duration=payload.get("duration", 30))
                await manager.send(scoped_session_id, {
                    "event": "timer_started",
                    "duration_minutes": payload.get("duration", 30),
                })

            elif event == "get_time":
                remaining_seconds = await timer.remaining(scoped_session_id)
                await manager.send(scoped_session_id, {
                    "event": "time_update",
                    "remaining_seconds": remaining_seconds,
                })

            elif event == "check_behavior":
                session_data = await store.get_session(scoped_session_id)
                flags = anti_cheat.detect_abnormal_activity(session_data)
                await manager.send(scoped_session_id, {
                    "event": "anti_cheat_report",
                    "flags": flags,
                })

            elif event == "ping":
                await manager.send(scoped_session_id, {
                    "event": "pong",
                    "timestamp": time.time(),
                })

            else:
                await manager.send(scoped_session_id, {
                    "event": "error",
                    "message": f"Unsupported event: {event}",
                })

    except WebSocketDisconnect:
        manager.disconnect(scoped_session_id)
        sub_task.cancel()
