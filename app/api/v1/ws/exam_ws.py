import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.exam.session_state import SessionStore
from app.services.exam.anti_cheat import AntiCheatEngine

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

manager = ConnectionManager()
store = SessionStore()
anti_cheat = AntiCheatEngine()

@router.websocket("/ws/exam/{session_id}")
async def exam_websocket(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    session = store.get(session_id) or store.create(session_id)

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            event = payload.get("event")

            if event == "ping":
                await manager.send(session_id, {
                    "event": "pong",
                    "timestamp": time.time(),
                })

            elif event == "answer":
                question_id = payload.get("question_id")
                answer = payload.get("answer")
                store.update_answer(session_id, question_id, answer)

                await manager.send(session_id, {
                    "event": "answer_received",
                    "question_id": question_id,
                    "status": "saved",
                    "answers": session.answers,
                })

            elif event == "sync_time":
                remaining_seconds = store.get_remaining_time(session_id)
                await manager.send(session_id, {
                    "event": "time_sync",
                    "server_time": time.time(),
                    "remaining_seconds": remaining_seconds,
                })

            elif event == "check_behavior":
                session_data = store.get(session_id)
                flags = anti_cheat.detect_abnormal_activity(session_data)
                await manager.send(session_id, {
                    "event": "anti_cheat_report",
                    "flags": flags,
                })

            else:
                await manager.send(session_id, {
                    "event": "error",
                    "message": f"Unsupported event: {event}",
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
