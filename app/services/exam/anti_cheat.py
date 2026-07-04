from typing import Any


class AntiCheatEngine:
    def detect_abnormal_activity(self, session_data: Any) -> list[str]:
        flags: list[str] = []

        if not session_data:
            return flags

        if len(session_data.answers) > 50:
            flags.append("rapid_answering")

        if session_data.last_activity and session_data.last_activity < 2:
            flags.append("fast_switching")

        return flags
