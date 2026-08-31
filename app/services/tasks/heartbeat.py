import time
from dataclasses import dataclass


@dataclass
class LeaseRecord:
    execution_id: str
    worker_id: str
    last_heartbeat: float
    lease_until: float


class HeartbeatManager:
    """
    Manages worker execution leases and heartbeats.
    Ensures long-running jobs (OCR, PDF ingestion) maintain active leases and can be recovered if workers crash.
    """

    def __init__(self):
        self._leases: dict[str, LeaseRecord] = {}

    def record_heartbeat(
        self, execution_id: str, worker_id: str, lease_seconds: int = 30
    ) -> LeaseRecord:
        now = time.time()
        record = LeaseRecord(
            execution_id=execution_id,
            worker_id=worker_id,
            last_heartbeat=now,
            lease_until=now + lease_seconds,
        )
        self._leases[execution_id] = record
        return record

    def is_lease_valid(self, execution_id: str) -> bool:
        record = self._leases.get(execution_id)
        if not record:
            return False
        return time.time() < record.lease_until

    def release(self, execution_id: str) -> None:
        self._leases.pop(execution_id, None)

    def get_lease(self, execution_id: str) -> LeaseRecord | None:
        return self._leases.get(execution_id)


_default_heartbeat_manager: HeartbeatManager | None = None


def get_heartbeat_manager() -> HeartbeatManager:
    global _default_heartbeat_manager
    if _default_heartbeat_manager is None:
        _default_heartbeat_manager = HeartbeatManager()
    return _default_heartbeat_manager
