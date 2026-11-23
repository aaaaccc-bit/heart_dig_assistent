"""SQLite-backed distributed task queue with leases and retries."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping
from uuid import uuid4


LOGGER = logging.getLogger(__name__)


class QueueError(RuntimeError):
    """Base exception for queue state errors."""


class QueueOwnershipError(QueueError):
    """Raised when a worker acknowledges a task it does not own."""


@dataclass(frozen=True)
class QueueTask:
    id: str
    kind: str
    payload: dict[str, Any]
    status: str
    attempts: int
    priority: int
    lease_owner: str | None
    lease_until: float | None
    created_at: float
    updated_at: float
    result: dict[str, Any] | None = None
    last_error: str | None = None


def _default_queue_path() -> Path:
    configured = os.getenv("CARDIO_TASK_QUEUE_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "CardioScope" / "tasks.sqlite3"
    return Path.home() / ".cache" / "cardioscope" / "tasks.sqlite3"


class DistributedTaskQueue:
    """A lease queue usable by multiple processes sharing the SQLite file."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path is not None else _default_queue_path()
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=5)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_queue (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    priority INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_until REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    result_json TEXT,
                    last_error TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_queue_claim
                ON task_queue(status, available_at, priority DESC, created_at)
                """
            )

    def enqueue(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        priority: int = 0,
        delay_s: float = 0.0,
        task_id: str | None = None,
    ) -> str:
        if not kind.strip():
            raise ValueError("任务类型不能为空")
        task_id = task_id or uuid4().hex
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO task_queue
                (id, kind, payload_json, status, attempts, priority,
                 available_at, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    kind.strip(),
                    json.dumps(dict(payload), ensure_ascii=False, default=str),
                    int(priority),
                    now + max(0.0, float(delay_s)),
                    now,
                    now,
                ),
            )
        return task_id

    def claim(self, worker_id: str | None = None, *, lease_s: float = 300.0) -> QueueTask | None:
        owner = worker_id or uuid4().hex
        now = time.time()
        lease_until = now + max(1.0, float(lease_s))
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM task_queue
                WHERE (status = 'pending' AND available_at <= ?)
                   OR (status = 'running' AND lease_until < ?)
                ORDER BY priority DESC, created_at
                LIMIT 1
                """,
                (now, now),
            ).fetchone()
            if row is None:
                return None
            attempts = int(row["attempts"]) + 1
            connection.execute(
                """
                UPDATE task_queue
                SET status = 'running', attempts = ?, lease_owner = ?,
                    lease_until = ?, updated_at = ?, last_error = NULL
                WHERE id = ?
                """,
                (attempts, owner, lease_until, now, row["id"]),
            )
            updated = connection.execute(
                "SELECT * FROM task_queue WHERE id = ?", (row["id"],)
            ).fetchone()
        return self._task_from_row(updated)

    def ack(
        self,
        task_id: str,
        worker_id: str,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE task_queue
                SET status = 'completed', lease_owner = NULL, lease_until = NULL,
                    updated_at = ?, result_json = ?, last_error = NULL
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                """,
                (
                    now,
                    json.dumps(dict(result or {}), ensure_ascii=False, default=str),
                    task_id,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise QueueOwnershipError(f"Worker 无权确认任务：{task_id}")

    def fail(
        self,
        task_id: str,
        worker_id: str,
        error: str,
        *,
        retry_delay_s: float = 5.0,
        max_attempts: int = 3,
    ) -> str:
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM task_queue WHERE id = ? AND lease_owner = ? AND status = 'running'",
                (task_id, worker_id),
            ).fetchone()
            if row is None:
                raise QueueOwnershipError(f"Worker 无权失败标记任务：{task_id}")
            attempts = int(row["attempts"])
            status = "pending" if attempts < max(1, int(max_attempts)) else "failed"
            available_at = now + max(0.0, float(retry_delay_s)) if status == "pending" else now
            connection.execute(
                """
                UPDATE task_queue
                SET status = ?, available_at = ?, lease_owner = NULL, lease_until = NULL,
                    updated_at = ?, last_error = ?
                WHERE id = ?
                """,
                (status, available_at, now, str(error)[:2000], task_id),
            )
        return status

    def get(self, task_id: str) -> QueueTask | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_queue WHERE id = ?", (task_id,)
            ).fetchone()
        return self._task_from_row(row) if row is not None else None

    def stats(self) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM task_queue GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def _task_from_row(self, row: sqlite3.Row | None) -> QueueTask | None:
        if row is None:
            return None
        return QueueTask(
            id=str(row["id"]),
            kind=str(row["kind"]),
            payload=dict(json.loads(row["payload_json"])),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            priority=int(row["priority"]),
            lease_owner=row["lease_owner"],
            lease_until=row["lease_until"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            result=dict(json.loads(row["result_json"])) if row["result_json"] else None,
            last_error=row["last_error"],
        )


class QueueWorker:
    """Claim one task, execute it, and commit success or retry state."""

    def __init__(
        self,
        queue: DistributedTaskQueue,
        *,
        worker_id: str | None = None,
        lease_s: float = 300.0,
        max_attempts: int = 3,
    ):
        self.queue = queue
        self.worker_id = worker_id or uuid4().hex
        self.lease_s = lease_s
        self.max_attempts = max_attempts

    def run_once(self, handler: Callable[[QueueTask], Mapping[str, Any] | None]) -> bool:
        task = self.queue.claim(self.worker_id, lease_s=self.lease_s)
        if task is None:
            return False
        try:
            result = handler(task)
        except Exception as exc:  # noqa: BLE001 - worker boundary records task failures
            status = self.queue.fail(
                task.id,
                self.worker_id,
                str(exc),
                max_attempts=self.max_attempts,
            )
            LOGGER.warning("队列任务失败（%s，状态=%s）：%s", task.id, status, exc)
        else:
            self.queue.ack(task.id, self.worker_id, result=result)
        return True


_SHARED_QUEUE: DistributedTaskQueue | None = None
_SHARED_QUEUE_LOCK = RLock()


def get_shared_task_queue() -> DistributedTaskQueue:
    global _SHARED_QUEUE
    with _SHARED_QUEUE_LOCK:
        if _SHARED_QUEUE is None:
            _SHARED_QUEUE = DistributedTaskQueue()
        return _SHARED_QUEUE


__all__ = [
    "DistributedTaskQueue",
    "QueueError",
    "QueueOwnershipError",
    "QueueTask",
    "QueueWorker",
    "get_shared_task_queue",
]
