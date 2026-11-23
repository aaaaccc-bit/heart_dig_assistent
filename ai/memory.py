"""Durable task memory and cross-task experience retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Mapping


LOGGER = logging.getLogger(__name__)


def _default_memory_path() -> Path:
    configured = os.getenv("CARDIO_MEMORY_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "CardioScope" / "memory.sqlite3"
    return Path.home() / ".cache" / "cardioscope" / "memory.sqlite3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class MemoryEntry:
    id: int
    task_fingerprint: str
    task_type: str
    summary: str
    lessons: tuple[str, ...]
    metadata: dict[str, Any]
    success: bool
    created_at: str


@dataclass(frozen=True)
class ExperienceEntry:
    id: int
    task_type: str
    signal: str
    recommendation: str
    evidence: dict[str, Any]
    confidence: float
    created_at: str


class LongTermMemory:
    """SQLite-backed memory suitable for one process or shared local storage."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path is not None else _default_memory_path()
        self._lock = RLock()
        self.enabled = True
        self.error: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        except (OSError, sqlite3.Error) as exc:
            self.enabled = False
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("长期记忆不可用，将跳过持久化：%s", self.error)

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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS task_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_fingerprint TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    lessons_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_memory_lookup
                    ON task_memory(task_fingerprint, task_type, success, created_at DESC);
                CREATE TABLE IF NOT EXISTS task_experience (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_experience_lookup
                    ON task_experience(task_type, confidence DESC, created_at DESC);
                """
            )

    @staticmethod
    def fingerprint(record: Any) -> str:
        parameters = getattr(record, "parameters", {})
        payload = {
            "title": str(getattr(record, "title", "")),
            "kind": str(getattr(record, "kind_label", "")),
            "dataset": str(getattr(record, "dataset_name", "")),
            "trajectory_count": getattr(record, "trajectory_count", None),
            "group_count": getattr(record, "group_count", None),
            "parameter_sections": sorted(str(key) for key in parameters)
            if isinstance(parameters, Mapping)
            else [],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def task_type(record: Any) -> str:
        return str(getattr(record, "kind_label", "未分类任务"))

    def remember_task(
        self,
        *,
        task_fingerprint: str,
        task_type: str,
        summary: str,
        lessons: list[str] | tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        success: bool = True,
    ) -> int | None:
        if not self.enabled:
            return None
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO task_memory
                    (task_fingerprint, task_type, summary, lessons_json,
                     metadata_json, success, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_fingerprint,
                        task_type,
                        summary[:2000],
                        json.dumps(list(lessons)[:20], ensure_ascii=False, default=str),
                        json.dumps(dict(metadata or {}), ensure_ascii=False, default=str),
                        int(success),
                        _now(),
                    ),
                )
                return int(cursor.lastrowid)
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            self._disable_after_error(exc)
            return None

    def record_experience(
        self,
        *,
        task_type: str,
        signal: str,
        recommendation: str,
        evidence: Mapping[str, Any] | None = None,
        confidence: float = 0.5,
    ) -> int | None:
        if not self.enabled:
            return None
        try:
            with self._lock, self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO task_experience
                    (task_type, signal, recommendation, evidence_json,
                     confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_type,
                        signal[:200],
                        recommendation[:1000],
                        json.dumps(dict(evidence or {}), ensure_ascii=False, default=str),
                        max(0.0, min(1.0, float(confidence))),
                        _now(),
                    ),
                )
                return int(cursor.lastrowid)
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            self._disable_after_error(exc)
            return None

    def recall_tasks(
        self,
        *,
        task_fingerprint: str,
        task_type: str,
        limit: int = 3,
    ) -> list[MemoryEntry]:
        if not self.enabled:
            return []
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM task_memory
                    WHERE task_fingerprint = ? OR task_type = ?
                    ORDER BY CASE WHEN task_fingerprint = ? THEN 0 ELSE 1 END,
                             success DESC, created_at DESC
                    LIMIT ?
                    """,
                    (task_fingerprint, task_type, task_fingerprint, max(1, min(limit, 20))),
                ).fetchall()
            return [self._memory_entry(row) for row in rows]
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            self._disable_after_error(exc)
            return []

    def recall_experiences(self, *, task_type: str, limit: int = 5) -> list[ExperienceEntry]:
        if not self.enabled:
            return []
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM task_experience
                    WHERE task_type = ?
                    ORDER BY confidence DESC, created_at DESC
                    LIMIT ?
                    """,
                    (task_type, max(1, min(limit, 20))),
                ).fetchall()
            return [self._experience_entry(row) for row in rows]
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            self._disable_after_error(exc)
            return []

    def format_context(
        self,
        *,
        task_fingerprint: str,
        task_type: str,
        limit: int = 3,
    ) -> str:
        memories = self.recall_tasks(
            task_fingerprint=task_fingerprint,
            task_type=task_type,
            limit=limit,
        )
        experiences = self.recall_experiences(task_type=task_type, limit=limit)
        if not memories and not experiences:
            return ""
        lines = [
            "跨任务经验（仅作为当前数据分析的参考，不得替代当前实验结果）："
        ]
        for entry in memories:
            status = "成功" if entry.success else "失败"
            lines.append(f"- 历史任务[{status}]：{entry.summary}")
            for lesson in entry.lessons[:4]:
                lines.append(f"  经验：{lesson}")
        for entry in experiences:
            lines.append(
                f"- 经验信号（置信度 {entry.confidence:.2f}）："
                f"{entry.signal} → {entry.recommendation}"
            )
        return "\n".join(lines)[:6000]

    def _disable_after_error(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("长期记忆读写失败，本进程后续将跳过记忆操作：%s", self.error)
        self.enabled = False

    @staticmethod
    def _memory_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=int(row["id"]),
            task_fingerprint=str(row["task_fingerprint"]),
            task_type=str(row["task_type"]),
            summary=str(row["summary"]),
            lessons=tuple(json.loads(row["lessons_json"])),
            metadata=dict(json.loads(row["metadata_json"])),
            success=bool(row["success"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _experience_entry(row: sqlite3.Row) -> ExperienceEntry:
        return ExperienceEntry(
            id=int(row["id"]),
            task_type=str(row["task_type"]),
            signal=str(row["signal"]),
            recommendation=str(row["recommendation"]),
            evidence=dict(json.loads(row["evidence_json"])),
            confidence=float(row["confidence"]),
            created_at=str(row["created_at"]),
        )


_SHARED_MEMORY: LongTermMemory | None = None
_SHARED_MEMORY_LOCK = RLock()


def get_shared_memory() -> LongTermMemory:
    global _SHARED_MEMORY
    with _SHARED_MEMORY_LOCK:
        if _SHARED_MEMORY is None:
            _SHARED_MEMORY = LongTermMemory()
        return _SHARED_MEMORY


__all__ = [
    "ExperienceEntry",
    "LongTermMemory",
    "MemoryEntry",
    "get_shared_memory",
]
