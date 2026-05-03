from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DrowsyEvent:
    source: str
    label: str
    score: float | None = None
    route: str | None = None
    backend: str | None = None
    ear: float | None = None
    mar: float | None = None
    frame_path: str | None = None
    s3_uri: str | None = None
    metadata: dict[str, Any] | None = None


class EventLogger:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def log(self, event: DrowsyEvent) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO events (
                    created_at, source, label, score, route, backend,
                    ear, mar, frame_path, s3_uri, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    event.source,
                    event.label,
                    event.score,
                    event.route,
                    event.backend,
                    event.ear,
                    event.mar,
                    event.frame_path,
                    event.s3_uri,
                    json.dumps(event.metadata or {}),
                ),
            )

    def summary(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            total_events = conn.execute("SELECT COUNT(*) AS value FROM events").fetchone()["value"]
            drowsy_events = conn.execute(
                "SELECT COUNT(*) AS value FROM events WHERE label = 'drowsy'"
            ).fetchone()["value"]
            alert_events = conn.execute(
                "SELECT COUNT(*) AS value FROM events WHERE label = 'alert'"
            ).fetchone()["value"]
            remote_events = conn.execute(
                "SELECT COUNT(*) AS value FROM events WHERE route = 'remote'"
            ).fetchone()["value"]
            local_events = conn.execute(
                "SELECT COUNT(*) AS value FROM events WHERE route = 'local'"
            ).fetchone()["value"]
            last_event = conn.execute(
                """
                SELECT created_at, label, route, backend, score, ear, mar, frame_path, s3_uri
                FROM events
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

        return {
            "total_events": total_events,
            "drowsy_events": drowsy_events,
            "alert_events": alert_events,
            "remote_events": remote_events,
            "local_events": local_events,
            "last_event": dict(last_event) if last_event else None,
            "db_path": str(self.db_path),
        }

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, created_at, source, label, score, route, backend,
                       ear, mar, frame_path, s3_uri, metadata_json
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        events = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            events.append(item)
        return events

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    label TEXT NOT NULL,
                    score REAL,
                    route TEXT,
                    backend TEXT,
                    ear REAL,
                    mar REAL,
                    frame_path TEXT,
                    s3_uri TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_label ON events(label)")
