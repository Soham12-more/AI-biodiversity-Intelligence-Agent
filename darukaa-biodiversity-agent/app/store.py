"""SQLite persistence for multi-turn memory: one row per session (current site profile + last
recommendations) and one row per message."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).resolve().parent.parent / "data" / "app.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  profile_json TEXT NOT NULL DEFAULT '{}',
  last_recs_json TEXT NOT NULL DEFAULT '[]',
  pending_questions_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id),
  role TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


@contextmanager
def conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def get_or_create(session_id: str | None) -> dict:
    sid = session_id or uuid.uuid4().hex[:12]
    with conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            c.execute("INSERT INTO sessions(id) VALUES (?)", (sid,))
            row = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return {"id": sid, "profile": json.loads(row["profile_json"]),
                "last_recs": json.loads(row["last_recs_json"]),
                "pending": json.loads(row["pending_questions_json"])}


def save(sid: str, profile: dict, last_recs: list | None = None, pending: list | None = None) -> None:
    with conn() as c:
        c.execute("UPDATE sessions SET profile_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                  (json.dumps(profile), sid))
        if last_recs is not None:
            c.execute("UPDATE sessions SET last_recs_json=? WHERE id=?", (json.dumps(last_recs), sid))
        if pending is not None:
            c.execute("UPDATE sessions SET pending_questions_json=? WHERE id=?", (json.dumps(pending), sid))


def log(sid: str, role: str, content: str) -> None:
    with conn() as c:
        c.execute("INSERT INTO messages(session_id, role, content) VALUES (?,?,?)", (sid, role, content))


def history(sid: str, limit: int = 20) -> list[dict]:
    with conn() as c:
        rows = c.execute("SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                         (sid, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]
