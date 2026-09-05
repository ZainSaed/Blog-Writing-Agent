from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("blogs.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS blogs (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                topic     TEXT NOT NULL,
                title     TEXT NOT NULL,
                markdown  TEXT NOT NULL,
                evidence  TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)


def save_blog(topic: str, title: str, markdown: str, evidence: list) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO blogs (topic, title, markdown, evidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (topic, title, markdown, json.dumps(evidence), datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def list_blogs() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, topic, title, created_at FROM blogs ORDER BY id DESC LIMIT 50"
        ).fetchall()
    return [dict(r) for r in rows]


def load_blog(blog_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM blogs WHERE id = ?", (blog_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["evidence"] = json.loads(d["evidence"])
    return d


def delete_blog(blog_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM blogs WHERE id = ?", (blog_id,))