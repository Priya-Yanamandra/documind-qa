import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict


DB_PATH = "./qa_memory.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    conn = _connect()
    conn.execute(
        "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def load_history(session_id: str, limit: int = 20) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content, timestamp FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in reversed(rows)]


def save_feedback(session_id: str, question: str, answer: str, rating: int):
    conn = _connect()
    conn.execute(
        "INSERT INTO feedback (session_id, question, answer, rating, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, question, answer, rating, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def load_feedback(session_id: str) -> List[Dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT question, answer, rating, timestamp FROM feedback WHERE session_id=? ORDER BY id DESC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_history(session_id: str):
    conn = _connect()
    conn.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()
