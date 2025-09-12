# app/services/chat_memory.py
from __future__ import annotations
import os
import sqlite3
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from app.services.local_llm import LocalLlama

DB_PATH = os.getenv("CHAT_DB_PATH", "./out/chat_memory.sqlite3")

# --------- schema & init ---------
def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    # sqlite3 is already redirected to pysqlite3 in app/main.py
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with _conn() as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
          id TEXT PRIMARY KEY,
          summary TEXT DEFAULT '',
          updated_at REAL NOT NULL
        )""")
        cx.execute("""
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          conversation_id TEXT NOT NULL,
          role TEXT NOT NULL,           -- 'user' | 'assistant' | 'system'
          content TEXT NOT NULL,
          tokens INTEGER NOT NULL,
          created_at REAL NOT NULL,
          FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )""")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_msgs_conv_created ON messages(conversation_id, created_at)")
        cx.commit()

# --------- datatypes ---------
@dataclass
class ChatMessage:
    role: str
    content: str
    tokens: int
    created_at: float

# very light heuristic to estimate tokens without a tokenizer
def _approx_tokens(text: str) -> int:
    # ~4 chars ≈ 1 token (roughly for English Llama)
    return max(1, int(len(text) / 4))

# --------- CRUD ---------
def ensure_conversation(conv_id: str):
    now = time.time()
    with _conn() as cx:
        cur = cx.execute("SELECT id FROM conversations WHERE id=?", (conv_id,))
        row = cur.fetchone()
        if not row:
            cx.execute("INSERT INTO conversations(id, summary, updated_at) VALUES(?,?,?)", (conv_id, "", now))
        else:
            cx.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conv_id))
        cx.commit()

def add_message(conv_id: str, role: str, content: str) -> None:
    ensure_conversation(conv_id)
    with _conn() as cx:
        cx.execute(
            "INSERT INTO messages(conversation_id, role, content, tokens, created_at) VALUES(?,?,?,?,?)",
            (conv_id, role, content, _approx_tokens(content), time.time()),
        )
        cx.execute("UPDATE conversations SET updated_at=? WHERE id=?", (time.time(), conv_id))
        cx.commit()

def get_recent_messages(conv_id: str, limit: int = 8) -> List[ChatMessage]:
    with _conn() as cx:
        cur = cx.execute("""
            SELECT role, content, tokens, created_at
            FROM messages
            WHERE conversation_id=?
            ORDER BY created_at DESC
            LIMIT ?""", (conv_id, limit))
        rows = cur.fetchall()
    out = [ChatMessage(role=r[0], content=r[1], tokens=int(r[2]), created_at=float(r[3])) for r in rows]
    out.reverse()
    return out

def get_summary(conv_id: str) -> str:
    with _conn() as cx:
        cur = cx.execute("SELECT summary FROM conversations WHERE id=?", (conv_id,))
        row = cur.fetchone()
    return row[0] if row and row[0] else ""

def set_summary(conv_id: str, summary: str) -> None:
    with _conn() as cx:
        cx.execute("UPDATE conversations SET summary=?, updated_at=? WHERE id=?", (summary, time.time(), conv_id))
        cx.commit()

def count_turns(conv_id: str) -> int:
    with _conn() as cx:
        cur = cx.execute("SELECT COUNT(1) FROM messages WHERE conversation_id=?", (conv_id,))
        return int(cur.fetchone()[0])

# --------- context builder ---------
def build_chat_context(
    conv_id: str,
    max_input_tokens: int = 3000,
    max_recent_turns: int = 8,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Returns a compact plain-text block to prepend to your prompt + structured recent turns.
    The block contains a short summary (if any) and last N turns, trimmed to budget.
    """
    summary = get_summary(conv_id).strip()
    recent = get_recent_messages(conv_id, limit=max_recent_turns)

    # pack into a simple, model-friendly format
    lines: List[str] = []
    used_tokens = 0

    if summary:
        s = f"Conversation summary: {summary}"
        t = _approx_tokens(s)
        if used_tokens + t < max_input_tokens * 0.25:  # keep summary small slice
            lines.append(s)
            used_tokens += t

    # include recent messages until budget hit (reserve 60% for RAG context + question)
    budget = int(max_input_tokens * 0.4)
    for m in recent:
        line = f"{m.role.upper()}: {m.content}"
        t = _approx_tokens(line)
        if used_tokens + t > budget:
            break
        lines.append(line)
        used_tokens += t

    block = "\n".join(lines).strip()
    structured = [{"role": m.role, "content": m.content} for m in recent]
    return block, structured

# --------- rolling summary (called occasionally) ---------
def maybe_update_summary(
    conv_id: str,
    min_turns_before_summarize: int = 6,
    target_summary_tokens: int = 200,
) -> Optional[str]:
    """
    Summarizes older context into a short digest when enough turns have happened.
    Returns the updated summary (or None if no update).
    """
    turns = count_turns(conv_id)
    if turns < min_turns_before_summarize:
        return None

    # build a short window of older+recent for summarization
    recent = get_recent_messages(conv_id, limit=10)
    # keep just the text to summarize; system prompt not needed
    convo_text = "\n".join(f"{m.role.upper()}: {m.content}" for m in recent)

    prev_sum = get_summary(conv_id)
    prompt = f"""Summarize the following conversation for future context in 4-6 sentences.
Preserve important facts (companies, amounts, dates, locations) as bullet points if present.
Be concise and avoid redundancy.

PREVIOUS SUMMARY (may be empty):
{prev_sum}

RECENT TURNS:
{convo_text}

Return only the new summary."""
    try:
        llm = LocalLlama()
        summary = llm.gen(prompt, temp=0.1, max_tokens=target_summary_tokens).strip()
        if summary:
            set_summary(conv_id, summary)
            return summary
    except Exception:
        # non-fatal; continue without updating
        pass
    return None
