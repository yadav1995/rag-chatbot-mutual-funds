"""
Thread Manager — Multi-Thread Conversation Persistence

Implements conversation management per RAGArchitecture.md §6:
- Each thread has its own independent message history
- Thread storage in SQLite for persistence
- Last 3 message pairs used as LLM context window
- Thread CRUD operations (create, list, get, add message, delete)
"""

import json
import logging
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_DIR

logger = logging.getLogger(__name__)

THREADS_DB_FILE = DATA_DIR / "threads.db"


@dataclass
class Message:
    """A single message in a conversation thread."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = ""
    citations: list[str] = field(default_factory=list)


@dataclass
class Thread:
    """A conversation thread."""
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[Message] = field(default_factory=list)


class ThreadManager:
    """
    Manages multi-thread conversations with SQLite persistence.
    """

    def __init__(self, db_path: str = None):
        db_path = db_path or str(THREADS_DB_FILE)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """Create threads and messages tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id   TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New Chat',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                citations   TEXT DEFAULT '[]',
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_thread
                ON messages(thread_id);
        """)
        self.conn.commit()

    # ── Thread CRUD ──────────────────────────────────────────────────────

    def create_thread(self, title: str = "New Chat") -> Thread:
        """Create a new conversation thread."""
        thread_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            "INSERT INTO threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, title, now, now),
        )
        self.conn.commit()

        thread = Thread(
            thread_id=thread_id,
            title=title,
            created_at=now,
            updated_at=now,
        )
        logger.info(f"Created thread: {thread_id} ({title})")
        return thread

    def get_thread(self, thread_id: str) -> Thread | None:
        """Get a thread with all its messages."""
        row = self.conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()

        if not row:
            return None

        thread = Thread(
            thread_id=row[0],
            title=row[1],
            created_at=row[2],
            updated_at=row[3],
        )

        # Load messages
        msg_rows = self.conn.execute(
            "SELECT role, content, citations, timestamp FROM messages "
            "WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()

        for msg_row in msg_rows:
            citations = json.loads(msg_row[2]) if msg_row[2] else []
            thread.messages.append(Message(
                role=msg_row[0],
                content=msg_row[1],
                citations=citations,
                timestamp=msg_row[3],
            ))

        return thread

    def list_threads(self, limit: int = 20) -> list[dict]:
        """List all threads, most recent first."""
        rows = self.conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [
            {
                "thread_id": r[0],
                "title": r[1],
                "created_at": r[2],
                "updated_at": r[3],
            }
            for r in rows
        ]

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and all its messages."""
        # Delete messages first
        self.conn.execute(
            "DELETE FROM messages WHERE thread_id = ?", (thread_id,)
        )
        cursor = self.conn.execute(
            "DELETE FROM threads WHERE thread_id = ?", (thread_id,)
        )
        self.conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted thread: {thread_id}")
        return deleted

    # ── Messages ─────────────────────────────────────────────────────────

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        citations: list[str] = None,
    ) -> Message:
        """Add a message to a thread."""
        now = datetime.now(timezone.utc).isoformat()
        citations = citations or []

        self.conn.execute(
            "INSERT INTO messages (thread_id, role, content, citations, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, role, content, json.dumps(citations), now),
        )

        # Update thread's updated_at
        self.conn.execute(
            "UPDATE threads SET updated_at = ? WHERE thread_id = ?",
            (now, thread_id),
        )

        # Auto-set thread title from first user message
        first_msg = self.conn.execute(
            "SELECT content FROM messages WHERE thread_id = ? AND role = 'user' ORDER BY id LIMIT 1",
            (thread_id,),
        ).fetchone()
        if first_msg:
            title = first_msg[0][:50] + ("..." if len(first_msg[0]) > 50 else "")
            self.conn.execute(
                "UPDATE threads SET title = ? WHERE thread_id = ?",
                (title, thread_id),
            )

        self.conn.commit()

        return Message(
            role=role,
            content=content,
            citations=citations,
            timestamp=now,
        )

    def get_recent_history(
        self, thread_id: str, max_pairs: int = 3
    ) -> list[dict]:
        """
        Get recent conversation history for LLM context.
        Returns last N user+assistant message pairs as OpenAI-format dicts.
        """
        rows = self.conn.execute(
            "SELECT role, content FROM messages "
            "WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
            (thread_id, max_pairs * 2),
        ).fetchall()

        # Reverse to get chronological order
        history = [
            {"role": r[0], "content": r[1]}
            for r in reversed(rows)
        ]

        return history

    def close(self):
        """Close the database connection."""
        self.conn.close()
