import sqlite3
from datetime import datetime


class NaiveContextAssembler:
    """SQLite-backed episodic memory. Stores pre-formatted lines, retrieves full history."""

    def __init__(self, db_path: str = "context.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                content   TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def store(self, content: str) -> None:
        self.conn.execute(
            "INSERT INTO conversation (timestamp, content) VALUES (?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), content),
        )
        self.conn.commit()

    def retrieve(self, query: str) -> str:
        rows = self.conn.execute(
            "SELECT timestamp, content FROM conversation ORDER BY id"
        ).fetchall()
        return "\n".join(f"(Current System Time: {ts}) {content}" for ts, content in rows)
