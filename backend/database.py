"""
SQLite Database manager for historical prediction logging.
"""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Generator
from backend.config import DB_PATH, DATA_DIR

logger = logging.getLogger("noise-history-db")


def init_db() -> None:
    """
    Ensures data directory exists and creates the prediction_history table.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                node_id TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence REAL,
                spl_db REAL NOT NULL,
                spl_status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON prediction_history (timestamp);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_node_id ON prediction_history (node_id);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_category ON prediction_history (predicted_category);
        """)
        conn.commit()
    logger.info("Database initialized at %s", DB_PATH)


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager providing a SQLite database connection with row factory enabled.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def is_db_healthy() -> bool:
    """
    Verifies that the database file is accessible and queryable.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return bool(cursor.fetchone())
    except Exception as e:
        logger.error("DB health probe failed: %s", e)
        return False
