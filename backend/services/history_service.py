"""
History service for storing, querying, and managing noise prediction records in SQLite.
"""

import logging
from typing import Any, Dict, List, Optional
from backend.database import get_db_connection

logger = logging.getLogger("noise-history-service")


class HistoryService:
    """
    CRUD operations for prediction history records.
    """

    @staticmethod
    def save_prediction(
        timestamp: float,
        node_id: str,
        predicted_category: str,
        confidence: Optional[float],
        spl_db: float,
        spl_status: str
    ) -> int:
        """
        Inserts a successful prediction record into SQLite.
        Returns the generated row ID.
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO prediction_history (
                    timestamp, node_id, predicted_category, confidence, spl_db, spl_status
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, node_id, predicted_category, confidence, spl_db, spl_status)
            )
            conn.commit()
            return cursor.lastrowid

    @staticmethod
    def get_history(
        limit: int = 50,
        category: Optional[str] = None,
        min_db: Optional[float] = None,
        node_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves historical prediction records with optional filters.
        """
        query = "SELECT id, timestamp, node_id, predicted_category, confidence, spl_db, spl_status, created_at FROM prediction_history WHERE 1=1"
        params: List[Any] = []

        if category and category.strip().lower() not in ("all", ""):
            query += " AND (predicted_category = ? OR predicted_category LIKE ?)"
            cat_val = category.strip()
            params.extend([cat_val, f"%{cat_val}%"])

        if min_db is not None and min_db > 0.0:
            query += " AND spl_db >= ?"
            params.append(min_db)

        if node_id and node_id.strip():
            query += " AND node_id = ?"
            params.append(node_id.strip())

        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    "id": row["id"],
                    "timestamp": float(row["timestamp"]),
                    "node_id": str(row["node_id"]),
                    "predicted_category": str(row["predicted_category"]),
                    "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
                    "spl_db": float(row["spl_db"]),
                    "spl_status": str(row["spl_status"]),
                    "created_at": str(row["created_at"])
                }
                for row in rows
            ]

    @staticmethod
    def clear_history() -> int:
        """
        Clears all records from the prediction_history table.
        Returns the number of rows deleted. Useful for test cleanup.
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) as cnt FROM prediction_history")
            count = cursor.fetchone()["cnt"]
            cursor.execute("DELETE FROM prediction_history")
            conn.commit()
            logger.info("Cleared %d prediction history rows.", count)
            return count
