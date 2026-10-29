"""
Prediction history query router.
"""

from typing import List, Optional
from fastapi import APIRouter, Query
from backend.models.schemas import HistoryRecord
from backend.services.history_service import HistoryService

router = APIRouter(prefix="/history", tags=["History"])


@router.get(
    "",
    response_model=List[HistoryRecord],
    summary="Query historical noise predictions",
    description="Returns persisted acoustic classifications stored in SQLite, ordered newest first."
)
def get_prediction_history(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of records to return"),
    category: Optional[str] = Query(None, description="Filter by sound category (or 'All')"),
    min_db: Optional[float] = Query(None, ge=0.0, description="Minimum decibels threshold filter"),
    node_id: Optional[str] = Query(None, description="Filter by sensor node ID")
) -> List[HistoryRecord]:
    rows = HistoryService.get_history(
        limit=limit,
        category=category,
        min_db=min_db,
        node_id=node_id
    )
    return [HistoryRecord(**r) for r in rows]


@router.delete(
    "",
    summary="Clear prediction history",
    description="Deletes all historical records from the SQLite database. Useful for resetting test environments."
)
def clear_prediction_history():
    deleted_count = HistoryService.clear_history()
    return {
        "status": "success",
        "deleted_count": deleted_count,
        "message": f"Successfully cleared {deleted_count} history records."
    }
