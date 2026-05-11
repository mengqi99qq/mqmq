"""Interview data service using SQLite database."""

import sqlite3
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.schemas.interview import Category, InterviewData, InterviewResult, InterviewType

OLD_SEARCH_REC_CATEGORY = "搜广推推荐算法"
NEW_SEARCH_REC_CATEGORY = Category.SEARCH_REC.value


def normalize_category_value(value: str | None) -> str:
    """Normalize legacy category values to the current enum value."""
    if value == OLD_SEARCH_REC_CATEGORY:
        return NEW_SEARCH_REC_CATEGORY
    return value or ""


def expand_category_filter_values(categories: list[Category]) -> list[str]:
    """Expand category filters to include legacy aliases when needed."""
    values: list[str] = []
    for category in categories:
        values.append(category.value)
        if category == Category.SEARCH_REC:
            values.append(OLD_SEARCH_REC_CATEGORY)
    return values


def _resolve_db_path() -> Path:
    """Resolve database path from settings."""
    settings = get_settings()
    db_path = Path(settings.interview_db_path)
    if db_path.is_absolute():
        return db_path
    base_dir = Path(__file__).resolve().parents[2]
    return (base_dir / db_path).resolve()


def get_connection() -> sqlite3.Connection:
    """Get a read-only database connection."""
    db_path = _resolve_db_path()
    if not db_path.exists():
        raise FileNotFoundError(f"Interview DB not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_interview(row: sqlite3.Row) -> InterviewData:
    """Convert a database row to InterviewData model."""
    raw_result = row["result"] or "null"
    try:
        result = InterviewResult(raw_result)
    except ValueError:
        result = InterviewResult.NULL

    raw_interview_type = row["interview_type"]
    interview_type: Optional[InterviewType] = None
    if raw_interview_type:
        try:
            interview_type = InterviewType(raw_interview_type)
        except ValueError:
            interview_type = None

    return InterviewData(
        id=row["id"],
        title=row["title"],
        content=row["content"] or "",
        publish_time=row["publish_time"] or "",
        category=Category(normalize_category_value(row["category"])),
        source=row["source"] or "",
        source_id=row["source_id"] or "",
        company=row["company"],
        department=row["department"],
        stage=row["stage"],
        result=result,
        interview_type=interview_type,
    )


class DataService:
    """Service for querying interview data."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        self._conn = conn

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        return get_connection()

    def _close(self, conn: sqlite3.Connection) -> None:
        if self._conn is None:
            conn.close()

    def filter_interviews(
        self,
        categories: list[Category] | None = None,
        interview_types: list[str] | None = None,
        company: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> tuple[list[InterviewData], int]:
        """Filter and paginate interviews."""
        conn = self._get_conn()
        cursor = conn.cursor()

        where_clauses = ["1=1"]
        params: list = []

        if categories:
            category_values = expand_category_filter_values(categories)
            placeholders = ",".join("?" * len(category_values))
            where_clauses.append(f"category IN ({placeholders})")
            params.extend(category_values)

        if interview_types:
            placeholders = ",".join("?" * len(interview_types))
            where_clauses.append(f"interview_type IN ({placeholders})")
            params.extend(interview_types)

        if company:
            where_clauses.append("company LIKE ?")
            params.append(f"%{company}%")

        where_sql = " AND ".join(where_clauses)
        cursor.execute(f"SELECT COUNT(*) FROM interviews WHERE {where_sql}", params)
        total_count = cursor.fetchone()[0]

        query = f"SELECT * FROM interviews WHERE {where_sql} ORDER BY publish_time DESC"
        if page is not None and page_size is not None:
            offset = (page - 1) * page_size
            query += f" LIMIT {page_size} OFFSET {offset}"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        self._close(conn)

        return [row_to_interview(row) for row in rows], total_count


_data_service: Optional[DataService] = None


def get_data_service() -> DataService:
    """Get the global data service instance."""
    global _data_service
    if _data_service is None:
        _data_service = DataService()
    return _data_service

