
import json
import sqlite3
from pathlib import Path
from datetime import datetime


# Keep the database in the project folder.
PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "research_history.db"


def get_connection():
    """Open a connection to the research history database."""
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create the research history table if it does not exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS research_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                report_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                sources_count INTEGER NOT NULL DEFAULT 0,
                validation_issues TEXT NOT NULL DEFAULT '[]'
            )
            """
        )


def add_research_run(
    topic: str,
    report_path: str,
    sources_count: int = 0,
    validation_issues: list | None = None,
) -> int:
    """Save a completed research run and return its database ID."""
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    issues_json = json.dumps(validation_issues or [])

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO research_runs (
                topic,
                report_path,
                created_at,
                sources_count,
                validation_issues
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                topic,
                report_path,
                created_at,
                sources_count,
                issues_json,
            ),
        )

        return cursor.lastrowid


def get_recent_runs(limit: int = 50) -> list[dict]:
    """Return the newest research runs first."""
    limit = max(1, min(int(limit), 500))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM research_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results = []

    for row in rows:
        item = dict(row)
        item["validation_issues"] = json.loads(
            item["validation_issues"]
        )
        results.append(item)

    return results


def get_run_by_id(run_id: int) -> dict | None:
    """Retrieve one research run by its database ID."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM research_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    item = dict(row)
    item["validation_issues"] = json.loads(
        item["validation_issues"]
    )
    return item


# Initialize the database when this module is imported.
init_db()

def search_research_runs(query: str = "", limit: int = 20) -> list[dict]:
    """Search research history by topic, newest first."""
    limit = max(1, min(int(limit), 500))
    query = query.strip()

    with get_connection() as connection:
        if query:
            rows = connection.execute(
                """
                SELECT *
                FROM research_runs
                WHERE topic LIKE ? ESCAPE '\\'
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",
                    limit,
                ),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM research_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    results = []

    for row in rows:
        item = dict(row)
        item["validation_issues"] = json.loads(
            item["validation_issues"]
        )
        results.append(item)

    return results


def delete_research_run(run_id: int) -> bool:
    """Delete a history entry without deleting its Markdown report."""
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM research_runs WHERE id = ?",
            (run_id,),
        )
        return cursor.rowcount > 0
