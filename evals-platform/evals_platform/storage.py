"""SQLite schema, persistence operations, and queries for evaluation history.

``EvalStorage`` owns two related tables: ``eval_runs`` stores run-level settings
and lifecycle state, while ``eval_results`` stores immutable case snapshots,
agent trajectories, metrics, optional judge output, and mutable human reviews.
Structured values are JSON-encoded at the database boundary and decoded back to
plain Python values for callers.

Each operation uses a short-lived connection with foreign keys, a busy timeout,
and write-ahead logging enabled during initialization. This design is sufficient
for a local Streamlit process and keeps transactions small; it is not intended as
a multi-host service database. Parameter binding protects values in all queries,
while filter clauses are selected only from fixed internal column names.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import EvalSettings, EvaluationResult


def utc_now() -> str:
    """Return an ISO-8601 timestamp in UTC for persisted lifecycle events."""

    return datetime.now(timezone.utc).isoformat()


class EvalStorage:
    """Persist, query, and manually review local evaluation runs and results."""

    def __init__(self, database_path: Path) -> None:
        """Create parent directories and initialize the database schema."""

        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a configured SQLite connection that returns named rows."""

        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        """Idempotently create tables and indexes required by the platform."""

        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model TEXT NOT NULL,
                    max_iterations INTEGER NOT NULL,
                    judge_enabled INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS eval_results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
                    case_id TEXT NOT NULL,
                    case_snapshot_json TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    status TEXT NOT NULL,
                    termination_reason TEXT,
                    final_answer TEXT,
                    messages_json TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    judge_json TEXT,
                    latency_total_ms REAL NOT NULL DEFAULT 0,
                    error_type TEXT,
                    error_message TEXT,
                    manual_verdict TEXT NOT NULL DEFAULT 'unreviewed'
                        CHECK (manual_verdict IN ('unreviewed', 'pass', 'fail')),
                    manual_notes TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_runs_started_at ON eval_runs(started_at);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON eval_runs(status);
                CREATE INDEX IF NOT EXISTS idx_results_run_id ON eval_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_results_case_id ON eval_results(case_id);
                CREATE INDEX IF NOT EXISTS idx_results_created_at ON eval_results(created_at);
                CREATE INDEX IF NOT EXISTS idx_results_status ON eval_results(status);
                CREATE INDEX IF NOT EXISTS idx_results_verdict ON eval_results(manual_verdict);
                """
            )

    def create_run(
        self,
        run_id: str,
        name: str,
        mode: str,
        settings: EvalSettings,
    ) -> None:
        """Insert a running evaluation record with a settings snapshot."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO eval_runs (
                    id, name, mode, status, model, max_iterations,
                    judge_enabled, config_json, started_at
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    name,
                    mode,
                    settings.model,
                    settings.max_iterations,
                    int(settings.judge_enabled),
                    json.dumps(asdict(settings)),
                    utc_now(),
                ),
            )

    def finalize_run(
        self, run_id: str, status: str, error_message: str | None = None
    ) -> None:
        """Mark a run terminal and record its completion time and optional error."""

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE eval_runs
                SET status = ?, completed_at = ?, error_message = ?
                WHERE id = ?
                """,
                (status, utc_now(), error_message, run_id),
            )

    def append_result(self, result: EvaluationResult) -> None:
        """Serialize and insert one independently queryable case result."""

        agent = result.agent_result
        created_at = result.created_at or utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO eval_results (
                    id, run_id, case_id, case_snapshot_json, prompt, tags_json,
                    difficulty, status, termination_reason, final_answer,
                    messages_json, steps_json, metrics_json, judge_json,
                    latency_total_ms, error_type, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.result_id,
                    result.run_id,
                    result.case.case_id,
                    json.dumps(result.case.to_dict(), ensure_ascii=False),
                    result.case.prompt,
                    json.dumps(result.case.tags),
                    result.case.difficulty,
                    result.status,
                    agent.get("termination_reason"),
                    agent.get("final_answer", ""),
                    json.dumps(agent.get("messages", []), ensure_ascii=False),
                    json.dumps(agent.get("steps", []), ensure_ascii=False),
                    json.dumps(result.metrics, ensure_ascii=False),
                    json.dumps(result.judge, ensure_ascii=False) if result.judge else None,
                    float(agent.get("total_latency_ms", 0)),
                    agent.get("error_type"),
                    agent.get("error_message"),
                    created_at,
                ),
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        """Decode JSON columns and expose metrics as ``metric_*`` convenience keys."""

        value = dict(row)
        for column in (
            "config_json",
            "case_snapshot_json",
            "tags_json",
            "messages_json",
            "steps_json",
            "metrics_json",
            "judge_json",
        ):
            if column in value and value[column] is not None:
                value[column.removesuffix("_json")] = json.loads(value[column])
        metrics = value.get("metrics") or {}
        value.update({f"metric_{key}": item for key, item in metrics.items()})
        return value

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return newest runs with aggregate result and successful-case counts."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, COUNT(e.id) AS result_count,
                       SUM(CASE WHEN e.status = 'success' THEN 1 ELSE 0 END) AS success_count
                FROM eval_runs r
                LEFT JOIN eval_results e ON e.run_id = r.id
                GROUP BY r.id
                ORDER BY r.started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def list_results(
        self,
        *,
        run_id: str | None = None,
        status: str | None = None,
        verdict: str | None = None,
        model: str | None = None,
        tag: str | None = None,
        text: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query newest results using optional run, status, model, tag, and text filters.

        The text filter searches prompts, final answers, and case IDs. Returned
        rows include decoded JSON fields, parent-run metadata, and flattened
        metric keys used directly by the dashboard.
        """

        clauses: list[str] = []
        parameters: list[Any] = []
        filters = {
            "e.run_id": run_id,
            "e.status": status,
            "e.manual_verdict": verdict,
            "r.model": model,
        }
        for column, value in filters.items():
            if value:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if tag:
            clauses.append("e.tags_json LIKE ?")
            parameters.append(f'%"{tag}"%')
        if text:
            clauses.append("(e.prompt LIKE ? OR e.final_answer LIKE ? OR e.case_id LIKE ?)")
            parameters.extend([f"%{text}%"] * 3)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT e.*, r.model, r.name AS run_name, r.started_at AS run_started_at
                FROM eval_results e
                JOIN eval_runs r ON r.id = e.run_id
                {where}
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._decode(row) for row in rows]

    def get_result(self, result_id: str) -> dict[str, Any] | None:
        """Return one fully decoded result and its run metadata, if it exists."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT e.*, r.model, r.name AS run_name
                FROM eval_results e JOIN eval_runs r ON r.id = e.run_id
                WHERE e.id = ?
                """,
                (result_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def update_review(self, result_id: str, verdict: str, notes: str) -> None:
        """Set a result's manual verdict, notes, and review timestamp.

        Resetting the verdict to ``unreviewed`` clears the timestamp. Unsupported
        verdicts are rejected before opening a write transaction.
        """

        if verdict not in {"unreviewed", "pass", "fail"}:
            raise ValueError("Invalid manual verdict.")
        reviewed_at = None if verdict == "unreviewed" else utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE eval_results
                SET manual_verdict = ?, manual_notes = ?, reviewed_at = ?
                WHERE id = ?
                """,
                (verdict, notes, reviewed_at, result_id),
            )
