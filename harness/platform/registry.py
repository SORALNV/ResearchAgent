from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from harness.platform.models import (
    EventKind,
    JobEvent,
    JobRecord,
    JobSpec,
    JobStatus,
    Project,
    ProjectStatus,
    SteeringEvent,
    WorkSession,
    WorkSessionStatus,
    decode_json,
    encode_json,
)
from harness.state import utc_timestamp


class PlatformRegistry:
    """SQLite source of truth for portable projects, threads, jobs, and events.

    The registry deliberately stores domain-neutral records. Research and Kaggle
    extensions keep their own metadata in JSON fields or dedicated domain tables.
    Every write runs in an explicit transaction and event sequence allocation is
    serialized, which makes Discord reconnects and duplicate interaction retries
    deterministic.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS work_sessions (
                    session_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(project_id),
                    title TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    parent_session_id TEXT REFERENCES work_sessions(session_id),
                    discord_guild_id TEXT,
                    discord_parent_channel_id TEXT,
                    discord_thread_id TEXT UNIQUE,
                    discord_live_message_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_project
                    ON work_sessions(project_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                    ON work_sessions(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    work_session_id TEXT NOT NULL REFERENCES work_sessions(session_id),
                    parent_job_id TEXT REFERENCES jobs(job_id),
                    domain TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    entrypoint TEXT NOT NULL,
                    backend_preferences_json TEXT NOT NULL,
                    resources_json TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    backend TEXT,
                    backend_job_id TEXT,
                    current_stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_session
                    ON jobs(work_session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_status
                    ON jobs(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS job_events (
                    event_id TEXT PRIMARY KEY,
                    job_id TEXT REFERENCES jobs(job_id),
                    work_session_id TEXT NOT NULL REFERENCES work_sessions(session_id),
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(work_session_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_events_session_sequence
                    ON job_events(work_session_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_events_job_sequence
                    ON job_events(job_id, sequence);

                CREATE TABLE IF NOT EXISTS steering_events (
                    steering_id TEXT PRIMARY KEY,
                    work_session_id TEXT NOT NULL REFERENCES work_sessions(session_id),
                    job_id TEXT REFERENCES jobs(job_id),
                    kind TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    apply_after TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    applied_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_steering_pending
                    ON steering_events(work_session_id, status, created_at);

                CREATE TABLE IF NOT EXISTS processed_interactions (
                    correlation_id TEXT PRIMARY KEY,
                    work_session_id TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS platform_kv (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO schema_info(key, value) VALUES('version', ?)",
                (str(self.SCHEMA_VERSION),),
            )

    # Projects

    def create_project(self, project: Project) -> Project:
        if not project.title:
            raise ValueError("project title must not be empty")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO projects(
                    project_id, domain, title, description, status,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.project_id,
                    project.domain.value,
                    project.title,
                    project.description,
                    project.status.value,
                    encode_json(project.metadata),
                    project.created_at,
                    project.updated_at,
                ),
            )
        return project

    def get_project(self, project_id: str) -> Project | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
        return self._project_from_row(row) if row else None

    def list_projects(
        self,
        *,
        status: ProjectStatus | str | None = None,
        limit: int = 100,
    ) -> list[Project]:
        query = "SELECT * FROM projects"
        parameters: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            parameters.append(ProjectStatus(str(status)).value)
        query += " ORDER BY updated_at DESC LIMIT ?"
        parameters.append(max(1, min(limit, 1000)))
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._project_from_row(row) for row in rows]

    def update_project(
        self,
        project_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: ProjectStatus | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Project:
        current = self.get_project(project_id)
        if current is None:
            raise KeyError(f"unknown project: {project_id}")
        updated = replace(
            current,
            title=title.strip() if title is not None else current.title,
            description=(
                description.strip() if description is not None else current.description
            ),
            status=ProjectStatus(str(status)) if status is not None else current.status,
            metadata=dict(metadata) if metadata is not None else current.metadata,
            updated_at=utc_timestamp(),
        )
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE projects
                   SET title = ?, description = ?, status = ?,
                       metadata_json = ?, updated_at = ?
                 WHERE project_id = ?
                """,
                (
                    updated.title,
                    updated.description,
                    updated.status.value,
                    encode_json(updated.metadata),
                    updated.updated_at,
                    project_id,
                ),
            )
        return updated

    # Work sessions and Discord routes

    def create_work_session(self, session: WorkSession) -> WorkSession:
        if self.get_project(session.project_id) is None:
            raise KeyError(f"unknown project: {session.project_id}")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO work_sessions(
                    session_id, project_id, title, objective, status,
                    current_stage, parent_session_id, discord_guild_id,
                    discord_parent_channel_id, discord_thread_id,
                    discord_live_message_id, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.project_id,
                    session.title,
                    session.objective,
                    session.status.value,
                    session.current_stage,
                    session.parent_session_id,
                    session.discord_guild_id,
                    session.discord_parent_channel_id,
                    session.discord_thread_id,
                    session.discord_live_message_id,
                    encode_json(session.metadata),
                    session.created_at,
                    session.updated_at,
                ),
            )
        return session

    def get_work_session(self, session_id: str) -> WorkSession | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM work_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._session_from_row(row) if row else None

    def find_work_session_by_thread(self, thread_id: str | int) -> WorkSession | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM work_sessions WHERE discord_thread_id = ?",
                (str(thread_id),),
            ).fetchone()
        return self._session_from_row(row) if row else None

    def list_work_sessions(
        self,
        *,
        project_id: str | None = None,
        statuses: tuple[WorkSessionStatus | str, ...] | None = None,
        limit: int = 100,
    ) -> list[WorkSession]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            parameters.append(project_id)
        if statuses:
            normalized = [WorkSessionStatus(str(item)).value for item in statuses]
            placeholders = ",".join("?" for _ in normalized)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(normalized)
        query = "SELECT * FROM work_sessions"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC LIMIT ?"
        parameters.append(max(1, min(limit, 1000)))
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._session_from_row(row) for row in rows]

    def update_work_session(
        self,
        session_id: str,
        *,
        status: WorkSessionStatus | str | None = None,
        current_stage: str | None = None,
        title: str | None = None,
        objective: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        discord_guild_id: str | int | None = None,
        discord_parent_channel_id: str | int | None = None,
        discord_thread_id: str | int | None = None,
        discord_live_message_id: str | int | None = None,
    ) -> WorkSession:
        current = self.get_work_session(session_id)
        if current is None:
            raise KeyError(f"unknown work session: {session_id}")
        updated = replace(
            current,
            status=(
                WorkSessionStatus(str(status)) if status is not None else current.status
            ),
            current_stage=(
                current_stage.strip() if current_stage is not None else current.current_stage
            ),
            title=title.strip() if title is not None else current.title,
            objective=objective.strip() if objective is not None else current.objective,
            metadata=dict(metadata) if metadata is not None else current.metadata,
            discord_guild_id=(
                str(discord_guild_id)
                if discord_guild_id is not None
                else current.discord_guild_id
            ),
            discord_parent_channel_id=(
                str(discord_parent_channel_id)
                if discord_parent_channel_id is not None
                else current.discord_parent_channel_id
            ),
            discord_thread_id=(
                str(discord_thread_id)
                if discord_thread_id is not None
                else current.discord_thread_id
            ),
            discord_live_message_id=(
                str(discord_live_message_id)
                if discord_live_message_id is not None
                else current.discord_live_message_id
            ),
            updated_at=utc_timestamp(),
        )
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE work_sessions
                   SET title = ?, objective = ?, status = ?, current_stage = ?,
                       discord_guild_id = ?, discord_parent_channel_id = ?,
                       discord_thread_id = ?, discord_live_message_id = ?,
                       metadata_json = ?, updated_at = ?
                 WHERE session_id = ?
                """,
                (
                    updated.title,
                    updated.objective,
                    updated.status.value,
                    updated.current_stage,
                    updated.discord_guild_id,
                    updated.discord_parent_channel_id,
                    updated.discord_thread_id,
                    updated.discord_live_message_id,
                    encode_json(updated.metadata),
                    updated.updated_at,
                    session_id,
                ),
            )
        return updated

    # Jobs

    def create_job(self, spec: JobSpec) -> JobRecord:
        if self.get_work_session(spec.work_session_id) is None:
            raise KeyError(f"unknown work session: {spec.work_session_id}")
        record = JobRecord(spec=spec)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, work_session_id, parent_job_id, domain, task_type,
                    entrypoint, backend_preferences_json, resources_json,
                    inputs_json, outputs_json, metadata_json, created_at,
                    status, backend, backend_job_id, current_stage, progress,
                    result_json, error, started_at, finished_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._job_values(record),
            )
        self.append_event(
            JobEvent.new(
                work_session_id=spec.work_session_id,
                job_id=spec.job_id,
                kind=EventKind.STATUS,
                message=f"Job created: {spec.job_id}",
                payload={"status": JobStatus.CREATED.value},
            )
        )
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._job_from_row(row) if row else None

    def list_jobs(
        self,
        *,
        work_session_id: str | None = None,
        statuses: tuple[JobStatus | str, ...] | None = None,
        limit: int = 200,
    ) -> list[JobRecord]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if work_session_id:
            clauses.append("work_session_id = ?")
            parameters.append(work_session_id)
        if statuses:
            normalized = [JobStatus(str(item)).value for item in statuses]
            placeholders = ",".join("?" for _ in normalized)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(normalized)
        query = "SELECT * FROM jobs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        parameters.append(max(1, min(limit, 5000)))
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._job_from_row(row) for row in rows]

    def update_job(
        self,
        job_id: str,
        *,
        status: JobStatus | str | None = None,
        backend: str | None = None,
        backend_job_id: str | None = None,
        current_stage: str | None = None,
        progress: float | None = None,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        emit_event: bool = True,
    ) -> JobRecord:
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(f"unknown job: {job_id}")
        new_status = JobStatus(str(status)) if status is not None else current.status
        now = utc_timestamp()
        if started_at is None and current.started_at is None and new_status in {
            JobStatus.PREPARING,
            JobStatus.SUBMITTED,
            JobStatus.RUNNING,
        }:
            started_at = now
        if finished_at is None and new_status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            finished_at = now
        updated = replace(
            current,
            status=new_status,
            backend=backend if backend is not None else current.backend,
            backend_job_id=(
                backend_job_id if backend_job_id is not None else current.backend_job_id
            ),
            current_stage=(
                current_stage if current_stage is not None else current.current_stage
            ),
            progress=(
                min(1.0, max(0.0, float(progress)))
                if progress is not None
                else current.progress
            ),
            result=dict(result) if result is not None else current.result,
            error=error if error is not None else current.error,
            started_at=started_at if started_at is not None else current.started_at,
            finished_at=finished_at if finished_at is not None else current.finished_at,
            updated_at=now,
        )
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE jobs
                   SET status = ?, backend = ?, backend_job_id = ?,
                       current_stage = ?, progress = ?, result_json = ?, error = ?,
                       started_at = ?, finished_at = ?, updated_at = ?
                 WHERE job_id = ?
                """,
                (
                    updated.status.value,
                    updated.backend,
                    updated.backend_job_id,
                    updated.current_stage,
                    updated.progress,
                    encode_json(updated.result),
                    updated.error,
                    updated.started_at,
                    updated.finished_at,
                    updated.updated_at,
                    job_id,
                ),
            )
        if emit_event:
            self.append_event(
                JobEvent.new(
                    work_session_id=updated.spec.work_session_id,
                    job_id=job_id,
                    kind=EventKind.STATUS,
                    message=(
                        f"{job_id}: {updated.status.value} / {updated.current_stage}"
                    ),
                    payload={
                        "status": updated.status.value,
                        "stage": updated.current_stage,
                        "progress": updated.progress,
                        "backend": updated.backend,
                        "error": updated.error,
                    },
                )
            )
        return updated

    # Events and steering

    def append_event(self, event: JobEvent) -> JobEvent:
        with self._transaction() as connection:
            next_sequence = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1
                  FROM job_events
                 WHERE work_session_id = ?
                """,
                (event.work_session_id,),
            ).fetchone()[0]
            stored = replace(event, sequence=int(next_sequence))
            connection.execute(
                """
                INSERT INTO job_events(
                    event_id, job_id, work_session_id, kind, message,
                    payload_json, sequence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored.event_id,
                    stored.job_id,
                    stored.work_session_id,
                    stored.kind.value,
                    stored.message,
                    encode_json(stored.payload),
                    stored.sequence,
                    stored.created_at,
                ),
            )
            connection.execute(
                "UPDATE work_sessions SET updated_at = ? WHERE session_id = ?",
                (utc_timestamp(), stored.work_session_id),
            )
        return stored

    def list_events(
        self,
        work_session_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> list[JobEvent]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM job_events
                 WHERE work_session_id = ? AND sequence > ?
                 ORDER BY sequence ASC
                 LIMIT ?
                """,
                (work_session_id, max(0, after_sequence), max(1, min(limit, 5000))),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def add_steering(self, event: SteeringEvent) -> SteeringEvent:
        if self.get_work_session(event.work_session_id) is None:
            raise KeyError(f"unknown work session: {event.work_session_id}")
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO steering_events(
                    steering_id, work_session_id, job_id, kind, instruction,
                    apply_after, status, metadata_json, created_at, applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.steering_id,
                    event.work_session_id,
                    event.job_id,
                    event.kind.value,
                    event.instruction,
                    event.apply_after,
                    event.status,
                    encode_json(event.metadata),
                    event.created_at,
                    event.applied_at,
                ),
            )
        self.append_event(
            JobEvent.new(
                work_session_id=event.work_session_id,
                job_id=event.job_id,
                kind=EventKind.STEERING,
                message=event.instruction,
                payload={
                    "steering_id": event.steering_id,
                    "kind": event.kind.value,
                    "apply_after": event.apply_after,
                    "status": event.status,
                },
            )
        )
        return event

    def list_pending_steering(
        self,
        work_session_id: str,
        *,
        job_id: str | None = None,
    ) -> list[SteeringEvent]:
        query = (
            "SELECT * FROM steering_events "
            "WHERE work_session_id = ? AND status = 'pending'"
        )
        parameters: list[Any] = [work_session_id]
        if job_id is not None:
            query += " AND (job_id IS NULL OR job_id = ?)"
            parameters.append(job_id)
        query += " ORDER BY created_at ASC"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._steering_from_row(row) for row in rows]

    def mark_steering_applied(self, steering_id: str) -> SteeringEvent:
        now = utc_timestamp()
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE steering_events
                   SET status = 'applied', applied_at = ?
                 WHERE steering_id = ? AND status = 'pending'
                """,
                (now, steering_id),
            )
            row = connection.execute(
                "SELECT * FROM steering_events WHERE steering_id = ?",
                (steering_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown steering event: {steering_id}")
        return self._steering_from_row(row)

    # Idempotency and misc state

    def claim_interaction(
        self,
        correlation_id: str,
        *,
        work_session_id: str | None = None,
    ) -> bool:
        if not correlation_id:
            raise ValueError("correlation_id must not be empty")
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO processed_interactions(
                        correlation_id, work_session_id, result_json, created_at
                    ) VALUES (?, ?, NULL, ?)
                    """,
                    (correlation_id, work_session_id, utc_timestamp()),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def store_interaction_result(
        self,
        correlation_id: str,
        result: Mapping[str, Any],
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE processed_interactions
                   SET result_json = ?
                 WHERE correlation_id = ?
                """,
                (encode_json(result), correlation_id),
            )

    def get_interaction_result(self, correlation_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM processed_interactions
                 WHERE correlation_id = ?
                """,
                (correlation_id,),
            ).fetchone()
        if row is None or row["result_json"] is None:
            return None
        value = decode_json(row["result_json"], {})
        return dict(value) if isinstance(value, Mapping) else None

    def set_value(self, key: str, value: Mapping[str, Any] | list[Any] | str | int) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO platform_kv(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, encode_json(value if not isinstance(value, str) else {"value": value}), utc_timestamp()),
            )

    def get_value(self, key: str, default: Any = None) -> Any:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value_json FROM platform_kv WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return default
        return decode_json(row["value_json"], default)

    # Row conversion

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> Project:
        return Project.from_dict(
            {
                "project_id": row["project_id"],
                "domain": row["domain"],
                "title": row["title"],
                "description": row["description"],
                "status": row["status"],
                "metadata": decode_json(row["metadata_json"], {}),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> WorkSession:
        return WorkSession.from_dict(
            {
                "session_id": row["session_id"],
                "project_id": row["project_id"],
                "title": row["title"],
                "objective": row["objective"],
                "status": row["status"],
                "current_stage": row["current_stage"],
                "parent_session_id": row["parent_session_id"],
                "discord_guild_id": row["discord_guild_id"],
                "discord_parent_channel_id": row["discord_parent_channel_id"],
                "discord_thread_id": row["discord_thread_id"],
                "discord_live_message_id": row["discord_live_message_id"],
                "metadata": decode_json(row["metadata_json"], {}),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        spec = JobSpec.from_dict(
            {
                "job_id": row["job_id"],
                "work_session_id": row["work_session_id"],
                "parent_job_id": row["parent_job_id"],
                "domain": row["domain"],
                "task_type": row["task_type"],
                "entrypoint": row["entrypoint"],
                "backend_preferences": decode_json(
                    row["backend_preferences_json"], []
                ),
                "resources": decode_json(row["resources_json"], {}),
                "inputs": decode_json(row["inputs_json"], {}),
                "outputs": decode_json(row["outputs_json"], []),
                "metadata": decode_json(row["metadata_json"], {}),
                "created_at": row["created_at"],
            }
        )
        return JobRecord.from_dict(
            {
                "spec": spec.to_dict(),
                "status": row["status"],
                "backend": row["backend"],
                "backend_job_id": row["backend_job_id"],
                "current_stage": row["current_stage"],
                "progress": row["progress"],
                "result": decode_json(row["result_json"], {}),
                "error": row["error"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "updated_at": row["updated_at"],
            }
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> JobEvent:
        return JobEvent.from_dict(
            {
                "event_id": row["event_id"],
                "job_id": row["job_id"],
                "work_session_id": row["work_session_id"],
                "kind": row["kind"],
                "message": row["message"],
                "payload": decode_json(row["payload_json"], {}),
                "sequence": row["sequence"],
                "created_at": row["created_at"],
            }
        )

    @staticmethod
    def _steering_from_row(row: sqlite3.Row) -> SteeringEvent:
        return SteeringEvent.from_dict(
            {
                "steering_id": row["steering_id"],
                "work_session_id": row["work_session_id"],
                "job_id": row["job_id"],
                "kind": row["kind"],
                "instruction": row["instruction"],
                "apply_after": row["apply_after"],
                "status": row["status"],
                "metadata": decode_json(row["metadata_json"], {}),
                "created_at": row["created_at"],
                "applied_at": row["applied_at"],
            }
        )

    @staticmethod
    def _job_values(record: JobRecord) -> tuple[Any, ...]:
        spec = record.spec
        return (
            spec.job_id,
            spec.work_session_id,
            spec.parent_job_id,
            spec.domain.value,
            spec.task_type,
            spec.entrypoint,
            encode_json(list(spec.backend_preferences)),
            encode_json(spec.resources.to_dict()),
            encode_json(spec.inputs),
            encode_json(list(spec.outputs)),
            encode_json(spec.metadata),
            spec.created_at,
            record.status.value,
            record.backend,
            record.backend_job_id,
            record.current_stage,
            record.progress,
            encode_json(record.result),
            record.error,
            record.started_at,
            record.finished_at,
            record.updated_at,
        )
