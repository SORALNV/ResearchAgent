from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.state import utc_timestamp


KAGGLE_PHASES = {
    "competition_created",
    "rules_review",
    "data_preparation",
    "baseline_building",
    "cv_validation",
    "experimenting",
    "submission_review",
    "submitting",
    "lb_analysis",
    "competition_closed",
}
EXPERIMENT_STATUSES = {
    "planned",
    "queued",
    "smoke_running",
    "training",
    "review",
    "completed",
    "failed",
    "cancelled",
    "rejected",
}
SUBMISSION_STATUSES = {
    "prepared",
    "invalid",
    "waiting_approval",
    "approved",
    "submitted",
    "failed",
    "rejected",
}


@dataclass(frozen=True)
class KaggleCompetition:
    project_id: str
    competition_slug: str
    competition_url: str
    title: str
    phase: str
    metric: str | None
    target_column: str | None
    id_column: str | None
    sample_submission_path: str | None
    dataset_fingerprint: str | None
    rules_confirmed: bool
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CVSpec:
    cv_spec_id: str
    project_id: str
    strategy: str
    n_splits: int
    shuffle: bool
    seed: int | None
    metric: str
    group_column: str | None
    time_column: str | None
    parameters: dict[str, Any]
    split_hash: str | None
    approved: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    project_id: str
    work_session_id: str
    parent_experiment_id: str | None
    hypothesis: str
    status: str
    config: dict[str, Any]
    code_snapshot: str | None
    dataset_fingerprint: str | None
    cv_spec_id: str | None
    job_id: str | None
    metrics: dict[str, Any]
    artifacts: list[dict[str, Any]]
    failure_reason: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SubmissionCandidate:
    candidate_id: str
    project_id: str
    experiment_id: str
    file_path: str
    sha256: str
    cv_score: float | None
    message: str
    validation: dict[str, Any]
    status: str
    approval_id: str | None
    kaggle_submission_ref: str | None
    created_at: str
    updated_at: str


class KaggleStore:
    """Kaggle-specific state stored alongside the generic control-plane DB."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def create_competition(
        self,
        *,
        project_id: str,
        competition_slug: str,
        competition_url: str,
        title: str,
        metric: str | None = None,
        target_column: str | None = None,
        id_column: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KaggleCompetition:
        slug = _normalize_slug(competition_slug)
        now = utc_timestamp()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO kaggle_competitions(
                    project_id, competition_slug, competition_url, title,
                    phase, metric, target_column, id_column,
                    sample_submission_path, dataset_fingerprint,
                    rules_confirmed, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'rules_review', ?, ?, ?, NULL, NULL, 0, ?, ?, ?)
                """,
                (
                    project_id,
                    slug,
                    competition_url,
                    title.strip() or slug,
                    metric,
                    target_column,
                    id_column,
                    _json(metadata or {}),
                    now,
                    now,
                ),
            )
        result = self.get_competition(project_id)
        assert result is not None
        return result

    def get_competition(self, project_id: str) -> KaggleCompetition | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM kaggle_competitions WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        finally:
            connection.close()
        return _competition_from_row(row) if row else None

    def find_competition_by_slug(self, slug: str) -> KaggleCompetition | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM kaggle_competitions WHERE competition_slug = ?",
                (_normalize_slug(slug),),
            ).fetchone()
        finally:
            connection.close()
        return _competition_from_row(row) if row else None

    def update_competition(
        self,
        project_id: str,
        *,
        phase: str | None = None,
        metric: str | None = None,
        target_column: str | None = None,
        id_column: str | None = None,
        sample_submission_path: str | None = None,
        dataset_fingerprint: str | None = None,
        rules_confirmed: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KaggleCompetition:
        current = self.get_competition(project_id)
        if current is None:
            raise KeyError(project_id)
        new_phase = phase or current.phase
        if new_phase not in KAGGLE_PHASES:
            raise ValueError(f"invalid Kaggle phase: {new_phase}")
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE kaggle_competitions
                SET phase = ?, metric = ?, target_column = ?, id_column = ?,
                    sample_submission_path = ?, dataset_fingerprint = ?,
                    rules_confirmed = ?, metadata_json = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (
                    new_phase,
                    metric if metric is not None else current.metric,
                    target_column if target_column is not None else current.target_column,
                    id_column if id_column is not None else current.id_column,
                    (
                        sample_submission_path
                        if sample_submission_path is not None
                        else current.sample_submission_path
                    ),
                    (
                        dataset_fingerprint
                        if dataset_fingerprint is not None
                        else current.dataset_fingerprint
                    ),
                    (
                        int(rules_confirmed)
                        if rules_confirmed is not None
                        else int(current.rules_confirmed)
                    ),
                    _json(metadata if metadata is not None else current.metadata),
                    utc_timestamp(),
                    project_id,
                ),
            )
        result = self.get_competition(project_id)
        assert result is not None
        return result

    def create_cv_spec(
        self,
        *,
        project_id: str,
        strategy: str,
        n_splits: int,
        metric: str,
        shuffle: bool = True,
        seed: int | None = 42,
        group_column: str | None = None,
        time_column: str | None = None,
        parameters: dict[str, Any] | None = None,
        split_hash: str | None = None,
        cv_spec_id: str | None = None,
    ) -> CVSpec:
        if self.get_competition(project_id) is None:
            raise KeyError(project_id)
        if int(n_splits) < 2:
            raise ValueError("n_splits must be >= 2")
        now = utc_timestamp()
        identifier = cv_spec_id or _new_id("CV", 6)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO kaggle_cv_specs(
                    cv_spec_id, project_id, strategy, n_splits, shuffle,
                    seed, metric, group_column, time_column, parameters_json,
                    split_hash, approved, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    identifier,
                    project_id,
                    strategy.strip(),
                    int(n_splits),
                    int(bool(shuffle)),
                    seed,
                    metric.strip(),
                    group_column,
                    time_column,
                    _json(parameters or {}),
                    split_hash,
                    now,
                    now,
                ),
            )
        result = self.get_cv_spec(identifier)
        assert result is not None
        return result

    def get_cv_spec(self, cv_spec_id: str) -> CVSpec | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM kaggle_cv_specs WHERE cv_spec_id = ?",
                (cv_spec_id,),
            ).fetchone()
        finally:
            connection.close()
        return _cv_from_row(row) if row else None

    def approve_cv_spec(self, cv_spec_id: str, split_hash: str | None = None) -> CVSpec:
        with self._transaction() as connection:
            updated = connection.execute(
                """
                UPDATE kaggle_cv_specs
                SET approved = 1, split_hash = COALESCE(?, split_hash), updated_at = ?
                WHERE cv_spec_id = ?
                """,
                (split_hash, utc_timestamp(), cv_spec_id),
            ).rowcount
        if not updated:
            raise KeyError(cv_spec_id)
        result = self.get_cv_spec(cv_spec_id)
        assert result is not None
        return result

    def create_experiment(
        self,
        *,
        project_id: str,
        work_session_id: str,
        hypothesis: str,
        config: dict[str, Any] | None = None,
        parent_experiment_id: str | None = None,
        code_snapshot: str | None = None,
        dataset_fingerprint: str | None = None,
        cv_spec_id: str | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        competition = self.get_competition(project_id)
        if competition is None:
            raise KeyError(project_id)
        if parent_experiment_id and self.get_experiment(parent_experiment_id) is None:
            raise KeyError(parent_experiment_id)
        if cv_spec_id:
            cv = self.get_cv_spec(cv_spec_id)
            if cv is None or cv.project_id != project_id:
                raise ValueError("cv_spec_id does not belong to project")
        now = utc_timestamp()
        identifier = experiment_id or _next_experiment_id(
            self._connect,
            project_id,
        )
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO kaggle_experiments(
                    experiment_id, project_id, work_session_id,
                    parent_experiment_id, hypothesis, status, config_json,
                    code_snapshot, dataset_fingerprint, cv_spec_id, job_id,
                    metrics_json, artifacts_json, failure_reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?, NULL, '{}', '[]', NULL, ?, ?)
                """,
                (
                    identifier,
                    project_id,
                    work_session_id,
                    parent_experiment_id,
                    hypothesis.strip(),
                    _json(config or {}),
                    code_snapshot,
                    dataset_fingerprint or competition.dataset_fingerprint,
                    cv_spec_id,
                    now,
                    now,
                ),
            )
        result = self.get_experiment(identifier)
        assert result is not None
        return result

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM kaggle_experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        finally:
            connection.close()
        return _experiment_from_row(row) if row else None

    def list_experiments(self, project_id: str) -> list[ExperimentRecord]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM kaggle_experiments
                WHERE project_id = ? ORDER BY created_at
                """,
                (project_id,),
            ).fetchall()
        finally:
            connection.close()
        return [_experiment_from_row(row) for row in rows]

    def update_experiment(
        self,
        experiment_id: str,
        *,
        status: str | None = None,
        job_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        failure_reason: str | None = None,
    ) -> ExperimentRecord:
        current = self.get_experiment(experiment_id)
        if current is None:
            raise KeyError(experiment_id)
        new_status = status or current.status
        if new_status not in EXPERIMENT_STATUSES:
            raise ValueError(f"invalid experiment status: {new_status}")
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE kaggle_experiments
                SET status = ?, job_id = ?, metrics_json = ?, artifacts_json = ?,
                    failure_reason = ?, updated_at = ?
                WHERE experiment_id = ?
                """,
                (
                    new_status,
                    job_id if job_id is not None else current.job_id,
                    _json(metrics if metrics is not None else current.metrics),
                    _json(artifacts if artifacts is not None else current.artifacts),
                    (
                        failure_reason
                        if failure_reason is not None
                        else current.failure_reason
                    ),
                    utc_timestamp(),
                    experiment_id,
                ),
            )
        result = self.get_experiment(experiment_id)
        assert result is not None
        return result

    def create_submission_candidate(
        self,
        *,
        project_id: str,
        experiment_id: str,
        file_path: str | Path,
        sha256: str,
        validation: dict[str, Any],
        message: str,
        cv_score: float | None = None,
        candidate_id: str | None = None,
    ) -> SubmissionCandidate:
        experiment = self.get_experiment(experiment_id)
        if experiment is None or experiment.project_id != project_id:
            raise ValueError("experiment does not belong to project")
        valid = bool(validation.get("valid", False))
        status = "waiting_approval" if valid else "invalid"
        now = utc_timestamp()
        identifier = candidate_id or _new_id("SUBC", 8)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO kaggle_submission_candidates(
                    candidate_id, project_id, experiment_id, file_path, sha256,
                    cv_score, message, validation_json, status, approval_id,
                    kaggle_submission_ref, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    identifier,
                    project_id,
                    experiment_id,
                    str(file_path),
                    sha256,
                    cv_score,
                    message[:100],
                    _json(validation),
                    status,
                    now,
                    now,
                ),
            )
        result = self.get_submission_candidate(identifier)
        assert result is not None
        return result

    def get_submission_candidate(self, candidate_id: str) -> SubmissionCandidate | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM kaggle_submission_candidates
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
        finally:
            connection.close()
        return _submission_from_row(row) if row else None

    def update_submission_candidate(
        self,
        candidate_id: str,
        *,
        status: str,
        approval_id: str | None = None,
        kaggle_submission_ref: str | None = None,
    ) -> SubmissionCandidate:
        if status not in SUBMISSION_STATUSES:
            raise ValueError(f"invalid submission status: {status}")
        current = self.get_submission_candidate(candidate_id)
        if current is None:
            raise KeyError(candidate_id)
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE kaggle_submission_candidates
                SET status = ?, approval_id = ?, kaggle_submission_ref = ?, updated_at = ?
                WHERE candidate_id = ?
                """,
                (
                    status,
                    approval_id if approval_id is not None else current.approval_id,
                    (
                        kaggle_submission_ref
                        if kaggle_submission_ref is not None
                        else current.kaggle_submission_ref
                    ),
                    utc_timestamp(),
                    candidate_id,
                ),
            )
        result = self.get_submission_candidate(candidate_id)
        assert result is not None
        return result

    def _initialize(self) -> None:
        with self._transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS kaggle_competitions(
                    project_id TEXT PRIMARY KEY,
                    competition_slug TEXT NOT NULL UNIQUE,
                    competition_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    metric TEXT,
                    target_column TEXT,
                    id_column TEXT,
                    sample_submission_path TEXT,
                    dataset_fingerprint TEXT,
                    rules_confirmed INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kaggle_cv_specs(
                    cv_spec_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    n_splits INTEGER NOT NULL,
                    shuffle INTEGER NOT NULL,
                    seed INTEGER,
                    metric TEXT NOT NULL,
                    group_column TEXT,
                    time_column TEXT,
                    parameters_json TEXT NOT NULL,
                    split_hash TEXT,
                    approved INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kaggle_experiments(
                    experiment_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    work_session_id TEXT NOT NULL,
                    parent_experiment_id TEXT,
                    hypothesis TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    code_snapshot TEXT,
                    dataset_fingerprint TEXT,
                    cv_spec_id TEXT,
                    job_id TEXT,
                    metrics_json TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    failure_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kaggle_submission_candidates(
                    candidate_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    cv_score REAL,
                    message TEXT NOT NULL,
                    validation_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    approval_id TEXT,
                    kaggle_submission_ref TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_kaggle_experiments_project
                    ON kaggle_experiments(project_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_kaggle_submissions_project
                    ON kaggle_submission_candidates(project_id, created_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _transaction(self):
        store = self

        class Transaction:
            def __enter__(self_nonlocal):
                store._lock.acquire()
                self_nonlocal.connection = store._connect()
                self_nonlocal.connection.execute("BEGIN IMMEDIATE")
                return self_nonlocal.connection

            def __exit__(self_nonlocal, exc_type, exc, traceback):
                try:
                    if exc_type is None:
                        self_nonlocal.connection.commit()
                    else:
                        self_nonlocal.connection.rollback()
                finally:
                    self_nonlocal.connection.close()
                    store._lock.release()
                return False

        return Transaction()


def _normalize_slug(value: str) -> str:
    value = value.strip().strip("/")
    if not value or "/" in value or " " in value:
        raise ValueError("competition_slug must be a single Kaggle slug")
    return value


def _new_id(prefix: str, length: int) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:length].upper()}"


def _next_experiment_id(connect, project_id: str) -> str:
    connection = connect()
    try:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM kaggle_experiments WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    finally:
        connection.close()
    return f"EXP-{int(row['count']) + 1:04d}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _competition_from_row(row: sqlite3.Row) -> KaggleCompetition:
    return KaggleCompetition(
        project_id=str(row["project_id"]),
        competition_slug=str(row["competition_slug"]),
        competition_url=str(row["competition_url"]),
        title=str(row["title"]),
        phase=str(row["phase"]),
        metric=str(row["metric"]) if row["metric"] is not None else None,
        target_column=(
            str(row["target_column"]) if row["target_column"] is not None else None
        ),
        id_column=str(row["id_column"]) if row["id_column"] is not None else None,
        sample_submission_path=(
            str(row["sample_submission_path"])
            if row["sample_submission_path"] is not None
            else None
        ),
        dataset_fingerprint=(
            str(row["dataset_fingerprint"])
            if row["dataset_fingerprint"] is not None
            else None
        ),
        rules_confirmed=bool(row["rules_confirmed"]),
        metadata=dict(_loads(row["metadata_json"], {})),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _cv_from_row(row: sqlite3.Row) -> CVSpec:
    return CVSpec(
        cv_spec_id=str(row["cv_spec_id"]),
        project_id=str(row["project_id"]),
        strategy=str(row["strategy"]),
        n_splits=int(row["n_splits"]),
        shuffle=bool(row["shuffle"]),
        seed=int(row["seed"]) if row["seed"] is not None else None,
        metric=str(row["metric"]),
        group_column=(str(row["group_column"]) if row["group_column"] is not None else None),
        time_column=(str(row["time_column"]) if row["time_column"] is not None else None),
        parameters=dict(_loads(row["parameters_json"], {})),
        split_hash=str(row["split_hash"]) if row["split_hash"] is not None else None,
        approved=bool(row["approved"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _experiment_from_row(row: sqlite3.Row) -> ExperimentRecord:
    artifacts = _loads(row["artifacts_json"], [])
    return ExperimentRecord(
        experiment_id=str(row["experiment_id"]),
        project_id=str(row["project_id"]),
        work_session_id=str(row["work_session_id"]),
        parent_experiment_id=(
            str(row["parent_experiment_id"])
            if row["parent_experiment_id"] is not None
            else None
        ),
        hypothesis=str(row["hypothesis"]),
        status=str(row["status"]),
        config=dict(_loads(row["config_json"], {})),
        code_snapshot=(
            str(row["code_snapshot"]) if row["code_snapshot"] is not None else None
        ),
        dataset_fingerprint=(
            str(row["dataset_fingerprint"])
            if row["dataset_fingerprint"] is not None
            else None
        ),
        cv_spec_id=str(row["cv_spec_id"]) if row["cv_spec_id"] is not None else None,
        job_id=str(row["job_id"]) if row["job_id"] is not None else None,
        metrics=dict(_loads(row["metrics_json"], {})),
        artifacts=[dict(item) for item in artifacts if isinstance(item, dict)],
        failure_reason=(
            str(row["failure_reason"]) if row["failure_reason"] is not None else None
        ),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _submission_from_row(row: sqlite3.Row) -> SubmissionCandidate:
    return SubmissionCandidate(
        candidate_id=str(row["candidate_id"]),
        project_id=str(row["project_id"]),
        experiment_id=str(row["experiment_id"]),
        file_path=str(row["file_path"]),
        sha256=str(row["sha256"]),
        cv_score=float(row["cv_score"]) if row["cv_score"] is not None else None,
        message=str(row["message"]),
        validation=dict(_loads(row["validation_json"], {})),
        status=str(row["status"]),
        approval_id=str(row["approval_id"]) if row["approval_id"] is not None else None,
        kaggle_submission_ref=(
            str(row["kaggle_submission_ref"])
            if row["kaggle_submission_ref"] is not None
            else None
        ),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
