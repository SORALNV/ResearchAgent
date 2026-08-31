from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from harness.domains.kaggle.models import (
    CVSpec,
    CompetitionPhase,
    ExperimentRecord,
    ExperimentStatus,
    KaggleCompetitionState,
    SubmissionCandidate,
    SubmissionStatus,
)
from harness.state import utc_timestamp


class KaggleRegistry:
    """Durable Kaggle-specific source of truth.

    PlatformRegistry tracks generic projects, sessions, jobs, and events. This
    registry tracks competition rules/CV contracts/experiments/submissions and
    can share the same SQLite file without coupling the generic platform schema.
    """

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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS kaggle_competitions (
                    competition_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL UNIQUE,
                    slug TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    evaluation_metric TEXT NOT NULL,
                    target_columns_json TEXT NOT NULL,
                    id_columns_json TEXT NOT NULL,
                    rules_acknowledged INTEGER NOT NULL,
                    rules_hash TEXT,
                    dataset_fingerprint TEXT,
                    active_cv_spec_id TEXT,
                    best_experiment_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kaggle_cv_specs (
                    cv_spec_id TEXT PRIMARY KEY,
                    competition_id TEXT NOT NULL REFERENCES kaggle_competitions(competition_id),
                    strategy TEXT NOT NULL,
                    n_splits INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    shuffle INTEGER NOT NULL,
                    group_column TEXT,
                    time_column TEXT,
                    stratify_column TEXT,
                    locked INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_cv_competition
                    ON kaggle_cv_specs(competition_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS kaggle_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    competition_id TEXT NOT NULL REFERENCES kaggle_competitions(competition_id),
                    parent_experiment_id TEXT REFERENCES kaggle_experiments(experiment_id),
                    hypothesis TEXT NOT NULL,
                    status TEXT NOT NULL,
                    work_session_id TEXT,
                    job_id TEXT,
                    cv_spec_id TEXT REFERENCES kaggle_cv_specs(cv_spec_id),
                    dataset_fingerprint TEXT,
                    code_snapshot TEXT,
                    config_json TEXT NOT NULL,
                    config_diff_json TEXT NOT NULL,
                    fold_scores_json TEXT NOT NULL,
                    cv_mean REAL,
                    cv_std REAL,
                    runtime_seconds REAL,
                    backend TEXT,
                    artifact_manifest_path TEXT,
                    review_json TEXT NOT NULL,
                    failure_reason TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_exp_competition
                    ON kaggle_experiments(competition_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_exp_parent
                    ON kaggle_experiments(parent_experiment_id);

                CREATE TABLE IF NOT EXISTS kaggle_submissions (
                    candidate_id TEXT PRIMARY KEY,
                    competition_id TEXT NOT NULL REFERENCES kaggle_competitions(competition_id),
                    experiment_id TEXT NOT NULL REFERENCES kaggle_experiments(experiment_id),
                    file_path TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    cv_score REAL,
                    previous_best_cv REAL,
                    validation_json TEXT NOT NULL,
                    risks_json TEXT NOT NULL,
                    approval_id TEXT,
                    submitted_at TEXT,
                    public_score REAL,
                    private_score REAL,
                    kaggle_ref TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_submission_hash_once
                    ON kaggle_submissions(competition_id, file_sha256, status)
                    WHERE status IN ('approved', 'submitted', 'accepted');
                CREATE INDEX IF NOT EXISTS idx_submission_competition
                    ON kaggle_submissions(competition_id, created_at DESC);
                """
            )

    def create_competition(
        self, competition: KaggleCompetitionState
    ) -> KaggleCompetitionState:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO kaggle_competitions VALUES(
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    competition.competition_id,
                    competition.project_id,
                    competition.slug,
                    competition.url,
                    competition.title,
                    competition.phase.value,
                    competition.evaluation_metric,
                    _json(list(competition.target_columns)),
                    _json(list(competition.id_columns)),
                    int(competition.rules_acknowledged),
                    competition.rules_hash,
                    competition.dataset_fingerprint,
                    competition.active_cv_spec_id,
                    competition.best_experiment_id,
                    _json(competition.metadata),
                    competition.created_at,
                    competition.updated_at,
                ),
            )
        return competition

    def get_competition(
        self,
        competition_id: str | None = None,
        *,
        project_id: str | None = None,
    ) -> KaggleCompetitionState | None:
        if not competition_id and not project_id:
            raise ValueError("competition_id or project_id is required")
        column, value = (
            ("competition_id", competition_id)
            if competition_id
            else ("project_id", project_id)
        )
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT * FROM kaggle_competitions WHERE {column} = ?", (value,)
            ).fetchone()
        return self._competition(row) if row else None

    def update_competition(
        self,
        competition_id: str,
        **changes: Any,
    ) -> KaggleCompetitionState:
        current = self.get_competition(competition_id)
        if current is None:
            raise KeyError(competition_id)
        allowed = {
            "phase",
            "evaluation_metric",
            "target_columns",
            "id_columns",
            "rules_acknowledged",
            "rules_hash",
            "dataset_fingerprint",
            "active_cv_spec_id",
            "best_experiment_id",
            "metadata",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown competition fields: {sorted(unknown)}")
        normalized = dict(changes)
        if "phase" in normalized:
            normalized["phase"] = CompetitionPhase(str(normalized["phase"]))
        if "target_columns" in normalized:
            normalized["target_columns"] = tuple(normalized["target_columns"])
        if "id_columns" in normalized:
            normalized["id_columns"] = tuple(normalized["id_columns"])
        if "metadata" in normalized:
            normalized["metadata"] = dict(normalized["metadata"])
        updated = replace(current, **normalized, updated_at=utc_timestamp())
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE kaggle_competitions SET
                    phase = ?, evaluation_metric = ?, target_columns_json = ?,
                    id_columns_json = ?, rules_acknowledged = ?, rules_hash = ?,
                    dataset_fingerprint = ?, active_cv_spec_id = ?,
                    best_experiment_id = ?, metadata_json = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (
                    updated.phase.value,
                    updated.evaluation_metric,
                    _json(list(updated.target_columns)),
                    _json(list(updated.id_columns)),
                    int(updated.rules_acknowledged),
                    updated.rules_hash,
                    updated.dataset_fingerprint,
                    updated.active_cv_spec_id,
                    updated.best_experiment_id,
                    _json(updated.metadata),
                    updated.updated_at,
                    competition_id,
                ),
            )
        return updated

    def create_cv_spec(self, spec: CVSpec) -> CVSpec:
        if self.get_competition(spec.competition_id) is None:
            raise KeyError(spec.competition_id)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO kaggle_cv_specs VALUES(
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    spec.cv_spec_id,
                    spec.competition_id,
                    spec.strategy,
                    spec.n_splits,
                    spec.metric,
                    spec.seed,
                    int(spec.shuffle),
                    spec.group_column,
                    spec.time_column,
                    spec.stratify_column,
                    int(spec.locked),
                    _json(spec.metadata),
                    spec.created_at,
                ),
            )
        return spec

    def get_cv_spec(self, cv_spec_id: str) -> CVSpec | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM kaggle_cv_specs WHERE cv_spec_id = ?", (cv_spec_id,)
            ).fetchone()
        return self._cv(row) if row else None

    def lock_cv_spec(self, cv_spec_id: str) -> CVSpec:
        current = self.get_cv_spec(cv_spec_id)
        if current is None:
            raise KeyError(cv_spec_id)
        with self._transaction() as connection:
            connection.execute(
                "UPDATE kaggle_cv_specs SET locked = 1 WHERE cv_spec_id = ?",
                (cv_spec_id,),
            )
        return replace(current, locked=True)

    def create_experiment(self, experiment: ExperimentRecord) -> ExperimentRecord:
        if self.get_competition(experiment.competition_id) is None:
            raise KeyError(experiment.competition_id)
        if experiment.parent_experiment_id and not self.get_experiment(
            experiment.parent_experiment_id
        ):
            raise KeyError(experiment.parent_experiment_id)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO kaggle_experiments VALUES(
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                self._experiment_values(experiment),
            )
        return experiment

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM kaggle_experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return self._experiment(row) if row else None

    def list_experiments(
        self, competition_id: str, *, limit: int = 500
    ) -> list[ExperimentRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM kaggle_experiments
                WHERE competition_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (competition_id, max(1, min(limit, 5000))),
            ).fetchall()
        return [self._experiment(row) for row in rows]

    def update_experiment(
        self,
        experiment_id: str,
        **changes: Any,
    ) -> ExperimentRecord:
        current = self.get_experiment(experiment_id)
        if current is None:
            raise KeyError(experiment_id)
        allowed = set(current.to_dict()) - {
            "experiment_id",
            "competition_id",
            "parent_experiment_id",
            "created_at",
            "updated_at",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown experiment fields: {sorted(unknown)}")
        normalized = dict(changes)
        if "status" in normalized:
            normalized["status"] = ExperimentStatus(str(normalized["status"]))
        if "fold_scores" in normalized:
            normalized["fold_scores"] = tuple(float(item) for item in normalized["fold_scores"])
        for key in ("config", "config_diff", "review", "metadata"):
            if key in normalized:
                normalized[key] = dict(normalized[key])
        updated = replace(current, **normalized, updated_at=utc_timestamp())
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE kaggle_experiments SET
                    hypothesis = ?, status = ?, work_session_id = ?, job_id = ?,
                    cv_spec_id = ?, dataset_fingerprint = ?, code_snapshot = ?,
                    config_json = ?, config_diff_json = ?, fold_scores_json = ?,
                    cv_mean = ?, cv_std = ?, runtime_seconds = ?, backend = ?,
                    artifact_manifest_path = ?, review_json = ?, failure_reason = ?,
                    metadata_json = ?, updated_at = ?
                WHERE experiment_id = ?
                """,
                (
                    updated.hypothesis,
                    updated.status.value,
                    updated.work_session_id,
                    updated.job_id,
                    updated.cv_spec_id,
                    updated.dataset_fingerprint,
                    updated.code_snapshot,
                    _json(updated.config),
                    _json(updated.config_diff),
                    _json(list(updated.fold_scores)),
                    updated.cv_mean,
                    updated.cv_std,
                    updated.runtime_seconds,
                    updated.backend,
                    updated.artifact_manifest_path,
                    _json(updated.review),
                    updated.failure_reason,
                    _json(updated.metadata),
                    updated.updated_at,
                    experiment_id,
                ),
            )
        return updated

    def create_submission(
        self, candidate: SubmissionCandidate
    ) -> SubmissionCandidate:
        if self.get_experiment(candidate.experiment_id) is None:
            raise KeyError(candidate.experiment_id)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO kaggle_submissions VALUES(
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                self._submission_values(candidate),
            )
        return candidate

    def get_submission(self, candidate_id: str) -> SubmissionCandidate | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM kaggle_submissions WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._submission(row) if row else None

    def update_submission(
        self, candidate_id: str, **changes: Any
    ) -> SubmissionCandidate:
        current = self.get_submission(candidate_id)
        if current is None:
            raise KeyError(candidate_id)
        allowed = set(current.to_dict()) - {
            "candidate_id",
            "competition_id",
            "experiment_id",
            "file_sha256",
            "created_at",
            "updated_at",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown submission fields: {sorted(unknown)}")
        normalized = dict(changes)
        if "status" in normalized:
            normalized["status"] = SubmissionStatus(str(normalized["status"]))
        if "risks" in normalized:
            normalized["risks"] = tuple(normalized["risks"])
        for key in ("validation", "metadata"):
            if key in normalized:
                normalized[key] = dict(normalized[key])
        updated = replace(current, **normalized, updated_at=utc_timestamp())
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE kaggle_submissions SET
                    file_path = ?, status = ?, message = ?, cv_score = ?,
                    previous_best_cv = ?, validation_json = ?, risks_json = ?,
                    approval_id = ?, submitted_at = ?, public_score = ?,
                    private_score = ?, kaggle_ref = ?, metadata_json = ?,
                    updated_at = ?
                WHERE candidate_id = ?
                """,
                (
                    updated.file_path,
                    updated.status.value,
                    updated.message,
                    updated.cv_score,
                    updated.previous_best_cv,
                    _json(updated.validation),
                    _json(list(updated.risks)),
                    updated.approval_id,
                    updated.submitted_at,
                    updated.public_score,
                    updated.private_score,
                    updated.kaggle_ref,
                    _json(updated.metadata),
                    updated.updated_at,
                    candidate_id,
                ),
            )
        return updated

    @staticmethod
    def _competition(row: sqlite3.Row) -> KaggleCompetitionState:
        return KaggleCompetitionState.from_dict(
            {
                **dict(row),
                "target_columns": _loads(row["target_columns_json"], []),
                "id_columns": _loads(row["id_columns_json"], []),
                "rules_acknowledged": bool(row["rules_acknowledged"]),
                "metadata": _loads(row["metadata_json"], {}),
            }
        )

    @staticmethod
    def _cv(row: sqlite3.Row) -> CVSpec:
        return CVSpec.from_dict(
            {
                **dict(row),
                "shuffle": bool(row["shuffle"]),
                "locked": bool(row["locked"]),
                "metadata": _loads(row["metadata_json"], {}),
            }
        )

    @staticmethod
    def _experiment(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord.from_dict(
            {
                **dict(row),
                "config": _loads(row["config_json"], {}),
                "config_diff": _loads(row["config_diff_json"], {}),
                "fold_scores": _loads(row["fold_scores_json"], []),
                "review": _loads(row["review_json"], {}),
                "metadata": _loads(row["metadata_json"], {}),
            }
        )

    @staticmethod
    def _submission(row: sqlite3.Row) -> SubmissionCandidate:
        return SubmissionCandidate.from_dict(
            {
                **dict(row),
                "validation": _loads(row["validation_json"], {}),
                "risks": _loads(row["risks_json"], []),
                "metadata": _loads(row["metadata_json"], {}),
            }
        )

    @staticmethod
    def _experiment_values(value: ExperimentRecord) -> tuple[Any, ...]:
        return (
            value.experiment_id,
            value.competition_id,
            value.parent_experiment_id,
            value.hypothesis,
            value.status.value,
            value.work_session_id,
            value.job_id,
            value.cv_spec_id,
            value.dataset_fingerprint,
            value.code_snapshot,
            _json(value.config),
            _json(value.config_diff),
            _json(list(value.fold_scores)),
            value.cv_mean,
            value.cv_std,
            value.runtime_seconds,
            value.backend,
            value.artifact_manifest_path,
            _json(value.review),
            value.failure_reason,
            _json(value.metadata),
            value.created_at,
            value.updated_at,
        )

    @staticmethod
    def _submission_values(value: SubmissionCandidate) -> tuple[Any, ...]:
        return (
            value.candidate_id,
            value.competition_id,
            value.experiment_id,
            value.file_path,
            value.file_sha256,
            value.status.value,
            value.message,
            value.cv_score,
            value.previous_best_cv,
            _json(value.validation),
            _json(list(value.risks)),
            value.approval_id,
            value.submitted_at,
            value.public_score,
            value.private_score,
            value.kaggle_ref,
            _json(value.metadata),
            value.created_at,
            value.updated_at,
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback
