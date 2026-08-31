from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from harness.state import utc_timestamp


class CompetitionPhase(StrEnum):
    CREATED = "created"
    RULES_REVIEW = "rules_review"
    DATA_PREPARATION = "data_preparation"
    BASELINE_BUILDING = "baseline_building"
    CV_VALIDATION = "cv_validation"
    EXPERIMENTING = "experimenting"
    SUBMISSION_REVIEW = "submission_review"
    SUBMITTING = "submitting"
    LB_ANALYSIS = "lb_analysis"
    CLOSED = "closed"


class ExperimentStatus(StrEnum):
    PROPOSED = "proposed"
    SMOKE_TEST = "smoke_test"
    RUNNING = "running"
    REVIEW = "review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubmissionStatus(StrEnum):
    CANDIDATE = "candidate"
    INVALID = "invalid"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class KaggleCompetitionState:
    competition_id: str
    project_id: str
    slug: str
    url: str
    title: str
    phase: CompetitionPhase = CompetitionPhase.CREATED
    evaluation_metric: str = ""
    target_columns: tuple[str, ...] = ()
    id_columns: tuple[str, ...] = ()
    rules_acknowledged: bool = False
    rules_hash: str | None = None
    dataset_fingerprint: str | None = None
    active_cv_spec_id: str | None = None
    best_experiment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        slug: str,
        url: str,
        title: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "KaggleCompetitionState":
        normalized = slug.strip().strip("/")
        if not normalized:
            raise ValueError("competition slug must not be empty")
        return cls(
            competition_id=f"COMP-{uuid.uuid4().hex[:10].upper()}",
            project_id=project_id,
            slug=normalized,
            url=url.strip(),
            title=title.strip() or normalized,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        payload["target_columns"] = list(self.target_columns)
        payload["id_columns"] = list(self.id_columns)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KaggleCompetitionState":
        return cls(
            competition_id=str(data["competition_id"]),
            project_id=str(data["project_id"]),
            slug=str(data.get("slug") or ""),
            url=str(data.get("url") or ""),
            title=str(data.get("title") or ""),
            phase=CompetitionPhase(
                str(data.get("phase") or CompetitionPhase.CREATED.value)
            ),
            evaluation_metric=str(data.get("evaluation_metric") or ""),
            target_columns=tuple(str(item) for item in data.get("target_columns", [])),
            id_columns=tuple(str(item) for item in data.get("id_columns", [])),
            rules_acknowledged=bool(data.get("rules_acknowledged", False)),
            rules_hash=_optional(data.get("rules_hash")),
            dataset_fingerprint=_optional(data.get("dataset_fingerprint")),
            active_cv_spec_id=_optional(data.get("active_cv_spec_id")),
            best_experiment_id=_optional(data.get("best_experiment_id")),
            metadata=_dict(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_timestamp()),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class CVSpec:
    cv_spec_id: str
    competition_id: str
    strategy: str
    n_splits: int
    metric: str
    seed: int = 42
    shuffle: bool = True
    group_column: str | None = None
    time_column: str | None = None
    stratify_column: str | None = None
    locked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)

    @classmethod
    def new(
        cls,
        *,
        competition_id: str,
        strategy: str,
        n_splits: int,
        metric: str,
        seed: int = 42,
        shuffle: bool = True,
        group_column: str | None = None,
        time_column: str | None = None,
        stratify_column: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "CVSpec":
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if group_column and time_column:
            raise ValueError("group_column and time_column require an explicit custom strategy")
        return cls(
            cv_spec_id=f"CV-{uuid.uuid4().hex[:10].upper()}",
            competition_id=competition_id,
            strategy=strategy.strip(),
            n_splits=n_splits,
            metric=metric.strip(),
            seed=seed,
            shuffle=shuffle,
            group_column=group_column,
            time_column=time_column,
            stratify_column=stratify_column,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CVSpec":
        return cls(
            cv_spec_id=str(data["cv_spec_id"]),
            competition_id=str(data["competition_id"]),
            strategy=str(data.get("strategy") or ""),
            n_splits=int(data.get("n_splits") or 0),
            metric=str(data.get("metric") or ""),
            seed=int(data.get("seed") or 42),
            shuffle=bool(data.get("shuffle", True)),
            group_column=_optional(data.get("group_column")),
            time_column=_optional(data.get("time_column")),
            stratify_column=_optional(data.get("stratify_column")),
            locked=bool(data.get("locked", False)),
            metadata=_dict(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    competition_id: str
    hypothesis: str
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    parent_experiment_id: str | None = None
    work_session_id: str | None = None
    job_id: str | None = None
    cv_spec_id: str | None = None
    dataset_fingerprint: str | None = None
    code_snapshot: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    config_diff: dict[str, Any] = field(default_factory=dict)
    fold_scores: tuple[float, ...] = ()
    cv_mean: float | None = None
    cv_std: float | None = None
    runtime_seconds: float | None = None
    backend: str | None = None
    artifact_manifest_path: str | None = None
    review: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    @classmethod
    def new(
        cls,
        *,
        competition_id: str,
        hypothesis: str,
        parent_experiment_id: str | None = None,
        work_session_id: str | None = None,
        cv_spec_id: str | None = None,
        dataset_fingerprint: str | None = None,
        config: Mapping[str, Any] | None = None,
        config_diff: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExperimentRecord":
        if not hypothesis.strip():
            raise ValueError("experiment hypothesis must not be empty")
        return cls(
            experiment_id=f"EXP-{uuid.uuid4().hex[:10].upper()}",
            competition_id=competition_id,
            hypothesis=hypothesis.strip(),
            parent_experiment_id=parent_experiment_id,
            work_session_id=work_session_id,
            cv_spec_id=cv_spec_id,
            dataset_fingerprint=dataset_fingerprint,
            config=dict(config or {}),
            config_diff=dict(config_diff or {}),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["fold_scores"] = list(self.fold_scores)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=str(data["experiment_id"]),
            competition_id=str(data["competition_id"]),
            hypothesis=str(data.get("hypothesis") or ""),
            status=ExperimentStatus(
                str(data.get("status") or ExperimentStatus.PROPOSED.value)
            ),
            parent_experiment_id=_optional(data.get("parent_experiment_id")),
            work_session_id=_optional(data.get("work_session_id")),
            job_id=_optional(data.get("job_id")),
            cv_spec_id=_optional(data.get("cv_spec_id")),
            dataset_fingerprint=_optional(data.get("dataset_fingerprint")),
            code_snapshot=_optional(data.get("code_snapshot")),
            config=_dict(data.get("config")),
            config_diff=_dict(data.get("config_diff")),
            fold_scores=tuple(float(item) for item in data.get("fold_scores", [])),
            cv_mean=_optional_float(data.get("cv_mean")),
            cv_std=_optional_float(data.get("cv_std")),
            runtime_seconds=_optional_float(data.get("runtime_seconds")),
            backend=_optional(data.get("backend")),
            artifact_manifest_path=_optional(data.get("artifact_manifest_path")),
            review=_dict(data.get("review")),
            failure_reason=_optional(data.get("failure_reason")),
            metadata=_dict(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_timestamp()),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class SubmissionCandidate:
    candidate_id: str
    competition_id: str
    experiment_id: str
    file_path: str
    file_sha256: str
    status: SubmissionStatus = SubmissionStatus.CANDIDATE
    message: str = ""
    cv_score: float | None = None
    previous_best_cv: float | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    risks: tuple[str, ...] = ()
    approval_id: str | None = None
    submitted_at: str | None = None
    public_score: float | None = None
    private_score: float | None = None
    kaggle_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    @classmethod
    def new(
        cls,
        *,
        competition_id: str,
        experiment_id: str,
        file_path: str,
        file_sha256: str,
        message: str,
        cv_score: float | None = None,
        previous_best_cv: float | None = None,
        validation: Mapping[str, Any] | None = None,
        risks: tuple[str, ...] | list[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "SubmissionCandidate":
        return cls(
            candidate_id=f"SUBC-{uuid.uuid4().hex[:10].upper()}",
            competition_id=competition_id,
            experiment_id=experiment_id,
            file_path=file_path,
            file_sha256=file_sha256,
            message=message.strip(),
            cv_score=cv_score,
            previous_best_cv=previous_best_cv,
            validation=dict(validation or {}),
            risks=tuple(str(item) for item in risks),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["risks"] = list(self.risks)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubmissionCandidate":
        return cls(
            candidate_id=str(data["candidate_id"]),
            competition_id=str(data["competition_id"]),
            experiment_id=str(data["experiment_id"]),
            file_path=str(data.get("file_path") or ""),
            file_sha256=str(data.get("file_sha256") or ""),
            status=SubmissionStatus(
                str(data.get("status") or SubmissionStatus.CANDIDATE.value)
            ),
            message=str(data.get("message") or ""),
            cv_score=_optional_float(data.get("cv_score")),
            previous_best_cv=_optional_float(data.get("previous_best_cv")),
            validation=_dict(data.get("validation")),
            risks=tuple(str(item) for item in data.get("risks", [])),
            approval_id=_optional(data.get("approval_id")),
            submitted_at=_optional(data.get("submitted_at")),
            public_score=_optional_float(data.get("public_score")),
            private_score=_optional_float(data.get("private_score")),
            kaggle_ref=_optional(data.get("kaggle_ref")),
            metadata=_dict(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_timestamp()),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
