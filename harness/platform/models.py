from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from harness.state import utc_timestamp


class Domain(StrEnum):
    RESEARCH = "research"
    KAGGLE = "kaggle"


class ProjectStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class WorkSessionStatus(StrEnum):
    PLANNING = "planning"
    WAITING_INPUT = "waiting_input"
    QUEUED = "queued"
    RUNNING = "running"
    REVIEW = "review"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    PREPARING = "preparing"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COLLECTING = "collecting"
    REVIEW = "review"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class EventKind(StrEnum):
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    STATUS = "status"
    MILESTONE = "milestone"
    PROGRESS = "progress"
    LOG = "log"
    ARTIFACT = "artifact"
    APPROVAL = "approval"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    STEERING = "steering"


class SteeringKind(StrEnum):
    QUESTION = "question"
    CONSTRAINT = "constraint"
    REDIRECT = "redirect"
    NEW_HYPOTHESIS = "new_hypothesis"
    CANCEL = "cancel"
    PAUSE = "pause"
    RESUME = "resume"


@dataclass(frozen=True)
class Project:
    project_id: str
    domain: Domain
    title: str
    status: ProjectStatus = ProjectStatus.ACTIVE
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    @classmethod
    def new(
        cls,
        *,
        domain: Domain | str,
        title: str,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "Project":
        normalized = Domain(str(domain))
        prefix = "KG" if normalized == Domain.KAGGLE else "RA"
        return cls(
            project_id=_new_id(prefix, 10),
            domain=normalized,
            title=title.strip(),
            description=description.strip(),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["domain"] = self.domain.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Project":
        return cls(
            project_id=str(data["project_id"]),
            domain=Domain(str(data["domain"])),
            title=str(data.get("title") or ""),
            status=ProjectStatus(str(data.get("status") or ProjectStatus.ACTIVE.value)),
            description=str(data.get("description") or ""),
            metadata=_dict(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_timestamp()),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class WorkSession:
    session_id: str
    project_id: str
    title: str
    objective: str
    status: WorkSessionStatus = WorkSessionStatus.PLANNING
    current_stage: str = "planning"
    parent_session_id: str | None = None
    discord_guild_id: str | None = None
    discord_parent_channel_id: str | None = None
    discord_thread_id: str | None = None
    discord_live_message_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    @classmethod
    def new(
        cls,
        *,
        project_id: str,
        title: str,
        objective: str,
        parent_session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "WorkSession":
        return cls(
            session_id=_new_id("WS", 12),
            project_id=project_id,
            title=title.strip(),
            objective=objective.strip(),
            parent_session_id=parent_session_id,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkSession":
        return cls(
            session_id=str(data["session_id"]),
            project_id=str(data["project_id"]),
            title=str(data.get("title") or ""),
            objective=str(data.get("objective") or ""),
            status=WorkSessionStatus(
                str(data.get("status") or WorkSessionStatus.PLANNING.value)
            ),
            current_stage=str(data.get("current_stage") or "planning"),
            parent_session_id=_optional_str(data.get("parent_session_id")),
            discord_guild_id=_optional_str(data.get("discord_guild_id")),
            discord_parent_channel_id=_optional_str(
                data.get("discord_parent_channel_id")
            ),
            discord_thread_id=_optional_str(data.get("discord_thread_id")),
            discord_live_message_id=_optional_str(
                data.get("discord_live_message_id")
            ),
            metadata=_dict(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_timestamp()),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class ResourceRequest:
    accelerator: str = "cpu"
    min_vram_gb: float = 0.0
    preferred_gpu_count: int = 0
    cpu_cores: int = 1
    ram_gb: float = 2.0
    max_runtime_minutes: int = 60
    network_required: bool = False
    capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = list(self.capabilities)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ResourceRequest":
        value = dict(data or {})
        return cls(
            accelerator=str(value.get("accelerator") or "cpu"),
            min_vram_gb=float(value.get("min_vram_gb") or 0.0),
            preferred_gpu_count=max(0, int(value.get("preferred_gpu_count") or 0)),
            cpu_cores=max(1, int(value.get("cpu_cores") or 1)),
            ram_gb=max(0.1, float(value.get("ram_gb") or 2.0)),
            max_runtime_minutes=max(1, int(value.get("max_runtime_minutes") or 60)),
            network_required=bool(value.get("network_required", False)),
            capabilities=tuple(str(item) for item in value.get("capabilities", [])),
        )


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    work_session_id: str
    domain: Domain
    task_type: str
    entrypoint: str = ""
    parent_job_id: str | None = None
    backend_preferences: tuple[str, ...] = ()
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)

    @classmethod
    def new(
        cls,
        *,
        work_session_id: str,
        domain: Domain | str,
        task_type: str,
        entrypoint: str = "",
        parent_job_id: str | None = None,
        backend_preferences: tuple[str, ...] | list[str] = (),
        resources: ResourceRequest | None = None,
        inputs: Mapping[str, Any] | None = None,
        outputs: tuple[str, ...] | list[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "JobSpec":
        normalized = Domain(str(domain))
        prefix = "JOB-KG" if normalized == Domain.KAGGLE else "JOB-RA"
        return cls(
            job_id=_new_id(prefix, 10),
            work_session_id=work_session_id,
            domain=normalized,
            task_type=task_type.strip(),
            entrypoint=entrypoint.strip(),
            parent_job_id=parent_job_id,
            backend_preferences=tuple(str(item) for item in backend_preferences),
            resources=resources or ResourceRequest(),
            inputs=dict(inputs or {}),
            outputs=tuple(str(item) for item in outputs),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "work_session_id": self.work_session_id,
            "domain": self.domain.value,
            "task_type": self.task_type,
            "entrypoint": self.entrypoint,
            "parent_job_id": self.parent_job_id,
            "backend_preferences": list(self.backend_preferences),
            "resources": self.resources.to_dict(),
            "inputs": self.inputs,
            "outputs": list(self.outputs),
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobSpec":
        return cls(
            job_id=str(data["job_id"]),
            work_session_id=str(data["work_session_id"]),
            domain=Domain(str(data["domain"])),
            task_type=str(data.get("task_type") or ""),
            entrypoint=str(data.get("entrypoint") or ""),
            parent_job_id=_optional_str(data.get("parent_job_id")),
            backend_preferences=tuple(
                str(item) for item in data.get("backend_preferences", [])
            ),
            resources=ResourceRequest.from_dict(
                data.get("resources") if isinstance(data.get("resources"), Mapping) else None
            ),
            inputs=_dict(data.get("inputs")),
            outputs=tuple(str(item) for item in data.get("outputs", [])),
            metadata=_dict(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class JobRecord:
    spec: JobSpec
    status: JobStatus = JobStatus.CREATED
    backend: str | None = None
    backend_job_id: str | None = None
    current_stage: str = "created"
    progress: float = 0.0
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    updated_at: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "backend": self.backend,
            "backend_job_id": self.backend_job_id,
            "current_stage": self.current_stage,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobRecord":
        raw_spec = data.get("spec")
        if not isinstance(raw_spec, Mapping):
            raise ValueError("JobRecord.spec must be an object")
        return cls(
            spec=JobSpec.from_dict(raw_spec),
            status=JobStatus(str(data.get("status") or JobStatus.CREATED.value)),
            backend=_optional_str(data.get("backend")),
            backend_job_id=_optional_str(data.get("backend_job_id")),
            current_stage=str(data.get("current_stage") or "created"),
            progress=min(1.0, max(0.0, float(data.get("progress") or 0.0))),
            result=_dict(data.get("result")),
            error=_optional_str(data.get("error")),
            started_at=_optional_str(data.get("started_at")),
            finished_at=_optional_str(data.get("finished_at")),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class JobEvent:
    event_id: str
    job_id: str | None
    work_session_id: str
    kind: EventKind
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    created_at: str = field(default_factory=utc_timestamp)

    @classmethod
    def new(
        cls,
        *,
        work_session_id: str,
        kind: EventKind | str,
        message: str,
        job_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> "JobEvent":
        return cls(
            event_id=_new_id("EV", 16),
            job_id=job_id,
            work_session_id=work_session_id,
            kind=EventKind(str(kind)),
            message=message,
            payload=dict(payload or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobEvent":
        return cls(
            event_id=str(data["event_id"]),
            job_id=_optional_str(data.get("job_id")),
            work_session_id=str(data["work_session_id"]),
            kind=EventKind(str(data["kind"])),
            message=str(data.get("message") or ""),
            payload=_dict(data.get("payload")),
            sequence=int(data.get("sequence") or 0),
            created_at=str(data.get("created_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class SteeringEvent:
    steering_id: str
    work_session_id: str
    kind: SteeringKind
    instruction: str
    apply_after: str = "next_checkpoint"
    job_id: str | None = None
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    applied_at: str | None = None

    @classmethod
    def new(
        cls,
        *,
        work_session_id: str,
        kind: SteeringKind | str,
        instruction: str,
        apply_after: str = "next_checkpoint",
        job_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SteeringEvent":
        return cls(
            steering_id=_new_id("ST", 14),
            work_session_id=work_session_id,
            kind=SteeringKind(str(kind)),
            instruction=instruction.strip(),
            apply_after=apply_after,
            job_id=job_id,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SteeringEvent":
        return cls(
            steering_id=str(data["steering_id"]),
            work_session_id=str(data["work_session_id"]),
            kind=SteeringKind(str(data["kind"])),
            instruction=str(data.get("instruction") or ""),
            apply_after=str(data.get("apply_after") or "next_checkpoint"),
            job_id=_optional_str(data.get("job_id")),
            status=str(data.get("status") or "pending"),
            metadata=_dict(data.get("metadata")),
            created_at=str(data.get("created_at") or utc_timestamp()),
            applied_at=_optional_str(data.get("applied_at")),
        )


def encode_json(value: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def decode_json(value: str | bytes | None, fallback: Any = None) -> Any:
    if not value:
        return {} if fallback is None else fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {} if fallback is None else fallback


def _new_id(prefix: str, length: int) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:length].upper()}"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
