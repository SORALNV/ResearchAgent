from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from harness.control_plane import Domain, Job, JobSpec, ResourceRequirements
from harness.state import utc_timestamp


class BackendState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


_TERMINAL_BACKEND_STATES = {
    BackendState.SUCCEEDED,
    BackendState.FAILED,
    BackendState.CANCELLED,
}


@dataclass(frozen=True)
class BackendCapabilities:
    accelerators: tuple[str, ...] = ("cpu",)
    domains: tuple[Domain, ...] = (Domain.RESEARCH, Domain.KAGGLE)
    gpu_count: int = 0
    gpu_memory_mb: int | None = None
    cpu_cores: float | None = None
    memory_mb: int | None = None
    ephemeral_storage_mb: int | None = None
    network_available: bool = True
    labels: tuple[str, ...] = ()
    supports_cancel: bool = True
    recoverable: bool = True

    def __post_init__(self) -> None:
        if self.gpu_count < 0:
            raise ValueError("gpu_count must be non-negative")
        for name in ("gpu_memory_mb", "memory_mb", "ephemeral_storage_mb"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cpu_cores is not None and self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["domains"] = [item.value for item in self.domains]
        payload["accelerators"] = list(self.accelerators)
        payload["labels"] = list(self.labels)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "BackendCapabilities":
        value = dict(data or {})
        domains: list[Domain] = []
        for item in value.get("domains", [Domain.RESEARCH.value, Domain.KAGGLE.value]):
            try:
                domains.append(Domain(str(item)))
            except ValueError:
                continue
        return cls(
            accelerators=tuple(
                str(item).strip().lower()
                for item in value.get("accelerators", ["cpu"])
                if str(item).strip()
            )
            or ("cpu",),
            domains=tuple(domains) or (Domain.RESEARCH, Domain.KAGGLE),
            gpu_count=max(0, int(value.get("gpu_count") or 0)),
            gpu_memory_mb=(
                int(value["gpu_memory_mb"])
                if value.get("gpu_memory_mb") is not None
                else None
            ),
            cpu_cores=(
                float(value["cpu_cores"])
                if value.get("cpu_cores") is not None
                else None
            ),
            memory_mb=(
                int(value["memory_mb"])
                if value.get("memory_mb") is not None
                else None
            ),
            ephemeral_storage_mb=(
                int(value["ephemeral_storage_mb"])
                if value.get("ephemeral_storage_mb") is not None
                else None
            ),
            network_available=bool(value.get("network_available", True)),
            labels=tuple(str(item) for item in value.get("labels", [])),
            supports_cancel=bool(value.get("supports_cancel", True)),
            recoverable=bool(value.get("recoverable", True)),
        )

    def satisfies(self, spec: JobSpec) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        request = spec.resources
        if spec.domain not in self.domains:
            reasons.append(f"domain {spec.domain.value} is unsupported")

        requested_accelerator = _requested_accelerator(request)
        supported_accelerators = {
            _normalize_accelerator(item) for item in self.accelerators
        }
        if requested_accelerator not in supported_accelerators:
            reasons.append(
                f"accelerator {requested_accelerator} is unsupported"
            )
        if request.gpu_count > self.gpu_count:
            reasons.append(
                f"requires {request.gpu_count} GPU(s); backend has {self.gpu_count}"
            )
        _require_known_capacity(
            reasons,
            requested=request.gpu_memory_mb,
            available=self.gpu_memory_mb,
            label="GPU memory",
            unit="MB",
        )
        _require_known_capacity(
            reasons,
            requested=request.cpu_cores,
            available=self.cpu_cores,
            label="CPU cores",
            unit="",
        )
        _require_known_capacity(
            reasons,
            requested=request.memory_mb,
            available=self.memory_mb,
            label="RAM",
            unit="MB",
        )
        _require_known_capacity(
            reasons,
            requested=request.ephemeral_storage_mb,
            available=self.ephemeral_storage_mb,
            label="ephemeral storage",
            unit="MB",
        )
        if request.network_required and not self.network_available:
            reasons.append("job requires network access")
        missing_labels = set(request.labels) - set(self.labels)
        if missing_labels:
            reasons.append(
                "missing backend labels: " + ", ".join(sorted(missing_labels))
            )
        return not reasons, tuple(reasons)


@dataclass(frozen=True)
class BackendSelection:
    selected: str | None
    ordered_candidates: tuple[str, ...]
    rejected: dict[str, tuple[str, ...]]
    reason: str
    approval_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "ordered_candidates": list(self.ordered_candidates),
            "rejected": {
                key: list(value) for key, value in self.rejected.items()
            },
            "reason": self.reason,
            "approval_required": self.approval_required,
        }


@dataclass(frozen=True)
class BackendHandle:
    backend: str
    backend_job_id: str
    state: BackendState
    stage: str
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        object.__setattr__(self, "progress", min(1.0, max(0.0, float(self.progress))))

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_BACKEND_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_job_id": self.backend_job_id,
            "state": self.state.value,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "result": json_copy(self.result),
            "error": self.error,
            "metadata": json_copy(self.metadata),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackendHandle":
        return cls(
            backend=str(data["backend"]),
            backend_job_id=str(data["backend_job_id"]),
            state=BackendState(str(data.get("state") or BackendState.UNKNOWN.value)),
            stage=str(data.get("stage") or "unknown"),
            progress=float(data.get("progress") or 0.0),
            message=str(data.get("message") or ""),
            result=json_dict(data.get("result")),
            error=(str(data["error"]) if data.get("error") else None),
            metadata=json_dict(data.get("metadata")),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class ComputeRuntimeRecord:
    job_id: str
    backend: str | None = None
    handle: BackendHandle | None = None
    workspace: str = ""
    artifacts_dir: str = ""
    approval_required: bool = False
    approved: bool = False
    unknown_polls: int = 0
    collection_complete: bool = False
    result_ref: str | None = None
    selected_at: str | None = None
    updated_at: str = field(default_factory=utc_timestamp)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "backend": self.backend,
            "handle": self.handle.to_dict() if self.handle else None,
            "workspace": self.workspace,
            "artifacts_dir": self.artifacts_dir,
            "approval_required": self.approval_required,
            "approved": self.approved,
            "unknown_polls": self.unknown_polls,
            "collection_complete": self.collection_complete,
            "result_ref": self.result_ref,
            "selected_at": self.selected_at,
            "updated_at": self.updated_at,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComputeRuntimeRecord":
        raw_handle = data.get("handle")
        return cls(
            job_id=str(data["job_id"]),
            backend=(str(data["backend"]) if data.get("backend") else None),
            handle=(
                BackendHandle.from_dict(raw_handle)
                if isinstance(raw_handle, Mapping)
                else None
            ),
            workspace=str(data.get("workspace") or ""),
            artifacts_dir=str(data.get("artifacts_dir") or ""),
            approval_required=bool(data.get("approval_required", False)),
            approved=bool(data.get("approved", False)),
            unknown_polls=max(0, int(data.get("unknown_polls") or 0)),
            collection_complete=bool(data.get("collection_complete", False)),
            result_ref=(
                str(data["result_ref"]) if data.get("result_ref") else None
            ),
            selected_at=(
                str(data["selected_at"])
                if data.get("selected_at")
                else None
            ),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
            metadata=json_dict(data.get("metadata")),
        )


@dataclass(frozen=True)
class CollectedResult:
    result: dict[str, Any]
    artifact_paths: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ComputeBackend(Protocol):
    name: str
    capabilities: BackendCapabilities
    approval_required: bool

    def available(self) -> tuple[bool, str]:
        ...

    def prepare(self, job: Job, workspace: Path) -> dict[str, Any]:
        ...

    def submit(self, job: Job, workspace: Path) -> BackendHandle:
        ...

    def poll(self, job: Job, handle: BackendHandle) -> BackendHandle:
        ...

    def cancel(self, job: Job, handle: BackendHandle | None) -> BackendHandle:
        ...

    def collect(
        self,
        job: Job,
        handle: BackendHandle,
        destination: Path,
    ) -> CollectedResult:
        ...


class FeedbackPlanner(Protocol):
    def propose(
        self,
        *,
        job: Job,
        result: Mapping[str, Any],
        result_ref: str,
    ) -> list[dict[str, Any]]:
        ...


def json_copy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    encoded = json.dumps(dict(value), ensure_ascii=False, allow_nan=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("expected a JSON object")
    return decoded


def json_dict(value: Any) -> dict[str, Any]:
    return json_copy(value) if isinstance(value, Mapping) else {}


def canonical_json_hash(value: Mapping[str, Any]) -> str:
    data = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def job_entrypoint(job: Job) -> str:
    return str(job.spec.payload.get("entrypoint") or "").strip()


def job_outputs(job: Job) -> tuple[str, ...]:
    raw = job.spec.payload.get("outputs") or []
    if isinstance(raw, str):
        raw = [raw]
    return tuple(
        item
        for item in (safe_relative_path(str(value)) for value in raw)
        if item
    )


def safe_relative_path(value: str) -> str:
    path = Path(str(value).strip())
    if not str(value).strip() or path.is_absolute() or ".." in path.parts:
        return ""
    return path.as_posix()


def _requested_accelerator(resources: ResourceRequirements) -> str:
    if resources.gpu_count > 0:
        return _normalize_accelerator(resources.accelerator or "gpu")
    return _normalize_accelerator(resources.accelerator or "cpu")


def _normalize_accelerator(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"cuda", "nvidia", "gpu"}:
        return "gpu"
    return normalized or "cpu"


def _require_known_capacity(
    reasons: list[str],
    *,
    requested: int | float | None,
    available: int | float | None,
    label: str,
    unit: str,
) -> None:
    if requested is None:
        return
    suffix = unit if unit else ""
    if available is None:
        reasons.append(
            f"requires {requested}{suffix} {label}; backend capacity is unknown"
        )
        return
    if requested > available:
        reasons.append(
            f"requires {requested}{suffix} {label}; "
            f"backend has {available}{suffix}"
        )
