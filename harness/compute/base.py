from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from harness.platform.models import Domain, JobRecord, JobSpec, JobStatus


class BackendStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    PREPARING = "preparing"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COLLECTING = "collecting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BackendCapabilities:
    accelerators: tuple[str, ...] = ("cpu",)
    max_vram_gb: float | None = None
    max_gpu_count: int | None = None
    max_cpu_cores: int | None = None
    max_ram_gb: float | None = None
    max_runtime_minutes: int | None = None
    network_available: bool = True
    detailed_progress: bool = False
    supports_cancel: bool = True
    supports_kaggle_data: bool = False
    domains: tuple[Domain, ...] = (Domain.RESEARCH, Domain.KAGGLE)
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["domains"] = [item.value for item in self.domains]
        return payload

    def satisfies(self, spec: JobSpec) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        resources = spec.resources
        if spec.domain not in self.domains:
            reasons.append(f"domain {spec.domain.value} is unsupported")
        if resources.accelerator not in self.accelerators:
            reasons.append(f"accelerator {resources.accelerator} is unsupported")
        if (
            self.max_vram_gb is not None
            and resources.min_vram_gb > self.max_vram_gb
        ):
            reasons.append(
                f"requires {resources.min_vram_gb}GB VRAM; backend has {self.max_vram_gb}GB"
            )
        if (
            self.max_gpu_count is not None
            and resources.preferred_gpu_count > self.max_gpu_count
        ):
            reasons.append(
                f"prefers {resources.preferred_gpu_count} GPUs; backend has {self.max_gpu_count}"
            )
        if self.max_cpu_cores is not None and resources.cpu_cores > self.max_cpu_cores:
            reasons.append(
                f"requires {resources.cpu_cores} CPU cores; backend has {self.max_cpu_cores}"
            )
        if self.max_ram_gb is not None and resources.ram_gb > self.max_ram_gb:
            reasons.append(
                f"requires {resources.ram_gb}GB RAM; backend has {self.max_ram_gb}GB"
            )
        if (
            self.max_runtime_minutes is not None
            and resources.max_runtime_minutes > self.max_runtime_minutes
        ):
            reasons.append(
                f"requires {resources.max_runtime_minutes} minutes; backend limit is "
                f"{self.max_runtime_minutes}"
            )
        if resources.network_required and not self.network_available:
            reasons.append("job requires network access")
        missing_tags = set(resources.capabilities) - set(self.tags)
        if missing_tags:
            reasons.append("missing capabilities: " + ", ".join(sorted(missing_tags)))
        return not reasons, reasons


@dataclass(frozen=True)
class ComputeHandle:
    backend: str
    backend_job_id: str
    status: BackendStatus
    stage: str
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status in {
            BackendStatus.COMPLETED,
            BackendStatus.FAILED,
            BackendStatus.CANCELLED,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@runtime_checkable
class ComputeBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def available(self) -> tuple[bool, str]:
        """Return whether this backend is configured and reachable enough to use."""

    def prepare(self, job: JobRecord, workspace: Path) -> dict[str, Any]:
        """Validate and package a job without starting paid/remote work."""

    def submit(self, job: JobRecord, workspace: Path) -> ComputeHandle:
        """Start one durable compute job."""

    def poll(self, job: JobRecord) -> ComputeHandle:
        """Return the latest backend status."""

    def cancel(self, job: JobRecord) -> ComputeHandle:
        """Request cancellation. Backends may expose only best-effort cancellation."""

    def collect(self, job: JobRecord, destination: Path) -> ComputeHandle:
        """Collect result files and return a terminal handle."""


@dataclass(frozen=True)
class BackendDecision:
    selected: str | None
    ordered_candidates: tuple[str, ...]
    rejected: dict[str, tuple[str, ...]]
    reason: str
    requires_approval: bool = False


class ComputeBroker:
    """Deterministically route jobs across Kaggle, remote GPU, VM, and CPU.

    The broker never starts work. It only selects a backend and reports whether
    the chosen backend requires an explicit cost/external-action approval.
    """

    def __init__(
        self,
        backends: Iterable[ComputeBackend] = (),
        *,
        default_order: tuple[str, ...] = (
            "kaggle_notebook",
            "remote_gpu",
            "gpu_vm",
            "local_process",
        ),
        paid_backends: tuple[str, ...] = ("gpu_vm",),
        allow_kaggle_for_research: bool = False,
    ) -> None:
        self.backends: dict[str, ComputeBackend] = {}
        self.default_order = default_order
        self.paid_backends = frozenset(paid_backends)
        self.allow_kaggle_for_research = allow_kaggle_for_research
        for backend in backends:
            self.register(backend)

    def register(self, backend: ComputeBackend) -> None:
        if backend.name in self.backends:
            raise ValueError(f"duplicate compute backend: {backend.name}")
        self.backends[backend.name] = backend

    def decide(self, spec: JobSpec) -> BackendDecision:
        requested = list(spec.backend_preferences)
        ordered: list[str] = []

        def add(name: str) -> None:
            if name in self.backends and name not in ordered:
                ordered.append(name)

        for name in requested:
            add(name)
        if spec.domain == Domain.KAGGLE:
            add("kaggle_notebook")
            add("remote_gpu")
            add("gpu_vm")
            add("local_process")
        else:
            add("remote_gpu")
            add("gpu_vm")
            add("local_process")
            if self.allow_kaggle_for_research:
                add("kaggle_notebook")
        for name in self.default_order:
            add(name)
        for name in sorted(self.backends):
            add(name)

        rejected: dict[str, tuple[str, ...]] = {}
        for name in ordered:
            backend = self.backends[name]
            available, detail = backend.available()
            if not available:
                rejected[name] = (f"unavailable: {detail}",)
                continue
            if (
                spec.domain == Domain.RESEARCH
                and name == "kaggle_notebook"
                and not self.allow_kaggle_for_research
            ):
                rejected[name] = ("Kaggle compute is disabled for non-Kaggle research",)
                continue
            supported, reasons = backend.capabilities.satisfies(spec)
            if not supported:
                rejected[name] = tuple(reasons)
                continue
            return BackendDecision(
                selected=name,
                ordered_candidates=tuple(ordered),
                rejected=rejected,
                reason=f"{name} is the first available backend satisfying the JobSpec",
                requires_approval=name in self.paid_backends,
            )
        return BackendDecision(
            selected=None,
            ordered_candidates=tuple(ordered),
            rejected=rejected,
            reason="No configured backend satisfies the JobSpec",
        )

    def backend(self, name: str) -> ComputeBackend:
        if name not in self.backends:
            raise KeyError(f"unknown compute backend: {name}")
        return self.backends[name]


def backend_status_to_job_status(status: BackendStatus) -> JobStatus:
    mapping = {
        BackendStatus.CREATED: JobStatus.CREATED,
        BackendStatus.QUEUED: JobStatus.QUEUED,
        BackendStatus.PREPARING: JobStatus.PREPARING,
        BackendStatus.SUBMITTED: JobStatus.SUBMITTED,
        BackendStatus.RUNNING: JobStatus.RUNNING,
        BackendStatus.COLLECTING: JobStatus.COLLECTING,
        BackendStatus.COMPLETED: JobStatus.COMPLETED,
        BackendStatus.FAILED: JobStatus.FAILED,
        BackendStatus.CANCEL_REQUESTED: JobStatus.CANCEL_REQUESTED,
        BackendStatus.CANCELLED: JobStatus.CANCELLED,
        BackendStatus.UNKNOWN: JobStatus.BLOCKED,
    }
    return mapping[status]
