from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from harness.control_plane import JobSpec


EventSink = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    domains: tuple[str, ...] = ("research", "kaggle")
    accelerators: tuple[str, ...] = ("cpu",)
    max_vram_gb: float = 0.0
    max_ram_gb: float = 0.0
    supports_cancel: bool = True
    supports_live_events: bool = True
    supports_network: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, spec: JobSpec) -> tuple[bool, str]:
        if self.domains and spec.domain not in self.domains:
            return False, f"domain {spec.domain} unsupported"
        accelerator = str(spec.resources.get("accelerator") or "cpu").lower()
        if accelerator not in self.accelerators:
            return False, f"accelerator {accelerator} unsupported"
        min_vram = float(spec.resources.get("min_vram_gb") or 0.0)
        if min_vram > 0 and self.max_vram_gb > 0 and min_vram > self.max_vram_gb:
            return False, f"requires {min_vram}GB VRAM, backend has {self.max_vram_gb}GB"
        min_ram = float(spec.resources.get("ram_gb") or 0.0)
        if min_ram > 0 and self.max_ram_gb > 0 and min_ram > self.max_ram_gb:
            return False, f"requires {min_ram}GB RAM, backend has {self.max_ram_gb}GB"
        network_required = bool(spec.resources.get("network_required", False))
        if network_required and not self.supports_network:
            return False, "network access required"
        return True, "compatible"


@dataclass(frozen=True)
class BackendRunResult:
    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    backend_job_id: str | None = None

    @classmethod
    def completed(
        cls,
        result: dict[str, Any] | None = None,
        *,
        backend_job_id: str | None = None,
    ) -> "BackendRunResult":
        return cls(
            status="completed",
            result=dict(result or {}),
            backend_job_id=backend_job_id,
        )

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        result: dict[str, Any] | None = None,
        backend_job_id: str | None = None,
    ) -> "BackendRunResult":
        return cls(
            status="failed",
            result=dict(result or {}),
            error=error,
            backend_job_id=backend_job_id,
        )

    @classmethod
    def cancelled(
        cls,
        error: str = "cancelled",
        *,
        result: dict[str, Any] | None = None,
        backend_job_id: str | None = None,
    ) -> "BackendRunResult":
        return cls(
            status="cancelled",
            result=dict(result or {}),
            error=error,
            backend_job_id=backend_job_id,
        )


class ComputeBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    def run(
        self,
        spec: JobSpec,
        *,
        emit: EventSink,
        cancel_event: threading.Event,
    ) -> BackendRunResult: ...


@dataclass(frozen=True)
class BackendSelection:
    backend: ComputeBackend
    reason: str


class ComputeBroker:
    """Select a compute backend from explicit preferences and domain defaults."""

    DEFAULTS = {
        "kaggle": ("kaggle_notebook", "remote_gpu", "gpu_vm", "local_cpu", "fake"),
        "research": ("remote_gpu", "gpu_vm", "local_cpu", "kaggle_notebook", "fake"),
    }

    def __init__(self, backends: list[ComputeBackend] | tuple[ComputeBackend, ...]) -> None:
        self._backends = {backend.capabilities.name: backend for backend in backends}
        if len(self._backends) != len(backends):
            raise ValueError("duplicate compute backend name")

    @property
    def backend_names(self) -> tuple[str, ...]:
        return tuple(self._backends)

    def get(self, name: str) -> ComputeBackend | None:
        return self._backends.get(name)

    def select(self, spec: JobSpec) -> BackendSelection:
        order = spec.backend_preferences or self.DEFAULTS.get(
            spec.domain,
            tuple(self._backends),
        )
        failures: list[str] = []
        for name in order:
            backend = self._backends.get(name)
            if backend is None:
                failures.append(f"{name}: not configured")
                continue
            compatible, reason = backend.capabilities.matches(spec)
            if compatible:
                return BackendSelection(backend=backend, reason=f"selected {name}: {reason}")
            failures.append(f"{name}: {reason}")
        raise RuntimeError(
            "no compatible compute backend for "
            f"{spec.job_id}; " + "; ".join(failures)
        )

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": backend.capabilities.name,
                "domains": list(backend.capabilities.domains),
                "accelerators": list(backend.capabilities.accelerators),
                "max_vram_gb": backend.capabilities.max_vram_gb,
                "max_ram_gb": backend.capabilities.max_ram_gb,
                "supports_cancel": backend.capabilities.supports_cancel,
                "supports_live_events": backend.capabilities.supports_live_events,
                "supports_network": backend.capabilities.supports_network,
                "metadata": dict(backend.capabilities.metadata),
            }
            for backend in self._backends.values()
        ]


class FakeComputeBackend:
    """Deterministic backend used by CI and end-to-end control-plane tests."""

    def __init__(
        self,
        name: str = "fake",
        *,
        accelerators: tuple[str, ...] = ("cpu", "gpu"),
        domains: tuple[str, ...] = ("research", "kaggle"),
    ) -> None:
        self._capabilities = BackendCapabilities(
            name=name,
            domains=domains,
            accelerators=accelerators,
            max_vram_gb=96.0,
            max_ram_gb=256.0,
            supports_cancel=True,
            supports_live_events=True,
            supports_network=True,
            metadata={"fake": True},
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def run(
        self,
        spec: JobSpec,
        *,
        emit: EventSink,
        cancel_event: threading.Event,
    ) -> BackendRunResult:
        steps = list(spec.payload.get("fake_steps") or [
            {"stage": "preparing", "progress": 0.1},
            {"stage": "running", "progress": 0.5},
            {"stage": "collecting", "progress": 0.9},
        ])
        delay = max(0.0, float(spec.payload.get("fake_delay_seconds") or 0.0))
        backend_job_id = str(
            spec.payload.get("backend_job_id") or f"FAKE-{spec.job_id}"
        )
        emit("backend_started", {"backend_job_id": backend_job_id})
        for index, raw in enumerate(steps, 1):
            if cancel_event.is_set():
                emit("backend_cancelled", {"step": index})
                return BackendRunResult.cancelled(
                    backend_job_id=backend_job_id,
                )
            step = dict(raw) if isinstance(raw, dict) else {"message": str(raw)}
            step.setdefault("step", index)
            step.setdefault("total_steps", len(steps))
            emit("backend_progress", step)
            if delay:
                cancel_event.wait(delay)
        if cancel_event.is_set():
            return BackendRunResult.cancelled(backend_job_id=backend_job_id)
        if spec.payload.get("fake_fail"):
            error = str(spec.payload.get("fake_error") or "fake backend failure")
            emit("backend_failed", {"error": error})
            return BackendRunResult.failed(error, backend_job_id=backend_job_id)
        result = dict(spec.payload.get("fake_result") or {})
        result.setdefault("job_id", spec.job_id)
        result.setdefault("backend", self.capabilities.name)
        emit("backend_completed", result)
        return BackendRunResult.completed(
            result,
            backend_job_id=backend_job_id,
        )
