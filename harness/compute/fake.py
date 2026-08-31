from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from harness.compute.base import (
    BackendCapabilities,
    BackendStatus,
    ComputeHandle,
)
from harness.platform.models import Domain, JobRecord


class FakeComputeBackend:
    """Deterministic backend used by CI and offline architecture tests."""

    name = "fake"
    capabilities = BackendCapabilities(
        accelerators=("cpu", "gpu"),
        max_vram_gb=96,
        max_gpu_count=8,
        max_cpu_cores=128,
        max_ram_gb=1024,
        max_runtime_minutes=10080,
        detailed_progress=True,
        supports_cancel=True,
        supports_kaggle_data=True,
        domains=(Domain.RESEARCH, Domain.KAGGLE),
        tags=("notebook", "training", "inference", "testing"),
    )

    def __init__(self, *, available: bool = True, fail: bool = False) -> None:
        self._available = available
        self.fail = fail
        self._lock = threading.RLock()
        self._states: dict[str, ComputeHandle] = {}
        self._poll_count: dict[str, int] = {}

    def available(self) -> tuple[bool, str]:
        return self._available, "deterministic fake backend"

    def prepare(self, job: JobRecord, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        manifest = {
            "job": job.to_dict(),
            "backend": self.name,
            "workspace": str(workspace),
        }
        (workspace / "prepared_job.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    def submit(self, job: JobRecord, workspace: Path) -> ComputeHandle:
        self.prepare(job, workspace)
        handle = ComputeHandle(
            backend=self.name,
            backend_job_id=f"fake-{job.spec.job_id}",
            status=BackendStatus.QUEUED,
            stage="queued",
            progress=0.0,
            message="Fake job queued",
            metadata={"workspace": str(workspace)},
        )
        with self._lock:
            self._states[job.spec.job_id] = handle
            self._poll_count[job.spec.job_id] = 0
        return handle

    def poll(self, job: JobRecord) -> ComputeHandle:
        with self._lock:
            current = self._states[job.spec.job_id]
            count = self._poll_count[job.spec.job_id] + 1
            self._poll_count[job.spec.job_id] = count
            if current.status in {
                BackendStatus.CANCELLED,
                BackendStatus.FAILED,
                BackendStatus.COMPLETED,
            }:
                return current
            if count == 1:
                current = replace(
                    current,
                    status=BackendStatus.RUNNING,
                    stage="execute",
                    progress=0.45,
                    message="Fake job running",
                )
            elif self.fail:
                current = replace(
                    current,
                    status=BackendStatus.FAILED,
                    stage="failed",
                    progress=1.0,
                    message="Fake job failed",
                    error="configured fake failure",
                )
            else:
                current = replace(
                    current,
                    status=BackendStatus.COMPLETED,
                    stage="completed",
                    progress=1.0,
                    message="Fake job completed",
                    result={
                        "metric": "fake_score",
                        "value": 0.75,
                        "job_id": job.spec.job_id,
                    },
                )
            self._states[job.spec.job_id] = current
            return current

    def cancel(self, job: JobRecord) -> ComputeHandle:
        with self._lock:
            current = self._states.get(job.spec.job_id)
            if current is None:
                current = ComputeHandle(
                    backend=self.name,
                    backend_job_id=f"fake-{job.spec.job_id}",
                    status=BackendStatus.CANCELLED,
                    stage="cancelled",
                    progress=0.0,
                    message="Fake job cancelled before submission",
                )
            else:
                current = replace(
                    current,
                    status=BackendStatus.CANCELLED,
                    stage="cancelled",
                    message="Fake job cancelled",
                )
            self._states[job.spec.job_id] = current
            return current

    def collect(self, job: JobRecord, destination: Path) -> ComputeHandle:
        with self._lock:
            current = self._states[job.spec.job_id]
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "result.json").write_text(
            json.dumps(
                {
                    "job_id": job.spec.job_id,
                    "status": current.status.value,
                    "result": current.result,
                    "error": current.error,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return replace(
            current,
            stage="collected" if current.status == BackendStatus.COMPLETED else current.stage,
            metadata={**current.metadata, "destination": str(destination)},
        )
