from __future__ import annotations

import threading
from pathlib import Path

from harness.compute.base import ComputeHandle
from harness.compute.local import LocalProcessBackend
from harness.platform.models import JobRecord


class PortableLocalProcessBackend(LocalProcessBackend):
    """LocalProcessBackend with durable workspace-aware artifact collection."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._workspace_lock = threading.RLock()
        self._workspaces: dict[str, Path] = {}

    def prepare(self, job: JobRecord, workspace: Path):
        resolved = workspace.expanduser().resolve()
        with self._workspace_lock:
            self._workspaces[job.spec.job_id] = resolved
        return super().prepare(job, resolved)

    def submit(self, job: JobRecord, workspace: Path) -> ComputeHandle:
        resolved = workspace.expanduser().resolve()
        with self._workspace_lock:
            self._workspaces[job.spec.job_id] = resolved
        return super().submit(job, resolved)

    def collect(self, job: JobRecord, destination: Path) -> ComputeHandle:
        with self._workspace_lock:
            workspace = self._workspaces.get(job.spec.job_id)
        if workspace is None:
            configured = job.spec.metadata.get("workspace")
            if configured:
                workspace = Path(str(configured)).expanduser().resolve()
        if workspace is None:
            raise RuntimeError(
                "local workspace is unavailable; local processes cannot be recovered "
                "across a Core restart"
            )
        destination.mkdir(parents=True, exist_ok=True)
        collected: list[str] = []
        for relative in job.spec.outputs:
            source = (workspace / relative).resolve()
            try:
                source.relative_to(workspace)
            except ValueError:
                continue
            if source.is_symlink() or not source.is_file():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            collected.append(relative)
        result = _read_json(workspace / "result.json")
        return ComputeHandle(
            backend=self.name,
            backend_job_id=job.backend_job_id or "local",
            status=__import__("harness.compute.base", fromlist=["BackendStatus"]).BackendStatus.COMPLETED,
            stage="collected",
            progress=1.0,
            message=f"Collected {len(collected)} local outputs",
            result=result,
            metadata={
                "workspace": str(workspace),
                "collected": collected,
                "destination": str(destination),
            },
        )


def _read_json(path: Path) -> dict:
    import json

    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}
