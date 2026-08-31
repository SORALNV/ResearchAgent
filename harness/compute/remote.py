from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from harness.compute.base import (
    BackendCapabilities,
    BackendStatus,
    ComputeHandle,
)
from harness.platform.models import Domain, JobRecord


@dataclass(frozen=True)
class RemoteWorkerDescriptor:
    name: str
    base_url: str
    token: str
    paid: bool = False
    capabilities: BackendCapabilities = BackendCapabilities(
        accelerators=("cpu", "gpu"),
        detailed_progress=True,
        supports_cancel=True,
        supports_kaggle_data=False,
        domains=(Domain.RESEARCH, Domain.KAGGLE),
        tags=("training", "inference", "remote_worker"),
    )


class RemoteWorkerBackend:
    """Typed HTTP client for an owned GPU PC or a provisioned GPU VM."""

    def __init__(
        self,
        descriptor: RemoteWorkerDescriptor,
        *,
        timeout_seconds: int = 30,
    ) -> None:
        self.descriptor = descriptor
        self.name = descriptor.name
        self.capabilities = descriptor.capabilities
        self.timeout_seconds = max(2, timeout_seconds)

    def available(self) -> tuple[bool, str]:
        if not self.descriptor.base_url or not self.descriptor.token:
            return False, "remote worker URL or token is missing"
        try:
            response = self._request("GET", "/health")
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        healthy = bool(response.get("ok", True))
        return healthy, str(response.get("detail") or self.descriptor.base_url)

    def prepare(self, job: JobRecord, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        spec_path = workspace / "job_spec.json"
        spec_path.write_text(
            json.dumps(job.spec.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "workspace": str(workspace.resolve()),
            "job_spec": str(spec_path),
            "remote_worker": self.descriptor.base_url,
        }

    def submit(self, job: JobRecord, workspace: Path) -> ComputeHandle:
        prepared = self.prepare(job, workspace)
        response = self._request(
            "POST",
            "/v1/jobs",
            {
                "job": job.to_dict(),
                "workspace_manifest": prepared,
            },
        )
        backend_job_id = str(
            response.get("backend_job_id") or response.get("job_id") or job.spec.job_id
        )
        return ComputeHandle(
            backend=self.name,
            backend_job_id=backend_job_id,
            status=_status(response.get("status"), BackendStatus.SUBMITTED),
            stage=str(response.get("stage") or "submitted"),
            progress=_progress(response.get("progress"), 0.05),
            message=str(response.get("message") or "Remote job submitted"),
            metadata={"remote": response, **prepared},
        )

    def poll(self, job: JobRecord) -> ComputeHandle:
        backend_job_id = _backend_job_id(job)
        response = self._request(
            "GET",
            "/v1/jobs/" + urllib.parse.quote(backend_job_id, safe=""),
        )
        return ComputeHandle(
            backend=self.name,
            backend_job_id=backend_job_id,
            status=_status(response.get("status"), BackendStatus.UNKNOWN),
            stage=str(response.get("stage") or "unknown"),
            progress=_progress(response.get("progress"), job.progress),
            message=str(response.get("message") or ""),
            result=(dict(response.get("result")) if isinstance(response.get("result"), Mapping) else {}),
            error=(str(response["error"]) if response.get("error") else None),
            metadata={"remote": response},
        )

    def cancel(self, job: JobRecord) -> ComputeHandle:
        backend_job_id = _backend_job_id(job)
        response = self._request(
            "POST",
            "/v1/jobs/" + urllib.parse.quote(backend_job_id, safe="") + "/cancel",
            {"reason": "ResearchAgent cancellation requested"},
        )
        return ComputeHandle(
            backend=self.name,
            backend_job_id=backend_job_id,
            status=_status(response.get("status"), BackendStatus.CANCEL_REQUESTED),
            stage=str(response.get("stage") or "cancel_requested"),
            progress=_progress(response.get("progress"), job.progress),
            message=str(response.get("message") or "Cancellation requested"),
            metadata={"remote": response},
        )

    def collect(self, job: JobRecord, destination: Path) -> ComputeHandle:
        backend_job_id = _backend_job_id(job)
        destination.mkdir(parents=True, exist_ok=True)
        response = self._request(
            "GET",
            "/v1/jobs/" + urllib.parse.quote(backend_job_id, safe="") + "/artifacts",
        )
        artifacts = response.get("artifacts") if isinstance(response.get("artifacts"), list) else []
        collected: list[dict[str, Any]] = []
        for item in artifacts:
            if not isinstance(item, Mapping):
                continue
            url = str(item.get("url") or "")
            relative = _safe_relative(str(item.get("path") or ""))
            if not url or not relative:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            self._download(url, target)
            collected.append({"path": relative, "destination": str(target)})
        manifest = destination / "remote_artifacts.json"
        manifest.write_text(
            json.dumps(
                {
                    "backend_job_id": backend_job_id,
                    "collected": collected,
                    "response": response,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return ComputeHandle(
            backend=self.name,
            backend_job_id=backend_job_id,
            status=BackendStatus.COMPLETED,
            stage="collected",
            progress=1.0,
            message=f"Collected {len(collected)} remote artifacts",
            result=(dict(response.get("result")) if isinstance(response.get("result"), Mapping) else {}),
            metadata={"destination": str(destination), "artifacts": collected},
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self.descriptor.base_url.rstrip("/") + path
        body = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.descriptor.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                content = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote worker HTTP {exc.code}: {detail[-4000:]}") from exc
        value = json.loads(content or "{}")
        if not isinstance(value, dict):
            raise RuntimeError("Remote worker returned a non-object response")
        return value

    def _download(self, url: str, target: Path) -> None:
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.descriptor.token}"},
        )
        with urllib.request.urlopen(request, timeout=max(60, self.timeout_seconds)) as response:
            with target.open("wb") as handle:
                shutil.copyfileobj(response, handle)


def _backend_job_id(job: JobRecord) -> str:
    if not job.backend_job_id:
        raise ValueError(f"backend_job_id missing for {job.spec.job_id}")
    return job.backend_job_id


def _status(value: Any, default: BackendStatus) -> BackendStatus:
    try:
        return BackendStatus(str(value))
    except ValueError:
        return default


def _progress(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _safe_relative(value: str) -> str:
    path = Path(value.strip())
    if not value.strip() or path.is_absolute() or ".." in path.parts:
        return ""
    return path.as_posix()
