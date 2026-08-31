from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from harness.compute.base import BackendStatus, ComputeHandle
from harness.compute.bundle import build_source_bundle
from harness.compute.remote import RemoteWorkerBackend
from harness.platform.models import JobRecord


class PortableRemoteWorkerBackend(RemoteWorkerBackend):
    """Remote worker client with safe source bundle transfer and URL joining."""

    def __init__(self, *args, max_bundle_bytes: int = 64 * 1024 * 1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_bundle_bytes = max(1024, max_bundle_bytes)

    def prepare(self, job: JobRecord, workspace: Path) -> dict[str, Any]:
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        spec_path = workspace / "job_spec.json"
        spec_path.write_text(
            json.dumps(job.spec.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        bundle = build_source_bundle(
            workspace,
            max_bytes=self.max_bundle_bytes,
        )
        manifest = {
            "workspace": str(workspace),
            "job_spec": str(spec_path),
            "remote_worker": self.descriptor.base_url,
            "source_bundle": {
                key: value for key, value in bundle.items() if key != "data"
            },
        }
        (workspace / "source_bundle_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {**manifest, "_bundle": bundle}

    def submit(self, job: JobRecord, workspace: Path) -> ComputeHandle:
        prepared = self.prepare(job, workspace)
        bundle = prepared.pop("_bundle")
        response = self._request(
            "POST",
            "/v1/jobs",
            {
                "job": job.to_dict(),
                "workspace_manifest": prepared,
                "source_bundle": bundle,
            },
        )
        backend_job_id = str(
            response.get("backend_job_id") or response.get("job_id") or job.spec.job_id
        )
        try:
            status = BackendStatus(str(response.get("status") or "submitted"))
        except ValueError:
            status = BackendStatus.SUBMITTED
        return ComputeHandle(
            backend=self.name,
            backend_job_id=backend_job_id,
            status=status,
            stage=str(response.get("stage") or "submitted"),
            progress=_progress(response.get("progress"), 0.05),
            message=str(response.get("message") or "Remote job submitted"),
            error=(str(response["error"]) if response.get("error") else None),
            metadata={"remote": response, **prepared},
        )

    def _download(self, url: str, target: Path) -> None:
        absolute = urllib.parse.urljoin(
            self.descriptor.base_url.rstrip("/") + "/",
            url,
        )
        request = urllib.request.Request(
            absolute,
            headers={"Authorization": f"Bearer {self.descriptor.token}"},
        )
        with urllib.request.urlopen(
            request,
            timeout=max(60, self.timeout_seconds),
        ) as response:
            target.write_bytes(response.read())


def _progress(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default
