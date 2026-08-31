from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from harness.compute.base import BackendCapabilities, BackendStatus
from harness.compute.bundle import extract_source_bundle
from harness.compute.portable_local import PortableLocalProcessBackend
from harness.platform.models import Domain, JobRecord, JobSpec, JobStatus
from harness.state import utc_timestamp


@dataclass
class WorkerJob:
    record: JobRecord
    workspace: Path
    artifacts: Path


class PortableWorkerJobManager:
    """Run bundled jobs inside a dedicated Worker container.

    GPU support is capability advertisement: the actual accelerator is provided
    by the container runtime (for example Docker `gpus: all`). Child jobs receive
    only a small environment allowlist and never receive the Worker bearer token.
    """

    def __init__(self, *, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        accelerators = tuple(
            item.strip()
            for item in os.getenv("RESEARCH_WORKER_ACCELERATORS", "cpu,gpu").split(",")
            if item.strip()
        )
        self.backend = PortableLocalProcessBackend(
            max_cpu_cores=_optional_int(os.getenv("RESEARCH_WORKER_MAX_CPU_CORES")),
            max_ram_gb=_optional_float(os.getenv("RESEARCH_WORKER_MAX_RAM_GB")),
        )
        self.backend.capabilities = BackendCapabilities(
            accelerators=accelerators or ("cpu",),
            max_vram_gb=_optional_float(os.getenv("RESEARCH_WORKER_MAX_VRAM_GB")),
            max_gpu_count=_optional_int(os.getenv("RESEARCH_WORKER_MAX_GPU_COUNT")),
            max_cpu_cores=self.backend.capabilities.max_cpu_cores,
            max_ram_gb=self.backend.capabilities.max_ram_gb,
            max_runtime_minutes=_optional_int(
                os.getenv("RESEARCH_WORKER_MAX_RUNTIME_MINUTES")
            ),
            network_available=_bool("RESEARCH_WORKER_NETWORK_AVAILABLE", True),
            detailed_progress=True,
            supports_cancel=True,
            supports_kaggle_data=False,
            domains=(Domain.RESEARCH, Domain.KAGGLE),
            tags=tuple(
                item.strip()
                for item in os.getenv(
                    "RESEARCH_WORKER_TAGS",
                    "training,inference,remote_worker,smoke_test",
                ).split(",")
                if item.strip()
            ),
        )
        self.jobs: dict[str, WorkerJob] = {}
        self._load_existing()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "detail": "ResearchAgent portable compute worker",
            "capabilities": self.backend.capabilities.to_dict(),
            "active_jobs": [
                job_id
                for job_id, item in self.jobs.items()
                if item.record.status
                not in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
            ],
        }

    def submit(self, body: Mapping[str, Any]) -> dict[str, Any]:
        raw_job = body.get("job") if isinstance(body.get("job"), Mapping) else body
        if not isinstance(raw_job, Mapping):
            raise ValueError("job must be an object")
        record = JobRecord.from_dict(raw_job) if isinstance(raw_job.get("spec"), Mapping) else JobRecord(spec=JobSpec.from_dict(raw_job))
        job_id = record.spec.job_id
        if job_id in self.jobs:
            return self._response(self.jobs[job_id].record)
        supported, reasons = self.backend.capabilities.satisfies(record.spec)
        if not supported:
            raise ValueError("worker capability mismatch: " + "; ".join(reasons))

        root = self.jobs_dir / _safe(job_id)
        workspace = root / "workspace"
        artifacts = root / "artifacts"
        workspace.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
        bundle = body.get("source_bundle")
        if not isinstance(bundle, dict):
            raise ValueError("source_bundle is required")
        bundle_result = extract_source_bundle(
            bundle,
            workspace,
            max_files=_int("RESEARCH_WORKER_MAX_BUNDLE_FILES", 5000),
            max_bytes=_int("RESEARCH_WORKER_MAX_BUNDLE_BYTES", 64 * 1024 * 1024),
        )
        metadata = {
            **record.spec.metadata,
            "workspace": str(workspace),
            "source_bundle": bundle_result,
        }
        normalized_spec = JobSpec.from_dict(
            {**record.spec.to_dict(), "metadata": metadata}
        )
        normalized = JobRecord.from_dict(
            {**record.to_dict(), "spec": normalized_spec.to_dict()}
        )
        handle = self.backend.submit(normalized, workspace)
        updated = JobRecord.from_dict(
            {
                **normalized.to_dict(),
                "status": _status(handle.status).value,
                "backend": "worker_process",
                "backend_job_id": handle.backend_job_id,
                "current_stage": handle.stage,
                "progress": handle.progress,
                "result": handle.result,
                "error": handle.error,
                "started_at": utc_timestamp(),
                "updated_at": utc_timestamp(),
            }
        )
        self.jobs[job_id] = WorkerJob(updated, workspace, artifacts)
        self._save(updated)
        return self._response(updated)

    def status(self, job_id: str) -> dict[str, Any]:
        item = self._require(job_id)
        record = item.record
        if record.status not in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            handle = self.backend.poll(record)
            record = JobRecord.from_dict(
                {
                    **record.to_dict(),
                    "status": _status(handle.status).value,
                    "current_stage": handle.stage,
                    "progress": handle.progress,
                    "result": handle.result or record.result,
                    "error": handle.error,
                    "finished_at": (
                        utc_timestamp()
                        if handle.status
                        in {
                            BackendStatus.COMPLETED,
                            BackendStatus.FAILED,
                            BackendStatus.CANCELLED,
                        }
                        else record.finished_at
                    ),
                    "updated_at": utc_timestamp(),
                }
            )
            self._store(item, record)
        return self._response(record)

    def cancel(self, job_id: str) -> dict[str, Any]:
        item = self._require(job_id)
        handle = self.backend.cancel(item.record)
        updated = JobRecord.from_dict(
            {
                **item.record.to_dict(),
                "status": JobStatus.CANCELLED.value,
                "current_stage": handle.stage,
                "progress": handle.progress,
                "error": handle.error,
                "finished_at": utc_timestamp(),
                "updated_at": utc_timestamp(),
            }
        )
        self._store(item, updated)
        return self._response(updated)

    def collect(self, job_id: str) -> dict[str, Any]:
        item = self._require(job_id)
        current = self.status(job_id)
        item = self._require(job_id)
        if item.record.status != JobStatus.COMPLETED:
            raise ValueError(
                f"job is not completed: {job_id} ({item.record.status.value})"
            )
        handle = self.backend.collect(item.record, item.artifacts)
        updated = JobRecord.from_dict(
            {
                **item.record.to_dict(),
                "current_stage": "collected",
                "result": handle.result or item.record.result,
                "updated_at": utc_timestamp(),
            }
        )
        self._store(item, updated)
        return {
            **self._response(updated),
            "artifacts": self.artifact_manifest(job_id),
        }

    def artifact_manifest(self, job_id: str) -> list[dict[str, Any]]:
        root = self._require(job_id).artifacts.resolve()
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            records.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "download_path": f"/v1/jobs/{job_id}/artifacts/{relative}",
                }
            )
        return records

    def artifact_path(self, job_id: str, relative: str) -> Path:
        root = self._require(job_id).artifacts.resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PermissionError(relative) from exc
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(relative)
        return path

    def _load_existing(self) -> None:
        for state in self.jobs_dir.glob("*/state.json"):
            try:
                record = JobRecord.from_dict(
                    json.loads(state.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
            root = state.parent
            if record.status not in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                record = JobRecord.from_dict(
                    {
                        **record.to_dict(),
                        "status": JobStatus.FAILED.value,
                        "current_stage": "worker_restarted",
                        "error": "Worker restarted while the process was active",
                        "finished_at": utc_timestamp(),
                        "updated_at": utc_timestamp(),
                    }
                )
                self._write_state(record, root / "state.json")
            self.jobs[record.spec.job_id] = WorkerJob(
                record,
                root / "workspace",
                root / "artifacts",
            )

    def _require(self, job_id: str) -> WorkerJob:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"unknown worker job: {job_id}") from exc

    def _store(self, item: WorkerJob, record: JobRecord) -> None:
        updated = WorkerJob(record, item.workspace, item.artifacts)
        self.jobs[record.spec.job_id] = updated
        self._save(record)

    def _save(self, record: JobRecord) -> None:
        self._write_state(
            record,
            self.jobs_dir / _safe(record.spec.job_id) / "state.json",
        )

    @staticmethod
    def _write_state(record: JobRecord, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _response(record: JobRecord) -> dict[str, Any]:
        return {
            "job_id": record.spec.job_id,
            "backend_job_id": record.backend_job_id or record.spec.job_id,
            "status": record.status.value,
            "stage": record.current_stage,
            "progress": record.progress,
            "result": record.result,
            "error": record.error,
            "message": f"{record.spec.job_id}: {record.status.value}",
        }


def create_app(manager: PortableWorkerJobManager | None = None):
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("Install platform API dependencies") from exc
    token = os.getenv("RESEARCH_WORKER_TOKEN", "").strip()
    if not token:
        raise RuntimeError("RESEARCH_WORKER_TOKEN is required")
    worker = manager or PortableWorkerJobManager(
        data_dir=os.getenv("RESEARCH_WORKER_DATA_DIR", "./worker-runtime")
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield

    app = FastAPI(title="ResearchAgent Portable Worker", lifespan=lifespan)

    def authorize(value: str | None) -> None:
        supplied = _bearer(value)
        if not hmac.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="invalid worker token")

    @app.get("/health")
    def health():
        return worker.health()

    @app.post("/v1/jobs")
    def submit(body: dict[str, Any], authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            return worker.submit(body)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    def status(job_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            return worker.status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            return worker.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/artifacts")
    def artifacts(job_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            response = worker.collect(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        base = os.getenv("RESEARCH_WORKER_PUBLIC_URL", "").rstrip("/")
        return {
            **response,
            "artifacts": [
                {
                    **item,
                    "url": base + item["download_path"] if base else item["download_path"],
                }
                for item in response["artifacts"]
            ],
        }

    @app.get("/v1/jobs/{job_id}/artifacts/{relative:path}")
    def artifact(
        job_id: str,
        relative: str,
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        try:
            path = worker.artifact_path(job_id, relative)
        except (KeyError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent portable compute worker")
    parser.add_argument("--host", default=os.getenv("RESEARCH_WORKER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("RESEARCH_WORKER_PORT", "8090")))
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install platform API dependencies") from exc
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


def _bearer(value: str | None) -> str:
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _status(status: BackendStatus) -> JobStatus:
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


def _safe(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:100] or "job"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


if __name__ == "__main__":
    main()
