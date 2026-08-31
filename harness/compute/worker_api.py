from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import shutil
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from harness.compute.local import LocalProcessBackend
from harness.platform.models import JobRecord, JobSpec, JobStatus
from harness.state import utc_timestamp


@dataclass
class WorkerJob:
    record: JobRecord
    workspace: Path
    result_dir: Path
    created_at: str


class WorkerJobManager:
    """Execute trusted Core jobs inside the isolated GPU Worker container.

    The Worker receives no Discord, Kaggle, or OpenAI credentials. Child
    processes get LocalProcessBackend's allowlisted environment. All writable
    paths remain under WORKER_DATA_DIR. After restart, terminal jobs are loaded;
    jobs that were running are marked failed because an OS process cannot be
    safely reattached by PID alone.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path,
        max_cpu_cores: int | None = None,
        max_ram_gb: float | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.backend = LocalProcessBackend(
            max_cpu_cores=max_cpu_cores,
            max_ram_gb=max_ram_gb,
        )
        self._lock = threading.RLock()
        self._jobs: dict[str, WorkerJob] = {}
        self._load_existing()

    def health(self) -> dict[str, Any]:
        with self._lock:
            active = [
                job.record.spec.job_id
                for job in self._jobs.values()
                if job.record.status
                not in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
            ]
        return {
            "ok": True,
            "detail": "ResearchAgent portable compute worker",
            "capabilities": self.backend.capabilities.to_dict(),
            "active_jobs": active,
            "data_dir": str(self.data_dir),
        }

    def submit(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        record = self._parse_record(raw)
        job_id = record.spec.job_id
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None:
                return self._response(existing.record)
        workspace = self.jobs_dir / _safe(job_id) / "workspace"
        result_dir = self.jobs_dir / _safe(job_id) / "artifacts"
        workspace.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        spec_metadata = dict(record.spec.metadata)
        spec_metadata["workspace"] = str(workspace)
        spec = JobSpec.from_dict({**record.spec.to_dict(), "metadata": spec_metadata})
        normalized = JobRecord.from_dict({**record.to_dict(), "spec": spec.to_dict()})
        self._write_record(normalized, workspace.parent)
        handle = self.backend.submit(normalized, workspace)
        updated = JobRecord.from_dict(
            {
                **normalized.to_dict(),
                "status": _job_status(handle.status.value).value,
                "backend": "local_process",
                "backend_job_id": handle.backend_job_id,
                "current_stage": handle.stage,
                "progress": handle.progress,
                "result": handle.result,
                "error": handle.error,
                "started_at": utc_timestamp(),
                "updated_at": utc_timestamp(),
            }
        )
        worker_job = WorkerJob(updated, workspace, result_dir, utc_timestamp())
        with self._lock:
            self._jobs[job_id] = worker_job
        self._write_record(updated, workspace.parent)
        return self._response(updated)

    def status(self, job_id: str) -> dict[str, Any]:
        worker_job = self._require(job_id)
        record = worker_job.record
        if record.status not in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            handle = self.backend.poll(record)
            record = JobRecord.from_dict(
                {
                    **record.to_dict(),
                    "status": _job_status(handle.status.value).value,
                    "backend_job_id": handle.backend_job_id,
                    "current_stage": handle.stage,
                    "progress": handle.progress,
                    "result": handle.result or record.result,
                    "error": handle.error,
                    "finished_at": (
                        utc_timestamp()
                        if handle.status.value in {"completed", "failed", "cancelled"}
                        else record.finished_at
                    ),
                    "updated_at": utc_timestamp(),
                }
            )
            self._store(worker_job, record)
        return self._response(record)

    def cancel(self, job_id: str) -> dict[str, Any]:
        worker_job = self._require(job_id)
        handle = self.backend.cancel(worker_job.record)
        updated = JobRecord.from_dict(
            {
                **worker_job.record.to_dict(),
                "status": JobStatus.CANCELLED.value,
                "current_stage": handle.stage,
                "progress": handle.progress,
                "error": handle.error,
                "finished_at": utc_timestamp(),
                "updated_at": utc_timestamp(),
            }
        )
        self._store(worker_job, updated)
        return self._response(updated)

    def collect(self, job_id: str) -> dict[str, Any]:
        worker_job = self._require(job_id)
        record = worker_job.record
        if record.status != JobStatus.COMPLETED:
            raise ValueError(f"job is not completed: {job_id} ({record.status.value})")
        handle = self.backend.collect(record, worker_job.result_dir)
        result = handle.result or record.result
        updated = JobRecord.from_dict(
            {
                **record.to_dict(),
                "current_stage": "collected",
                "result": result,
                "updated_at": utc_timestamp(),
            }
        )
        self._store(worker_job, updated)
        artifacts = self.artifact_manifest(job_id)
        return {
            **self._response(updated),
            "result": result,
            "artifacts": artifacts,
        }

    def artifact_manifest(self, job_id: str) -> list[dict[str, Any]]:
        worker_job = self._require(job_id)
        records: list[dict[str, Any]] = []
        root = worker_job.result_dir.resolve()
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
        worker_job = self._require(job_id)
        root = worker_job.result_dir.resolve()
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PermissionError("artifact path escapes result directory") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(relative)
        return candidate

    def _parse_record(self, raw: Mapping[str, Any]) -> JobRecord:
        value = raw.get("job") if isinstance(raw.get("job"), Mapping) else raw
        if not isinstance(value, Mapping):
            raise ValueError("job payload must be an object")
        if isinstance(value.get("spec"), Mapping):
            record = JobRecord.from_dict(value)
        else:
            spec = JobSpec.from_dict(value)
            record = JobRecord(spec=spec)
        if record.spec.resources.accelerator not in self.backend.capabilities.accelerators:
            raise ValueError(
                f"worker does not support accelerator {record.spec.resources.accelerator}"
            )
        return record

    def _load_existing(self) -> None:
        for state_path in self.jobs_dir.glob("*/state.json"):
            try:
                value = json.loads(state_path.read_text(encoding="utf-8"))
                record = JobRecord.from_dict(value)
            except Exception:
                continue
            workspace = state_path.parent / "workspace"
            result_dir = state_path.parent / "artifacts"
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
                        "error": "Worker restarted while the local process was active",
                        "finished_at": utc_timestamp(),
                        "updated_at": utc_timestamp(),
                    }
                )
                self._write_record(record, state_path.parent)
            self._jobs[record.spec.job_id] = WorkerJob(
                record=record,
                workspace=workspace,
                result_dir=result_dir,
                created_at=record.spec.created_at,
            )

    def _require(self, job_id: str) -> WorkerJob:
        with self._lock:
            value = self._jobs.get(job_id)
        if value is None:
            raise KeyError(f"unknown worker job: {job_id}")
        return value

    def _store(self, worker_job: WorkerJob, record: JobRecord) -> None:
        updated = WorkerJob(
            record=record,
            workspace=worker_job.workspace,
            result_dir=worker_job.result_dir,
            created_at=worker_job.created_at,
        )
        with self._lock:
            self._jobs[record.spec.job_id] = updated
        self._write_record(record, worker_job.workspace.parent)

    @staticmethod
    def _write_record(record: JobRecord, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "state.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

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


def create_worker_app(manager: WorkerJobManager | None = None):
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("Install with `pip install -e '.[api]'`") from exc

    token = os.getenv("RESEARCH_WORKER_TOKEN", "").strip()
    if not token:
        raise RuntimeError("RESEARCH_WORKER_TOKEN is required")
    worker = manager or WorkerJobManager(
        data_dir=os.getenv("RESEARCH_WORKER_DATA_DIR", "./worker-runtime")
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield

    app = FastAPI(title="ResearchAgent Compute Worker", lifespan=lifespan)

    def authorize(authorization: str | None = Header(default=None)) -> None:
        supplied = _bearer(authorization)
        if not hmac.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="invalid worker token")

    @app.get("/health")
    def health() -> dict[str, Any]:
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
            collected = worker.collect(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        base = os.getenv("RESEARCH_WORKER_PUBLIC_URL", "").rstrip("/")
        result = []
        for item in collected["artifacts"]:
            result.append(
                {
                    **item,
                    "url": base + item["download_path"] if base else item["download_path"],
                }
            )
        return {**collected, "artifacts": result}

    @app.get("/v1/jobs/{job_id}/artifacts/{relative:path}")
    def artifact(
        job_id: str,
        relative: str,
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        try:
            path = worker.artifact_path(job_id, relative)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (PermissionError, FileNotFoundError) as exc:
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
        raise SystemExit("Install with `pip install -e '.[api]'`") from exc
    uvicorn.run(create_worker_app(), host=args.host, port=args.port, log_level="info")


def _bearer(value: str | None) -> str:
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _job_status(value: str) -> JobStatus:
    mapping = {
        "created": JobStatus.CREATED,
        "queued": JobStatus.QUEUED,
        "preparing": JobStatus.PREPARING,
        "submitted": JobStatus.SUBMITTED,
        "running": JobStatus.RUNNING,
        "collecting": JobStatus.COLLECTING,
        "completed": JobStatus.COMPLETED,
        "failed": JobStatus.FAILED,
        "cancel_requested": JobStatus.CANCEL_REQUESTED,
        "cancelled": JobStatus.CANCELLED,
        "unknown": JobStatus.BLOCKED,
    }
    return mapping.get(value, JobStatus.BLOCKED)


def _safe(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:100] or "job"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
