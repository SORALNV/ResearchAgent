from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import urllib.parse
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from harness.artifacts import build_artifact_manifest
from harness.compute_backends import LocalGpuInventory, LocalProcessBackend
from harness.compute_bundle import extract_source_bundle
from harness.compute_models import (
    BackendCapabilities,
    BackendHandle,
    BackendState,
    safe_relative_path,
)
from harness.control_plane import Domain, Job
from harness.state import utc_timestamp


@dataclass(frozen=True)
class WorkerJobRecord:
    job: Job
    handle: BackendHandle
    workspace: str
    artifacts_dir: str
    collected: bool = False
    updated_at: str = utc_timestamp()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.to_dict(),
            "handle": self.handle.to_dict(),
            "workspace": self.workspace,
            "artifacts_dir": self.artifacts_dir,
            "collected": self.collected,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkerJobRecord":
        return cls(
            job=Job.from_dict(data["job"]),
            handle=BackendHandle.from_dict(data["handle"]),
            workspace=str(data["workspace"]),
            artifacts_dir=str(data["artifacts_dir"]),
            collected=bool(data.get("collected", False)),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


class PortableComputeWorker:
    """Credential-isolated execution service for an owned or rented GPU host."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        capabilities: BackendCapabilities | None = None,
        public_url: str = "",
        max_bundle_files: int = 5000,
        max_bundle_bytes: int = 64 * 1024 * 1024,
        artifact_max_files: int = 2000,
        artifact_max_bytes: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.public_url = public_url.rstrip("/")
        self.max_bundle_files = max(1, max_bundle_files)
        self.max_bundle_bytes = max(1, max_bundle_bytes)
        self.artifact_max_files = max(1, artifact_max_files)
        self.artifact_max_bytes = max(1, artifact_max_bytes)
        self.capabilities = capabilities or worker_capabilities_from_env()
        inventory = LocalGpuInventory(
            gpu_count=self.capabilities.gpu_count,
            gpu_memory_mb=self.capabilities.gpu_memory_mb,
            detail="worker-advertised GPU inventory",
        )
        self.backend = LocalProcessBackend(
            name="worker_process",
            gpu="gpu" in self.capabilities.accelerators,
            inventory=inventory,
            max_cpu_cores=self.capabilities.cpu_cores,
            max_memory_mb=self.capabilities.memory_mb,
            max_storage_mb=self.capabilities.ephemeral_storage_mb,
        )
        self.backend.capabilities = self.capabilities
        self.records: dict[str, WorkerJobRecord] = {}
        self._load_existing()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "detail": "ResearchAgent portable compute worker",
            "capabilities": self.capabilities.to_dict(),
            "jobs": len(self.records),
            "active_jobs": sorted(
                job_id
                for job_id, record in self.records.items()
                if not record.handle.terminal
            ),
        }

    def submit(self, body: Mapping[str, Any]) -> dict[str, Any]:
        raw_job = body.get("job")
        if not isinstance(raw_job, Mapping):
            raise ValueError("job must be an object")
        job = Job.from_dict(raw_job)
        existing = self.records.get(job.job_id)
        if existing is not None:
            return self._status_payload(existing)
        supported, reasons = self.capabilities.satisfies(job.spec)
        if not supported:
            raise ValueError("worker capability mismatch: " + "; ".join(reasons))
        bundle = body.get("source_bundle")
        if not isinstance(bundle, dict):
            raise ValueError("source_bundle is required")

        root = self.jobs_dir / _safe_component(job.job_id)
        workspace = root / "workspace"
        artifacts = root / "artifacts"
        workspace.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
        bundle_result = extract_source_bundle(
            bundle,
            workspace,
            max_files=self.max_bundle_files,
            max_bytes=self.max_bundle_bytes,
        )
        payload = {
            **job.spec.payload,
            "workspace": str(workspace),
            "source_bundle": bundle_result,
        }
        effective = replace(job, spec=replace(job.spec, payload=payload))
        handle = self.backend.submit(effective, workspace)
        record = WorkerJobRecord(
            job=effective,
            handle=handle,
            workspace=str(workspace),
            artifacts_dir=str(artifacts),
            updated_at=utc_timestamp(),
        )
        self._save(record)
        return self._status_payload(record)

    def status(self, job_id: str) -> dict[str, Any]:
        record = self._require(job_id)
        if not record.handle.terminal:
            handle = self.backend.poll(record.job, record.handle)
            record = replace(record, handle=handle, updated_at=utc_timestamp())
            self._save(record)
        return self._status_payload(record)

    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self._require(job_id)
        handle = self.backend.cancel(record.job, record.handle)
        record = replace(record, handle=handle, updated_at=utc_timestamp())
        self._save(record)
        return self._status_payload(record)

    def collect(self, job_id: str) -> dict[str, Any]:
        self.status(job_id)
        record = self._require(job_id)
        if record.handle.state != BackendState.SUCCEEDED:
            raise ValueError(
                f"job is not successful: {job_id} ({record.handle.state.value})"
            )
        artifacts = Path(record.artifacts_dir).expanduser().resolve()
        if not record.collected:
            collected = self.backend.collect(
                record.job,
                record.handle,
                artifacts,
            )
            manifest, warnings = build_artifact_manifest(
                artifacts,
                max_files=self.artifact_max_files,
                max_bytes=self.artifact_max_bytes,
            )
            record = replace(
                record,
                handle=replace(
                    record.handle,
                    result=collected.result or record.handle.result,
                    metadata={
                        **record.handle.metadata,
                        "collection": {
                            "artifact_paths": list(collected.artifact_paths),
                            "warnings": [*collected.warnings, *warnings],
                        },
                    },
                ),
                collected=True,
                updated_at=utc_timestamp(),
            )
            self._save(record)
        return {
            **self._status_payload(record),
            "artifacts": self.artifact_manifest(job_id),
        }

    def artifact_manifest(self, job_id: str) -> list[dict[str, Any]]:
        root = Path(self._require(job_id).artifacts_dir).expanduser().resolve()
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            download_path = f"/v1/jobs/{job_id}/artifacts/{relative}"
            records.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "download_path": download_path,
                    "url": self.public_url + download_path
                    if self.public_url
                    else download_path,
                }
            )
        return records

    def artifact_path(self, job_id: str, relative: str) -> Path:
        safe = safe_relative_path(relative)
        if not safe:
            raise FileNotFoundError(relative)
        root = Path(self._require(job_id).artifacts_dir).expanduser().resolve()
        target = (root / safe).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise FileNotFoundError(relative) from exc
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError(relative)
        return target

    def _load_existing(self) -> None:
        for state in self.jobs_dir.glob("*/state.json"):
            try:
                value = json.loads(state.read_text(encoding="utf-8"))
                if isinstance(value, Mapping):
                    record = WorkerJobRecord.from_dict(value)
                    self.records[record.job.job_id] = record
            except Exception:
                continue

    def _save(self, record: WorkerJobRecord) -> None:
        self.records[record.job.job_id] = record
        root = self.jobs_dir / _safe_component(record.job.job_id)
        _atomic_json(root / "state.json", record.to_dict())

    def _require(self, job_id: str) -> WorkerJobRecord:
        try:
            return self.records[job_id]
        except KeyError as exc:
            raise KeyError(f"unknown worker job: {job_id}") from exc

    @staticmethod
    def _status_payload(record: WorkerJobRecord) -> dict[str, Any]:
        return {
            "job_id": record.job.job_id,
            "backend_job_id": record.job.job_id,
            "status": record.handle.state.value,
            "stage": record.handle.stage,
            "progress": record.handle.progress,
            "message": record.handle.message,
            "result": record.handle.result,
            "error": record.handle.error,
            "updated_at": record.updated_at,
        }


class WorkerRequestHandler(BaseHTTPRequestHandler):
    server_version = "ResearchAgentWorker/0.2"

    @property
    def worker(self) -> PortableComputeWorker:
        return self.server.worker  # type: ignore[attr-defined]

    @property
    def token(self) -> str:
        return self.server.worker_token  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            self._json(HTTPStatus.OK, self.worker.health())
            return
        if not self._authorized():
            return
        parts = [part for part in parsed.path.split("/") if part]
        try:
            if len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
                self._json(HTTPStatus.OK, self.worker.status(parts[2]))
                return
            if (
                len(parts) == 4
                and parts[:2] == ["v1", "jobs"]
                and parts[3] == "artifacts"
            ):
                self._json(HTTPStatus.OK, self.worker.collect(parts[2]))
                return
            if (
                len(parts) >= 5
                and parts[:2] == ["v1", "jobs"]
                and parts[3] == "artifacts"
            ):
                relative = urllib.parse.unquote("/".join(parts[4:]))
                self._file(self.worker.artifact_path(parts[2], relative))
                return
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        except (ValueError, FileNotFoundError) as exc:
            self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        parsed = urllib.parse.urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        try:
            body = self._body()
            if parts == ["v1", "jobs"]:
                self._json(HTTPStatus.ACCEPTED, self.worker.submit(body))
                return
            if (
                len(parts) == 4
                and parts[:2] == ["v1", "jobs"]
                and parts[3] == "cancel"
            ):
                self._json(HTTPStatus.OK, self.worker.cancel(parts[2]))
                return
        except KeyError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, format: str, *args: Any) -> None:
        if os.getenv("WORKER_HTTP_LOG", "true").lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return
        super().log_message(format, *args)

    def _authorized(self) -> bool:
        value = self.headers.get("Authorization", "")
        scheme, _, supplied = value.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            supplied.strip(), self.token
        ):
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "invalid worker token"})
            return False
        return True

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > _int_env("WORKER_MAX_REQUEST_BYTES", 96 * 1024 * 1024):
            raise ValueError("invalid request body size")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def _json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-SHA256", _sha256(path))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class WorkerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        worker: PortableComputeWorker,
        token: str,
    ) -> None:
        self.worker = worker
        self.worker_token = token
        super().__init__(address, WorkerRequestHandler)


def worker_capabilities_from_env() -> BackendCapabilities:
    accelerators = _csv_env("WORKER_ACCELERATORS") or ("cpu", "gpu")
    domains: list[Domain] = []
    for item in _csv_env("WORKER_DOMAINS") or ("research", "kaggle"):
        try:
            domains.append(Domain(item))
        except ValueError:
            continue
    return BackendCapabilities(
        accelerators=accelerators,
        domains=tuple(domains) or (Domain.RESEARCH, Domain.KAGGLE),
        gpu_count=_int_env("WORKER_GPU_COUNT", 1 if "gpu" in accelerators else 0),
        gpu_memory_mb=_optional_int_env("WORKER_GPU_MEMORY_MB"),
        cpu_cores=_optional_float_env("WORKER_CPU_CORES"),
        memory_mb=_optional_int_env("WORKER_MEMORY_MB"),
        ephemeral_storage_mb=_optional_int_env("WORKER_STORAGE_MB"),
        network_available=_bool_env("WORKER_NETWORK_AVAILABLE", True),
        labels=_csv_env("WORKER_LABELS")
        or ("training", "inference", "remote_worker", "smoke_test"),
        supports_cancel=True,
        recoverable=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent external compute worker")
    parser.add_argument("--host", default=os.getenv("WORKER_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=_int_env("WORKER_PORT", 8090),
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("WORKER_DATA_DIR", "./worker-runtime"),
    )
    args = parser.parse_args()
    token = os.getenv("WORKER_TOKEN", "").strip()
    if not token:
        raise SystemExit("WORKER_TOKEN is required")
    worker = PortableComputeWorker(
        data_dir=args.data_dir,
        capabilities=worker_capabilities_from_env(),
        public_url=os.getenv("WORKER_PUBLIC_URL", ""),
        max_bundle_files=_int_env("WORKER_MAX_BUNDLE_FILES", 5000),
        max_bundle_bytes=_int_env(
            "WORKER_MAX_BUNDLE_BYTES", 64 * 1024 * 1024
        ),
        artifact_max_files=_int_env("WORKER_ARTIFACT_MAX_FILES", 2000),
        artifact_max_bytes=_int_env(
            "WORKER_ARTIFACT_MAX_BYTES", 4 * 1024 * 1024 * 1024
        ),
    )
    server = WorkerHTTPServer((args.host, args.port), worker, token)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _safe_component(value: str) -> str:
    cleaned = "".join(
        character
        if character.isalnum() or character in "-_"
        else "-"
        for character in str(value)
    )
    return cleaned.strip("-")[:120] or "job"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in {None, ""} else None
    except ValueError:
        return None


def _optional_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in {None, ""} else None
    except ValueError:
        return None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip().lower()
            for item in os.getenv(name, "").split(",")
            if item.strip()
        )
    )


if __name__ == "__main__":
    main()
