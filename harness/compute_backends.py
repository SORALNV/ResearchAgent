from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from harness.compute_bundle import build_source_bundle
from harness.compute_models import (
    BackendCapabilities,
    BackendHandle,
    BackendState,
    CollectedResult,
    ComputeBackend,
    job_entrypoint,
    job_outputs,
    safe_relative_path,
)
from harness.control_plane import Domain, Job
from harness.state import utc_timestamp


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], CommandResult]


@dataclass(frozen=True)
class LocalGpuInventory:
    gpu_count: int = 0
    gpu_memory_mb: int | None = None
    detail: str = "no local GPU detected"


class FakeComputeBackend:
    """Pure deterministic backend for scheduler and feedback tests."""

    def __init__(
        self,
        *,
        name: str = "fake",
        capabilities: BackendCapabilities | None = None,
        available: bool = True,
        complete_after_polls: int = 2,
        result: Mapping[str, Any] | None = None,
        fail: bool = False,
        approval_required: bool = False,
    ) -> None:
        self.name = name
        self.capabilities = capabilities or BackendCapabilities(
            accelerators=("cpu", "gpu"),
            domains=(Domain.RESEARCH, Domain.KAGGLE),
            gpu_count=8,
            gpu_memory_mb=96 * 1024,
            cpu_cores=128,
            memory_mb=1024 * 1024,
            network_available=True,
            labels=("training", "inference", "testing", "kaggle_data"),
            recoverable=True,
        )
        self._available = available
        self.complete_after_polls = max(1, complete_after_polls)
        self._result = dict(
            result
            or {
                "summary": "deterministic fake experiment completed",
                "metrics": {"score": 0.75},
                "primary_metric": {
                    "name": "score",
                    "value": 0.75,
                    "direction": "maximize",
                },
            }
        )
        self.fail = fail
        self.approval_required = approval_required

    def available(self) -> tuple[bool, str]:
        return self._available, "deterministic fake compute backend"

    def prepare(self, job: Job, workspace: Path) -> dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        manifest = {
            "job": job.to_dict(),
            "backend": self.name,
            "workspace": str(workspace),
        }
        _atomic_json(workspace / "prepared_job.json", manifest)
        return manifest

    def submit(self, job: Job, workspace: Path) -> BackendHandle:
        self.prepare(job, workspace)
        return BackendHandle(
            backend=self.name,
            backend_job_id=f"{self.name}-{job.job_id}",
            state=BackendState.QUEUED,
            stage="queued",
            progress=0.0,
            message="Fake job queued",
            metadata={"workspace": str(workspace), "poll_count": 0},
        )

    def poll(self, job: Job, handle: BackendHandle) -> BackendHandle:
        if handle.terminal:
            return handle
        count = int(handle.metadata.get("poll_count") or 0) + 1
        metadata = {**handle.metadata, "poll_count": count}
        if count < self.complete_after_polls:
            return replace(
                handle,
                state=BackendState.RUNNING,
                stage="execute",
                progress=min(0.9, count / self.complete_after_polls),
                message="Fake job running",
                metadata=metadata,
                updated_at=utc_timestamp(),
            )
        if self.fail:
            return replace(
                handle,
                state=BackendState.FAILED,
                stage="failed",
                progress=1.0,
                message="Fake job failed",
                error="configured fake failure",
                metadata=metadata,
                updated_at=utc_timestamp(),
            )
        return replace(
            handle,
            state=BackendState.SUCCEEDED,
            stage="completed",
            progress=1.0,
            message="Fake job completed",
            result=dict(self._result),
            metadata=metadata,
            updated_at=utc_timestamp(),
        )

    def cancel(self, job: Job, handle: BackendHandle | None) -> BackendHandle:
        current = handle or BackendHandle(
            backend=self.name,
            backend_job_id=f"{self.name}-{job.job_id}",
            state=BackendState.QUEUED,
            stage="queued",
        )
        return replace(
            current,
            state=BackendState.CANCELLED,
            stage="cancelled",
            message="Fake job cancelled",
            updated_at=utc_timestamp(),
        )

    def collect(
        self,
        job: Job,
        handle: BackendHandle,
        destination: Path,
    ) -> CollectedResult:
        destination.mkdir(parents=True, exist_ok=True)
        result = dict(handle.result or self._result)
        _atomic_json(destination / "result.json", result)
        _atomic_json(destination / "metrics.json", result.get("metrics") or {})
        (destination / "stdout.log").write_text(
            "fake backend completed\n", encoding="utf-8"
        )
        return CollectedResult(
            result=result,
            artifact_paths=("result.json", "metrics.json", "stdout.log"),
            metadata={"destination": str(destination)},
        )


class LocalProcessBackend:
    """Recoverable local CPU or NVIDIA GPU process backend.

    Commands are argv-only and run through ``harness.compute_process``. The
    child receives a strict environment allowlist; Discord, OpenAI, Kaggle, and
    Worker credentials are never copied into the experiment process.
    """

    approval_required = False

    def __init__(
        self,
        *,
        name: str,
        gpu: bool,
        inventory: LocalGpuInventory | None = None,
        max_cpu_cores: float | None = None,
        max_memory_mb: int | None = None,
        max_storage_mb: int | None = None,
    ) -> None:
        self.name = name
        self.gpu = bool(gpu)
        self.inventory = inventory or detect_local_gpu()
        accelerators = ("gpu",) if self.gpu else ("cpu",)
        labels = (
            ("training", "inference", "smoke_test", "local_gpu")
            if self.gpu
            else ("preprocessing", "validation", "inference", "local_cpu")
        )
        self.capabilities = BackendCapabilities(
            accelerators=accelerators,
            domains=(Domain.RESEARCH, Domain.KAGGLE),
            gpu_count=self.inventory.gpu_count if self.gpu else 0,
            gpu_memory_mb=self.inventory.gpu_memory_mb if self.gpu else None,
            cpu_cores=max_cpu_cores or float(max(1, os.cpu_count() or 1)),
            memory_mb=max_memory_mb,
            ephemeral_storage_mb=max_storage_mb,
            network_available=_bool_env("LOCAL_COMPUTE_NETWORK_AVAILABLE", False),
            labels=labels,
            supports_cancel=True,
            recoverable=True,
        )

    def available(self) -> tuple[bool, str]:
        if self.gpu:
            enabled = os.getenv("LOCAL_GPU_ENABLED", "auto").strip().lower()
            if enabled in {"0", "false", "no", "off"}:
                return False, "LOCAL_GPU_ENABLED disables the local GPU backend"
            if self.inventory.gpu_count <= 0:
                return False, self.inventory.detail
            return True, self.inventory.detail
        return True, f"local CPU cores={self.capabilities.cpu_cores:g}"

    def prepare(self, job: Job, workspace: Path) -> dict[str, Any]:
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        command = _job_command(job)
        if not command:
            raise ValueError("local compute job requires payload.entrypoint")
        _validate_command_paths(command, workspace)
        exit_path = workspace / ".compute_exit.json"
        spec_path = workspace / ".compute_process_spec.json"
        manifest = {
            "job_id": job.job_id,
            "command": command,
            "cwd": str(workspace),
            "exit_path": str(exit_path),
            "timeout_seconds": job.spec.max_runtime_seconds,
            "resources": job.spec.resources.to_dict(),
            "backend": self.name,
        }
        _atomic_json(spec_path, manifest)
        return {
            **manifest,
            "spec_path": str(spec_path),
            "workspace": str(workspace),
        }

    def submit(self, job: Job, workspace: Path) -> BackendHandle:
        prepared = self.prepare(job, workspace)
        exit_path = Path(str(prepared["exit_path"]))
        exit_path.unlink(missing_ok=True)
        stdout_path = workspace / "stdout.log"
        stderr_path = workspace / "stderr.log"
        stdout_handle = stdout_path.open("ab", buffering=0)
        stderr_handle = stderr_path.open("ab", buffering=0)
        command = [
            sys.executable,
            "-m",
            "harness.compute_process",
            "--spec",
            str(prepared["spec_path"]),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=workspace,
                env=self._environment(job, workspace),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=os.name != "nt",
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
        return BackendHandle(
            backend=self.name,
            backend_job_id=str(process.pid),
            state=BackendState.RUNNING,
            stage="running",
            progress=0.05,
            message=f"Local process wrapper started: pid={process.pid}",
            metadata={
                "workspace": str(workspace),
                "pid": process.pid,
                "exit_path": str(exit_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "command": prepared["command"],
            },
        )

    def poll(self, job: Job, handle: BackendHandle) -> BackendHandle:
        workspace = _handle_workspace(handle)
        exit_path = Path(
            str(handle.metadata.get("exit_path") or workspace / ".compute_exit.json")
        )
        if exit_path.is_file():
            exit_record = _read_json(exit_path)
            returncode = int(exit_record.get("returncode") or 0)
            if returncode == 0:
                return replace(
                    handle,
                    state=BackendState.SUCCEEDED,
                    stage="completed",
                    progress=1.0,
                    message=_tail(workspace / "stdout.log", 2000)
                    or "Local compute completed",
                    result=_read_result(workspace),
                    error=None,
                    metadata={**handle.metadata, "exit": exit_record},
                    updated_at=utc_timestamp(),
                )
            return replace(
                handle,
                state=BackendState.FAILED,
                stage="failed",
                progress=1.0,
                message="Local compute failed",
                error=(
                    str(exit_record.get("error") or "")
                    or _tail(workspace / "stderr.log", 4000)
                    or f"returncode={returncode}"
                ),
                metadata={**handle.metadata, "exit": exit_record},
                updated_at=utc_timestamp(),
            )

        pid = _optional_int(handle.metadata.get("pid") or handle.backend_job_id)
        if pid and _pid_alive(pid):
            progress, stage = _read_progress(workspace)
            return replace(
                handle,
                state=BackendState.RUNNING,
                stage=stage,
                progress=max(handle.progress, progress),
                message=_tail(workspace / "stdout.log", 1200)
                or "Local compute running",
                updated_at=utc_timestamp(),
            )
        return replace(
            handle,
            state=BackendState.UNKNOWN,
            stage="process_lost",
            message="Local process exited without a durable exit marker",
            error="process handle is unavailable and .compute_exit.json is missing",
            updated_at=utc_timestamp(),
        )

    def cancel(self, job: Job, handle: BackendHandle | None) -> BackendHandle:
        current = handle or BackendHandle(
            backend=self.name,
            backend_job_id="unknown",
            state=BackendState.UNKNOWN,
            stage="unknown",
        )
        pid = _optional_int(current.metadata.get("pid") or current.backend_job_id)
        if pid:
            _terminate_pid(pid)
        return replace(
            current,
            state=BackendState.CANCELLED,
            stage="cancelled",
            message="Local process group cancellation requested",
            updated_at=utc_timestamp(),
        )

    def collect(
        self,
        job: Job,
        handle: BackendHandle,
        destination: Path,
    ) -> CollectedResult:
        workspace = _handle_workspace(handle)
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        requested = list(job_outputs(job))
        for default in (
            "result.json",
            "metrics.json",
            "progress.json",
            "stdout.log",
            "stderr.log",
            ".compute_exit.json",
        ):
            if default not in requested:
                requested.append(default)
        collected: list[str] = []
        warnings: list[str] = []
        for relative in requested:
            safe = safe_relative_path(relative)
            if not safe:
                warnings.append(f"unsafe output ignored: {relative}")
                continue
            source = (workspace / safe).resolve()
            try:
                source.relative_to(workspace)
            except ValueError:
                warnings.append(f"output escapes workspace: {relative}")
                continue
            if source.is_symlink() or not source.is_file():
                continue
            target = (destination / safe).resolve()
            try:
                target.relative_to(destination)
            except ValueError:
                warnings.append(f"unsafe destination ignored: {relative}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            collected.append(safe)
        return CollectedResult(
            result=_read_result(workspace) or dict(handle.result),
            artifact_paths=tuple(collected),
            warnings=tuple(warnings),
            metadata={"workspace": str(workspace), "destination": str(destination)},
        )

    @staticmethod
    def _environment(job: Job, workspace: Path) -> dict[str, str]:
        allowed_names = {
            "PATH",
            "HOME",
            "USERPROFILE",
            "LANG",
            "LC_ALL",
            "PYTHONPATH",
            "PYTHONHOME",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "TMP",
            "TEMP",
            "TMPDIR",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "LD_LIBRARY_PATH",
        }
        environment = {
            key: value for key, value in os.environ.items() if key in allowed_names
        }
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "RESEARCH_AGENT_JOB_ID": job.job_id,
                "RESEARCH_AGENT_PROJECT_ID": job.spec.project_id,
                "RESEARCH_AGENT_WORK_SESSION_ID": job.spec.work_session_id,
                "RESEARCH_AGENT_WORKSPACE": str(workspace),
            }
        )
        return environment


class KaggleNotebookBackend:
    """Kaggle Kernels/Notebooks backend with Core-only credentials."""

    name = "kaggle_notebook"
    approval_required = False
    capabilities = BackendCapabilities(
        accelerators=("cpu", "gpu"),
        domains=(Domain.KAGGLE,),
        gpu_count=1,
        gpu_memory_mb=None,
        network_available=True,
        labels=("training", "inference", "notebook", "kaggle_data"),
        supports_cancel=False,
        recoverable=True,
    )

    def __init__(
        self,
        *,
        command: str = "kaggle",
        api_token: str | None = None,
        username: str | None = None,
        command_runner: CommandRunner | None = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.command = command.strip()
        self.api_token = api_token
        self.username = username
        self.command_runner = command_runner or self._default_runner
        self.timeout_seconds = max(10, timeout_seconds)

    def available(self) -> tuple[bool, str]:
        if not self.command:
            return False, "Kaggle command is empty"
        if not self.api_token and not _has_kaggle_config():
            return False, "KAGGLE_API_TOKEN or Kaggle CLI config is unavailable"
        return True, self.command

    def prepare(self, job: Job, workspace: Path) -> dict[str, Any]:
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        metadata_path = workspace / "kernel-metadata.json"
        if not metadata_path.is_file():
            _atomic_json(metadata_path, self._metadata_from_job(job))
        metadata = _read_json(metadata_path)
        kernel_id = str(metadata.get("id") or "").strip()
        if "/" not in kernel_id:
            raise ValueError("Kaggle kernel metadata id must be owner/slug")
        code_file = _metadata_code_file(metadata)
        if not code_file or not (workspace / code_file).is_file():
            raise FileNotFoundError(f"Kaggle code file not found: {code_file}")
        return {
            "kernel_id": kernel_id,
            "code_file": code_file,
            "metadata_path": str(metadata_path),
            "workspace": str(workspace),
        }

    def submit(self, job: Job, workspace: Path) -> BackendHandle:
        prepared = self.prepare(job, workspace)
        result = self.command_runner(
            [self.command, "kernels", "push", "-p", str(workspace)],
            workspace,
            self._environment(),
        )
        if result.returncode != 0:
            return BackendHandle(
                backend=self.name,
                backend_job_id=str(prepared["kernel_id"]),
                state=BackendState.FAILED,
                stage="submit_failed",
                progress=1.0,
                message="Kaggle kernel push failed",
                error=(result.stderr or result.stdout)[-4000:],
                metadata=prepared,
            )
        return BackendHandle(
            backend=self.name,
            backend_job_id=str(prepared["kernel_id"]),
            state=BackendState.QUEUED,
            stage="submitted",
            progress=0.05,
            message=result.stdout.strip() or "Kaggle kernel submitted",
            metadata=prepared,
        )

    def poll(self, job: Job, handle: BackendHandle) -> BackendHandle:
        workspace = _handle_workspace(handle)
        result = self.command_runner(
            [self.command, "kernels", "status", handle.backend_job_id],
            workspace,
            self._environment(),
        )
        if result.returncode != 0:
            return replace(
                handle,
                state=BackendState.UNKNOWN,
                stage="status_error",
                message="Unable to read Kaggle kernel status",
                error=(result.stderr or result.stdout)[-4000:],
                updated_at=utc_timestamp(),
            )
        state, progress = _parse_kaggle_status(result.stdout + "\n" + result.stderr)
        return replace(
            handle,
            state=state,
            stage=state.value,
            progress=max(handle.progress, progress),
            message=(result.stdout or result.stderr).strip(),
            error=None,
            updated_at=utc_timestamp(),
        )

    def cancel(self, job: Job, handle: BackendHandle | None) -> BackendHandle:
        current = handle or BackendHandle(
            backend=self.name,
            backend_job_id=str(job.spec.payload.get("kaggle_kernel_id") or job.job_id),
            state=BackendState.UNKNOWN,
            stage="unknown",
        )
        return replace(
            current,
            state=BackendState.CANCELLED,
            stage="polling_cancelled",
            message=(
                "Kaggle CLI has no reliable running-kernel cancellation; "
                "ResearchAgent stopped polling and manual Kaggle review is required"
            ),
            metadata={**current.metadata, "manual_action_required": True},
            updated_at=utc_timestamp(),
        )

    def collect(
        self,
        job: Job,
        handle: BackendHandle,
        destination: Path,
    ) -> CollectedResult:
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        result = self.command_runner(
            [
                self.command,
                "kernels",
                "output",
                handle.backend_job_id,
                "-p",
                str(destination),
                "--force",
            ],
            destination,
            self._environment(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Kaggle output collection failed: "
                + (result.stderr or result.stdout)[-4000:]
            )
        artifacts = tuple(
            path.relative_to(destination).as_posix()
            for path in sorted(destination.rglob("*"))
            if path.is_file() and not path.is_symlink()
        )
        return CollectedResult(
            result=_read_result(destination) or dict(handle.result),
            artifact_paths=artifacts,
            metadata={"kernel_id": handle.backend_job_id, "destination": str(destination)},
        )

    def _metadata_from_job(self, job: Job) -> dict[str, Any]:
        payload = job.spec.payload
        owner = str(payload.get("kaggle_owner") or self.username or "").strip()
        slug = str(payload.get("kaggle_kernel_slug") or job.job_id.lower()).strip()
        if not owner:
            raise ValueError(
                "payload.kaggle_owner is required when kernel-metadata.json is absent"
            )
        code_file = str(payload.get("code_file") or _entrypoint_code_file(job)).strip()
        if not code_file:
            raise ValueError("Kaggle job requires payload.code_file or entrypoint")
        return {
            "id": f"{owner}/{slug}",
            "title": str(payload.get("title") or job.job_id),
            "code_file": code_file,
            "language": "python",
            "kernel_type": "notebook" if code_file.endswith(".ipynb") else "script",
            "is_private": bool(payload.get("is_private", True)),
            "enable_gpu": job.spec.resources.gpu_count > 0,
            "enable_internet": bool(job.spec.resources.network_required),
            "dataset_sources": list(payload.get("dataset_sources") or []),
            "competition_sources": list(payload.get("competition_sources") or []),
            "kernel_sources": list(payload.get("kernel_sources") or []),
        }

    def _environment(self) -> dict[str, str]:
        names = {
            "PATH",
            "HOME",
            "USERPROFILE",
            "LANG",
            "LC_ALL",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "KAGGLE_CONFIG_DIR",
            "KAGGLE_USERNAME",
            "KAGGLE_KEY",
            "KAGGLE_API_TOKEN",
        }
        environment = {
            key: value for key, value in os.environ.items() if key in names
        }
        if self.api_token:
            environment["KAGGLE_API_TOKEN"] = self.api_token
        if self.username:
            environment["KAGGLE_USERNAME"] = self.username
        return environment

    def _default_runner(
        self,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> CommandResult:
        try:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(environment),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CommandResult(127, "", f"{type(exc).__name__}: {exc}")
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class RemoteTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        ...

    def download(
        self,
        url: str,
        destination: Path,
        *,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> None:
        ...


@dataclass(frozen=True)
class RemoteWorkerDescriptor:
    name: str
    base_url: str
    token: str
    capabilities: BackendCapabilities
    paid: bool = False


class UrllibRemoteTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        data = (
            json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers=dict(headers),
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"remote worker HTTP {exc.code}: {detail[-4000:]}") from exc
        value = json.loads(body or "{}")
        if not isinstance(value, dict):
            raise RuntimeError("remote worker returned a non-object response")
        return value

    def download(
        self,
        url: str,
        destination: Path,
        *,
        headers: Mapping[str, str],
        timeout_seconds: int,
    ) -> None:
        request = urllib.request.Request(url, headers=dict(headers))
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output)


class RemoteGpuBackend:
    """Authenticated client for an owned GPU PC or external GPU Worker."""

    def __init__(
        self,
        descriptor: RemoteWorkerDescriptor,
        *,
        transport: RemoteTransport | None = None,
        timeout_seconds: int = 45,
        max_bundle_files: int = 5000,
        max_bundle_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.descriptor = descriptor
        self.name = descriptor.name
        self.capabilities = descriptor.capabilities
        self.approval_required = descriptor.paid
        self.transport = transport or UrllibRemoteTransport()
        self.timeout_seconds = max(2, timeout_seconds)
        self.max_bundle_files = max(1, max_bundle_files)
        self.max_bundle_bytes = max(1, max_bundle_bytes)

    def available(self) -> tuple[bool, str]:
        if not self.descriptor.base_url or not self.descriptor.token:
            return False, "remote worker URL or token is missing"
        try:
            response = self._request("GET", "/health")
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return bool(response.get("ok", True)), str(
            response.get("detail") or self.descriptor.base_url
        )

    def prepare(self, job: Job, workspace: Path) -> dict[str, Any]:
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        _atomic_json(workspace / "job_spec.json", job.spec.to_dict())
        bundle = build_source_bundle(
            workspace,
            max_files=self.max_bundle_files,
            max_bytes=self.max_bundle_bytes,
        )
        return {
            "workspace": str(workspace),
            "bundle": bundle,
        }

    def submit(self, job: Job, workspace: Path) -> BackendHandle:
        prepared = self.prepare(job, workspace)
        response = self._request(
            "POST",
            "/v1/jobs",
            payload={"job": job.to_dict(), "source_bundle": prepared["bundle"]},
        )
        backend_job_id = str(
            response.get("backend_job_id")
            or response.get("job_id")
            or job.job_id
        )
        return BackendHandle(
            backend=self.name,
            backend_job_id=backend_job_id,
            state=_backend_state(response.get("status"), BackendState.QUEUED),
            stage=str(response.get("stage") or "submitted"),
            progress=_progress(response.get("progress"), 0.05),
            message=str(response.get("message") or "Remote job submitted"),
            result=(
                dict(response.get("result"))
                if isinstance(response.get("result"), Mapping)
                else {}
            ),
            error=(str(response["error"]) if response.get("error") else None),
            metadata={
                "workspace": str(workspace),
                "remote": response,
                "bundle_sha256": str(prepared["bundle"].get("sha256") or ""),
            },
        )

    def poll(self, job: Job, handle: BackendHandle) -> BackendHandle:
        response = self._request(
            "GET",
            "/v1/jobs/"
            + urllib.parse.quote(handle.backend_job_id, safe=""),
        )
        return replace(
            handle,
            state=_backend_state(response.get("status"), BackendState.UNKNOWN),
            stage=str(response.get("stage") or "unknown"),
            progress=_progress(response.get("progress"), handle.progress),
            message=str(response.get("message") or ""),
            result=(
                dict(response.get("result"))
                if isinstance(response.get("result"), Mapping)
                else handle.result
            ),
            error=(str(response["error"]) if response.get("error") else None),
            metadata={**handle.metadata, "remote": response},
            updated_at=utc_timestamp(),
        )

    def cancel(self, job: Job, handle: BackendHandle | None) -> BackendHandle:
        current = handle or BackendHandle(
            backend=self.name,
            backend_job_id=job.job_id,
            state=BackendState.UNKNOWN,
            stage="unknown",
        )
        response = self._request(
            "POST",
            "/v1/jobs/"
            + urllib.parse.quote(current.backend_job_id, safe="")
            + "/cancel",
            payload={"reason": "ResearchAgent cancellation requested"},
        )
        return replace(
            current,
            state=_backend_state(response.get("status"), BackendState.CANCELLED),
            stage=str(response.get("stage") or "cancelled"),
            progress=_progress(response.get("progress"), current.progress),
            message=str(response.get("message") or "Cancellation requested"),
            error=(str(response["error"]) if response.get("error") else None),
            metadata={**current.metadata, "remote": response},
            updated_at=utc_timestamp(),
        )

    def collect(
        self,
        job: Job,
        handle: BackendHandle,
        destination: Path,
    ) -> CollectedResult:
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        response = self._request(
            "GET",
            "/v1/jobs/"
            + urllib.parse.quote(handle.backend_job_id, safe="")
            + "/artifacts",
        )
        raw_artifacts = (
            response.get("artifacts")
            if isinstance(response.get("artifacts"), list)
            else []
        )
        collected: list[str] = []
        warnings: list[str] = []
        for item in raw_artifacts:
            if not isinstance(item, Mapping):
                continue
            relative = safe_relative_path(str(item.get("path") or ""))
            if not relative:
                warnings.append("remote artifact with unsafe path was ignored")
                continue
            url = str(item.get("url") or item.get("download_path") or "").strip()
            if not url:
                warnings.append(f"remote artifact has no URL: {relative}")
                continue
            if url.startswith("/"):
                url = self.descriptor.base_url.rstrip("/") + url
            target = (destination / relative).resolve()
            try:
                target.relative_to(destination)
            except ValueError:
                warnings.append(f"remote artifact escapes destination: {relative}")
                continue
            self.transport.download(
                url,
                target,
                headers=self._headers(),
                timeout_seconds=max(60, self.timeout_seconds),
            )
            expected_size = item.get("size_bytes")
            if expected_size is not None and target.stat().st_size != int(expected_size):
                target.unlink(missing_ok=True)
                raise ValueError(f"remote artifact size mismatch: {relative}")
            expected_hash = str(item.get("sha256") or "")
            if expected_hash and _sha256(target) != expected_hash:
                target.unlink(missing_ok=True)
                raise ValueError(f"remote artifact hash mismatch: {relative}")
            collected.append(relative)
        result = (
            dict(response.get("result"))
            if isinstance(response.get("result"), Mapping)
            else dict(handle.result)
        )
        if result and not (destination / "result.json").is_file():
            _atomic_json(destination / "result.json", result)
            collected.append("result.json")
        return CollectedResult(
            result=result,
            artifact_paths=tuple(dict.fromkeys(collected)),
            warnings=tuple(warnings),
            metadata={"response": response, "destination": str(destination)},
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.transport.request(
            method,
            self.descriptor.base_url.rstrip("/") + path,
            payload=payload,
            headers=self._headers(),
            timeout_seconds=self.timeout_seconds,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.descriptor.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }


def detect_local_gpu() -> LocalGpuInventory:
    configured_count = _optional_int(os.getenv("LOCAL_GPU_COUNT"))
    configured_memory = _optional_int(os.getenv("LOCAL_GPU_MEMORY_MB"))
    if configured_count is not None:
        return LocalGpuInventory(
            gpu_count=max(0, configured_count),
            gpu_memory_mb=configured_memory,
            detail=(
                f"configured local GPU count={max(0, configured_count)}"
                + (
                    f", min memory={configured_memory}MB"
                    if configured_memory is not None
                    else ""
                )
            ),
        )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return LocalGpuInventory(detail=f"nvidia-smi unavailable: {exc}")
    if completed.returncode != 0:
        return LocalGpuInventory(
            detail=(completed.stderr or completed.stdout).strip()
            or "nvidia-smi returned an error"
        )
    memories: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            memories.append(int(line.strip()))
        except ValueError:
            continue
    if not memories:
        return LocalGpuInventory(detail="nvidia-smi reported no usable GPUs")
    return LocalGpuInventory(
        gpu_count=len(memories),
        gpu_memory_mb=min(memories),
        detail=f"local NVIDIA GPUs={len(memories)}, min memory={min(memories)}MB",
    )


def _job_command(job: Job) -> list[str]:
    value = job.spec.payload.get("entrypoint")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item for item in value if item]
    text = job_entrypoint(job)
    return shlex.split(text) if text else []


def _validate_command_paths(command: Sequence[str], workspace: Path) -> None:
    for item in command[1:]:
        if not item or item.startswith("-"):
            continue
        candidate = Path(item)
        if candidate.is_absolute():
            continue
        if candidate.suffix not in {".py", ".sh", ".ipynb"} and "/" not in item:
            continue
        resolved = (workspace / candidate).resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise PermissionError(f"entrypoint path escapes workspace: {item}") from exc


def _entrypoint_code_file(job: Job) -> str:
    command = _job_command(job)
    for item in reversed(command):
        if item.endswith((".py", ".ipynb")):
            return safe_relative_path(item)
    return ""


def _handle_workspace(handle: BackendHandle) -> Path:
    value = str(handle.metadata.get("workspace") or "").strip()
    if not value:
        raise ValueError("backend handle does not contain a workspace")
    return Path(value).expanduser().resolve()


def _read_result(root: Path) -> dict[str, Any]:
    path = root / "result.json"
    return _read_json(path) if path.is_file() else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_progress(workspace: Path) -> tuple[float, str]:
    value = _read_json(workspace / "progress.json")
    try:
        progress = min(0.99, max(0.05, float(value.get("progress") or 0.05)))
    except (TypeError, ValueError):
        progress = 0.05
    return progress, str(value.get("stage") or "running")


def _tail(path: Path, limit: int) -> str:
    if not path.is_file():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", errors="replace").strip()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return


def _parse_kaggle_status(text: str) -> tuple[BackendState, float]:
    normalized = text.lower()
    if any(item in normalized for item in ("complete", "completed", "success")):
        return BackendState.SUCCEEDED, 1.0
    if any(item in normalized for item in ("error", "failed", "failure")):
        return BackendState.FAILED, 1.0
    if "running" in normalized:
        return BackendState.RUNNING, 0.5
    if any(item in normalized for item in ("queued", "pending")):
        return BackendState.QUEUED, 0.1
    if any(item in normalized for item in ("cancel", "canceled", "cancelled")):
        return BackendState.CANCELLED, 1.0
    return BackendState.UNKNOWN, 0.0


def _metadata_code_file(metadata: Mapping[str, Any]) -> str:
    return str(
        metadata.get("code_file")
        or metadata.get("notebook_file")
        or metadata.get("script_file")
        or ""
    ).strip()


def _has_kaggle_config() -> bool:
    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    return (config_dir / "kaggle.json").is_file() or bool(
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    )


def _backend_state(value: Any, default: BackendState) -> BackendState:
    normalized = str(value or "").strip().lower()
    aliases = {
        "created": BackendState.QUEUED,
        "submitted": BackendState.QUEUED,
        "preparing": BackendState.QUEUED,
        "collecting": BackendState.RUNNING,
        "completed": BackendState.SUCCEEDED,
        "complete": BackendState.SUCCEEDED,
        "success": BackendState.SUCCEEDED,
        "cancel_requested": BackendState.CANCELLED,
        "canceled": BackendState.CANCELLED,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return BackendState(normalized)
    except ValueError:
        return default


def _progress(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
