from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from harness.compute.base import (
    BackendCapabilities,
    BackendStatus,
    ComputeHandle,
)
from harness.platform.models import Domain, JobRecord


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Path, Mapping[str, str]], CommandResult]


class KaggleNotebookBackend:
    """Treat Kaggle Notebooks/Kernels as a remote compute backend.

    Kaggle credentials live only in ResearchAgent Core. They are not forwarded
    to Codex/OpenAI runtimes. A job workspace must contain kernel-metadata.json
    and the notebook/script referenced by that metadata.
    """

    name = "kaggle_notebook"
    capabilities = BackendCapabilities(
        accelerators=("cpu", "gpu"),
        max_vram_gb=None,
        max_gpu_count=None,
        max_cpu_cores=None,
        max_ram_gb=None,
        max_runtime_minutes=None,
        network_available=True,
        detailed_progress=False,
        supports_cancel=False,
        supports_kaggle_data=True,
        domains=(Domain.KAGGLE,),
        tags=("notebook", "training", "inference", "kaggle_data"),
    )

    def __init__(
        self,
        *,
        command: str = "kaggle",
        api_token: str | None = None,
        username: str | None = None,
        command_runner: CommandRunner | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.command = command
        self.api_token = api_token
        self.username = username
        self.command_runner = command_runner or self._default_runner
        self.timeout_seconds = max(10, timeout_seconds)

    def available(self) -> tuple[bool, str]:
        if not self.command.strip():
            return False, "Kaggle command is empty"
        if not self.api_token and not _has_kaggle_config():
            return False, "KAGGLE_API_TOKEN or Kaggle config is not available"
        return True, self.command

    def prepare(self, job: JobRecord, workspace: Path) -> dict[str, Any]:
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        metadata_path = workspace / "kernel-metadata.json"
        if not metadata_path.is_file():
            generated = self._metadata_from_job(job)
            metadata_path.write_text(
                json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid kernel-metadata.json: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("kernel-metadata.json must be an object")
        kernel_id = str(metadata.get("id") or "").strip()
        if "/" not in kernel_id:
            raise ValueError("kernel metadata id must be owner/slug")
        code_file = _metadata_code_file(metadata)
        if code_file and not (workspace / code_file).is_file():
            raise FileNotFoundError(f"Kaggle code file not found: {code_file}")
        return {
            "backend": self.name,
            "kernel_id": kernel_id,
            "metadata_path": str(metadata_path),
            "workspace": str(workspace),
            "code_file": code_file,
        }

    def submit(self, job: JobRecord, workspace: Path) -> ComputeHandle:
        prepared = self.prepare(job, workspace)
        result = self.command_runner(
            [self.command, "kernels", "push", "-p", str(workspace.resolve())],
            workspace,
            self._environment(),
        )
        if result.returncode != 0:
            return ComputeHandle(
                backend=self.name,
                backend_job_id=prepared["kernel_id"],
                status=BackendStatus.FAILED,
                stage="submit_failed",
                progress=1.0,
                message="Kaggle kernel push failed",
                error=(result.stderr or result.stdout)[-4000:],
                metadata=prepared,
            )
        return ComputeHandle(
            backend=self.name,
            backend_job_id=prepared["kernel_id"],
            status=BackendStatus.SUBMITTED,
            stage="submitted",
            progress=0.05,
            message=result.stdout.strip() or "Kaggle kernel submitted",
            metadata=prepared,
        )

    def poll(self, job: JobRecord) -> ComputeHandle:
        kernel_id = _backend_job_id(job)
        workspace = _job_workspace(job)
        result = self.command_runner(
            [self.command, "kernels", "status", kernel_id],
            workspace,
            self._environment(),
        )
        if result.returncode != 0:
            return ComputeHandle(
                backend=self.name,
                backend_job_id=kernel_id,
                status=BackendStatus.UNKNOWN,
                stage="status_error",
                progress=job.progress,
                message="Unable to read Kaggle kernel status",
                error=(result.stderr or result.stdout)[-4000:],
            )
        status, progress = _parse_kaggle_status(result.stdout + "\n" + result.stderr)
        return ComputeHandle(
            backend=self.name,
            backend_job_id=kernel_id,
            status=status,
            stage=status.value,
            progress=progress,
            message=(result.stdout or result.stderr).strip(),
        )

    def cancel(self, job: JobRecord) -> ComputeHandle:
        return ComputeHandle(
            backend=self.name,
            backend_job_id=_backend_job_id(job),
            status=BackendStatus.CANCEL_REQUESTED,
            stage="cancel_not_supported",
            progress=job.progress,
            message=(
                "Kaggle CLI does not expose reliable running-kernel cancellation. "
                "The local scheduler stopped polling and requires manual Kaggle review."
            ),
            metadata={"manual_action_required": True},
        )

    def collect(self, job: JobRecord, destination: Path) -> ComputeHandle:
        kernel_id = _backend_job_id(job)
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        result = self.command_runner(
            [
                self.command,
                "kernels",
                "output",
                kernel_id,
                "-p",
                str(destination),
                "--force",
            ],
            destination,
            self._environment(),
        )
        if result.returncode != 0:
            return ComputeHandle(
                backend=self.name,
                backend_job_id=kernel_id,
                status=BackendStatus.FAILED,
                stage="collect_failed",
                progress=1.0,
                message="Kaggle output collection failed",
                error=(result.stderr or result.stdout)[-4000:],
                metadata={"destination": str(destination)},
            )
        result_path = destination / "result.json"
        structured: dict[str, Any] = {}
        if result_path.is_file():
            try:
                value = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    structured = value
            except (OSError, json.JSONDecodeError):
                structured = {}
        return ComputeHandle(
            backend=self.name,
            backend_job_id=kernel_id,
            status=BackendStatus.COMPLETED,
            stage="collected",
            progress=1.0,
            message=result.stdout.strip() or "Kaggle output collected",
            result=structured,
            metadata={"destination": str(destination)},
        )

    def _metadata_from_job(self, job: JobRecord) -> dict[str, Any]:
        metadata = dict(job.spec.metadata)
        owner = str(metadata.get("kaggle_owner") or self.username or "").strip()
        slug = str(metadata.get("kaggle_kernel_slug") or job.spec.job_id.lower()).strip()
        if not owner:
            raise ValueError(
                "kaggle_owner is required when kernel-metadata.json is not provided"
            )
        code_file = job.spec.entrypoint or "run.py"
        is_notebook = code_file.endswith(".ipynb")
        return {
            "id": f"{owner}/{slug}",
            "title": str(metadata.get("title") or job.spec.job_id),
            "code_file": code_file,
            "language": "python",
            "kernel_type": "notebook" if is_notebook else "script",
            "is_private": bool(metadata.get("is_private", True)),
            "enable_gpu": job.spec.resources.accelerator == "gpu",
            "enable_internet": bool(job.spec.resources.network_required),
            "dataset_sources": list(metadata.get("dataset_sources") or []),
            "competition_sources": list(metadata.get("competition_sources") or []),
            "kernel_sources": list(metadata.get("kernel_sources") or []),
        }

    def _environment(self) -> dict[str, str]:
        allowed = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
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
        }
        if self.api_token:
            allowed["KAGGLE_API_TOKEN"] = self.api_token
        if self.username:
            allowed["KAGGLE_USERNAME"] = self.username
        return allowed

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
        return CommandResult(
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )


def _parse_kaggle_status(text: str) -> tuple[BackendStatus, float]:
    normalized = text.lower()
    if any(item in normalized for item in ("complete", "completed", "success")):
        return BackendStatus.COMPLETED, 1.0
    if any(item in normalized for item in ("error", "failed", "failure")):
        return BackendStatus.FAILED, 1.0
    if "running" in normalized:
        return BackendStatus.RUNNING, 0.5
    if any(item in normalized for item in ("queued", "pending")):
        return BackendStatus.QUEUED, 0.1
    if any(item in normalized for item in ("cancel", "canceled", "cancelled")):
        return BackendStatus.CANCELLED, 1.0
    return BackendStatus.UNKNOWN, 0.0


def _metadata_code_file(metadata: Mapping[str, Any]) -> str:
    return str(
        metadata.get("code_file")
        or metadata.get("notebook_file")
        or metadata.get("script_file")
        or ""
    ).strip()


def _backend_job_id(job: JobRecord) -> str:
    value = job.backend_job_id or job.spec.metadata.get("kaggle_kernel_id")
    if not value:
        raise ValueError(f"Kaggle backend_job_id is missing for {job.spec.job_id}")
    return str(value)


def _job_workspace(job: JobRecord) -> Path:
    value = job.spec.metadata.get("workspace")
    return Path(str(value)).expanduser().resolve() if value else Path.cwd()


def _has_kaggle_config() -> bool:
    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    return (config_dir / "kaggle.json").is_file() or bool(
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    )
