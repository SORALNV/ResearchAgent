from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from harness.compute_models import job_entrypoint, safe_relative_path
from harness.config import HarnessConfig
from harness.control_plane import Job
from harness.process_manager import ProcessCancellationController
from harness.provider_runtime import ProviderAwareAgentCommandExecutor
from harness.state import ResearchSession, utc_timestamp


class MaterializationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterializationResult:
    job: Job
    workspace: str
    provider: str
    smoke_command: tuple[str, ...]
    smoke_stdout: str
    smoke_stderr: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.to_dict(),
            "workspace": self.workspace,
            "provider": self.provider,
            "smoke_command": list(self.smoke_command),
            "smoke_stdout": self.smoke_stdout,
            "smoke_stderr": self.smoke_stderr,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaterializationResult":
        return cls(
            job=Job.from_dict(data["job"]),
            workspace=str(data["workspace"]),
            provider=str(data.get("provider") or "existing_workspace"),
            smoke_command=tuple(
                str(item) for item in data.get("smoke_command", [])
            ),
            smoke_stdout=str(data.get("smoke_stdout") or ""),
            smoke_stderr=str(data.get("smoke_stderr") or ""),
            generated_at=str(data.get("generated_at") or utc_timestamp()),
        )


class ExperimentMaterializer(Protocol):
    def materialize(self, job: Job, workspace: Path) -> MaterializationResult:
        ...


class ProviderExperimentMaterializer:
    """Prepare a self-contained workspace before remote or long-running work.

    Existing source may be copied from ``payload.source_dir``. When an
    ``implementation_prompt`` is present, the configured workspace-write
    provider (normally Codex CLI) can create or repair the experiment. Every GPU
    experiment must pass an explicit smoke command or a generated ``smoke.py``.
    """

    def __init__(
        self,
        config: HarnessConfig,
        *,
        executor: Any | None = None,
        smoke_timeout_seconds: int | None = None,
    ) -> None:
        self.config = config
        self.cancellation = ProcessCancellationController(
            config.agent_cancel_grace_seconds
        )
        self.executor = executor or ProviderAwareAgentCommandExecutor(
            config,
            threading.RLock(),
            self.cancellation,
        )
        configured = os.getenv("COMPUTE_SMOKE_TIMEOUT_SECONDS")
        try:
            default_timeout = int(configured) if configured else 180
        except ValueError:
            default_timeout = 180
        self.smoke_timeout_seconds = max(
            5,
            smoke_timeout_seconds
            if smoke_timeout_seconds is not None
            else min(config.max_command_seconds, default_timeout),
        )

    def materialize(self, job: Job, workspace: Path) -> MaterializationResult:
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        self._copy_source(job, workspace)
        _atomic_json(workspace / "JOB_SPEC.json", job.spec.to_dict())

        effective = job
        provider = "existing_workspace"
        implementation_prompt = str(
            job.spec.payload.get("implementation_prompt") or ""
        ).strip()
        entrypoint = _command(job.spec.payload.get("entrypoint"))
        if implementation_prompt:
            provider = self._implement(job, workspace, implementation_prompt)

        if not entrypoint:
            entrypoint = _discover_entrypoint(workspace)
        if not entrypoint:
            raise MaterializationError(
                "experiment has no executable entrypoint after materialization"
            )
        _validate_entrypoint(entrypoint, workspace)

        payload = dict(job.spec.payload)
        payload["entrypoint"] = list(entrypoint)
        payload["workspace"] = str(workspace)
        payload["materialized_at"] = utc_timestamp()
        effective = replace(job, spec=replace(job.spec, payload=payload))

        smoke_command = _command(payload.get("smoke_command"))
        if not smoke_command and (workspace / "smoke.py").is_file():
            smoke_command = (sys.executable, "smoke.py")
        if not smoke_command:
            code_file = _python_code_file(entrypoint)
            if code_file:
                smoke_command = (sys.executable, "-m", "py_compile", code_file)
        if effective.spec.resources.gpu_count > 0 and not smoke_command:
            raise MaterializationError(
                "GPU jobs require payload.smoke_command or smoke.py before long execution"
            )

        smoke_stdout = ""
        smoke_stderr = ""
        if smoke_command:
            completed = subprocess.run(
                list(smoke_command),
                cwd=workspace,
                env=_experiment_environment(effective, workspace),
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                timeout=self.smoke_timeout_seconds,
                check=False,
            )
            smoke_stdout = completed.stdout[-8000:]
            smoke_stderr = completed.stderr[-8000:]
            if completed.returncode != 0:
                raise MaterializationError(
                    "smoke test failed: "
                    + (smoke_stderr or smoke_stdout or f"returncode={completed.returncode}")
                )

        result = MaterializationResult(
            job=effective,
            workspace=str(workspace),
            provider=provider,
            smoke_command=smoke_command,
            smoke_stdout=smoke_stdout,
            smoke_stderr=smoke_stderr,
            generated_at=utc_timestamp(),
        )
        _atomic_json(workspace / "MATERIALIZATION.json", result.to_dict())
        return result

    def _copy_source(self, job: Job, workspace: Path) -> None:
        raw = str(job.spec.payload.get("source_dir") or "").strip()
        if not raw:
            return
        source = Path(raw).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        try:
            workspace.relative_to(source)
        except ValueError:
            pass
        else:
            raise MaterializationError(
                "compute workspace must not be nested inside payload.source_dir"
            )
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(source)
            if any(
                part
                in {
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                    ".pytest_cache",
                    "runtime",
                    "control_plane",
                }
                for part in relative.parts
            ):
                continue
            if path.is_symlink():
                raise MaterializationError(f"source_dir contains symlink: {relative}")
            target = (workspace / relative).resolve()
            try:
                target.relative_to(workspace)
            except ValueError as exc:
                raise MaterializationError(
                    f"source path escapes workspace: {relative}"
                ) from exc
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def _implement(
        self,
        job: Job,
        workspace: Path,
        implementation_prompt: str,
    ) -> str:
        session = ResearchSession.new(
            str(job.spec.payload.get("hypothesis") or job.spec.kind),
            project_name=(
                "KaggleAgent"
                if job.spec.domain.value == "kaggle"
                else "ResearchAgent"
            ),
        )
        session.session_id = job.spec.work_session_id
        session.research_dir = str(workspace)
        invocation = self.executor.run(
            session=session,
            role="main",
            stage="compute_materialization",
            prompt=_implementation_prompt(job, implementation_prompt),
            command_text=(
                self.config.main_agent_command
                or self.config.sub_agent_command
                or self.config.review_agent_command
            ),
            sandbox="workspace-write",
            task_id=job.job_id,
            working_dir=workspace,
        )
        if not bool(getattr(invocation, "ok", False)):
            raise MaterializationError(
                "workspace implementation provider failed: "
                + str(
                    getattr(invocation, "stderr", "")
                    or getattr(invocation, "output", "")
                    or "unknown provider failure"
                )[-8000:]
            )
        return _runtime_provider(invocation)


def _implementation_prompt(job: Job, user_prompt: str) -> str:
    return f"""You are the workspace implementation provider for ResearchAgent.
Implement the approved experiment in the current workspace. Do not execute the
long training run. Do not access Discord, OpenAI, Kaggle, or Worker credentials.

Mandatory contract:
- preserve source data; never overwrite raw data
- create a deterministic executable entrypoint
- create smoke.py or honor the requested smoke_command
- the long run must update progress.json with progress in [0,1] and a stage
- the long run must write result.json and metrics.json atomically
- result.json should include summary, metrics, primary_metric, risks, and may
  include next_hypotheses as structured proposals
- record seeds, parameters, dataset identifiers, code/version identifiers, and
  failure details needed for reproduction
- Kaggle mode must preserve the fixed CV definition and must not submit anything
- keep changes confined to this workspace

Approved JobSpec:
{json.dumps(job.spec.to_dict(), ensure_ascii=False, indent=2)}

Implementation request:
{user_prompt}

Finish by checking that the entrypoint and smoke test files exist. Do not claim
that the long experiment was run.
"""


def _command(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    text = str(value or "").strip()
    return tuple(shlex.split(text)) if text else ()


def _discover_entrypoint(workspace: Path) -> tuple[str, ...]:
    for name in ("run.py", "train.py", "experiment.py"):
        if (workspace / name).is_file():
            return (sys.executable, name)
    notebooks = sorted(workspace.glob("*.ipynb"))
    if len(notebooks) == 1:
        return (notebooks[0].name,)
    return ()


def _validate_entrypoint(command: Sequence[str], workspace: Path) -> None:
    for item in command:
        if item.startswith("-"):
            continue
        relative = safe_relative_path(item)
        if not relative or not item.endswith((".py", ".ipynb", ".sh")):
            continue
        path = (workspace / relative).resolve()
        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise MaterializationError(
                f"entrypoint escapes workspace: {item}"
            ) from exc
        if not path.is_file() or path.is_symlink():
            raise MaterializationError(f"entrypoint file not found: {item}")


def _python_code_file(command: Sequence[str]) -> str:
    for item in reversed(command):
        relative = safe_relative_path(item)
        if relative and relative.endswith(".py"):
            return relative
    return ""


def _experiment_environment(job: Job, workspace: Path) -> dict[str, str]:
    allowed = {
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
        key: value for key, value in os.environ.items() if key in allowed
    }
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RESEARCH_AGENT_JOB_ID": job.job_id,
            "RESEARCH_AGENT_PROJECT_ID": job.spec.project_id,
            "RESEARCH_AGENT_WORK_SESSION_ID": job.spec.work_session_id,
            "RESEARCH_AGENT_WORKSPACE": str(workspace),
            "RESEARCH_AGENT_SMOKE_TEST": "1",
        }
    )
    return environment


def _runtime_provider(invocation: Any) -> str:
    command = tuple(getattr(invocation, "command", ()) or ())
    if command and str(command[0]).startswith("provider:"):
        return str(command[0]).split(":", 1)[1]
    return "local_cli" if command else "unknown"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
