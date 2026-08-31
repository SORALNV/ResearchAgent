from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip()
            for item in os.getenv(name, "").split(",")
            if item.strip()
        )
    )


@dataclass(frozen=True)
class ControlPlaneConfig:
    runtime_root: Path
    database_path: Path
    projects_root: Path
    jobs_root: Path
    kaggle_packages_root: Path
    kaggle_outputs_root: Path

    mode: str = "core"
    scheduler_workers: int = 2
    scheduler_queue_size: int = 256
    requeue_interrupted_jobs: bool = False

    enable_fake_backend: bool = False
    enable_local_cpu_backend: bool = True
    local_cpu_max_ram_gb: float = 0.0

    enable_kaggle_backend: bool = True
    kaggle_executable: str = "kaggle"
    kaggle_cli_timeout_seconds: int = 120
    kaggle_poll_interval_seconds: float = 15.0
    kaggle_max_poll_seconds: int = 6 * 60 * 60
    kaggle_max_vram_gb: float = 16.0

    remote_gpu_url: str | None = None
    remote_gpu_token: str | None = None
    remote_gpu_vram_gb: float = 0.0
    remote_gpu_ram_gb: float = 0.0

    gpu_vm_url: str | None = None
    gpu_vm_token: str | None = None
    gpu_vm_vram_gb: float = 0.0
    gpu_vm_ram_gb: float = 0.0

    discord_inbox_channel_id: str | None = None
    discord_work_forum_id: str | None = None
    discord_approval_channel_id: str | None = None
    discord_ops_channel_id: str | None = None
    discord_allowed_user_ids: tuple[str, ...] = ()
    discord_allowed_guild_ids: tuple[str, ...] = ()
    discord_progress_min_interval_seconds: float = 10.0

    @classmethod
    def from_env(
        cls,
        project_root: str | Path | None = None,
    ) -> "ControlPlaneConfig":
        root = Path(project_root or os.getenv("PROJECT_ROOT", ".")).expanduser().resolve()
        runtime_root = Path(
            os.getenv("CONTROL_PLANE_RUNTIME_ROOT", str(root / "runtime"))
        ).expanduser()
        if not runtime_root.is_absolute():
            runtime_root = root / runtime_root
        runtime_root = runtime_root.resolve()

        def path_env(name: str, default: Path) -> Path:
            value = Path(os.getenv(name, str(default))).expanduser()
            if not value.is_absolute():
                value = root / value
            return value.resolve()

        mode = os.getenv("CONTROL_PLANE_MODE", "core").strip().lower()
        if mode not in {"core", "edge", "standalone", "worker"}:
            mode = "core"
        return cls(
            runtime_root=runtime_root,
            database_path=path_env(
                "CONTROL_PLANE_DATABASE",
                runtime_root / "control-plane.sqlite3",
            ),
            projects_root=path_env(
                "CONTROL_PLANE_PROJECTS_ROOT",
                runtime_root / "projects",
            ),
            jobs_root=path_env(
                "CONTROL_PLANE_JOBS_ROOT",
                runtime_root / "jobs",
            ),
            kaggle_packages_root=path_env(
                "KAGGLE_PACKAGES_ROOT",
                runtime_root / "kaggle" / "packages",
            ),
            kaggle_outputs_root=path_env(
                "KAGGLE_OUTPUTS_ROOT",
                runtime_root / "kaggle" / "outputs",
            ),
            mode=mode,
            scheduler_workers=max(1, _int("SCHEDULER_WORKERS", 2)),
            scheduler_queue_size=max(1, _int("SCHEDULER_QUEUE_SIZE", 256)),
            requeue_interrupted_jobs=_bool("REQUEUE_INTERRUPTED_JOBS", False),
            enable_fake_backend=_bool("ENABLE_FAKE_COMPUTE_BACKEND", False),
            enable_local_cpu_backend=_bool("ENABLE_LOCAL_CPU_BACKEND", True),
            local_cpu_max_ram_gb=max(0.0, _float("LOCAL_CPU_MAX_RAM_GB", 0.0)),
            enable_kaggle_backend=_bool("ENABLE_KAGGLE_NOTEBOOK_BACKEND", True),
            kaggle_executable=os.getenv("KAGGLE_EXECUTABLE", "kaggle").strip() or "kaggle",
            kaggle_cli_timeout_seconds=max(1, _int("KAGGLE_CLI_TIMEOUT_SECONDS", 120)),
            kaggle_poll_interval_seconds=max(0.5, _float("KAGGLE_POLL_INTERVAL_SECONDS", 15.0)),
            kaggle_max_poll_seconds=max(1, _int("KAGGLE_MAX_POLL_SECONDS", 6 * 60 * 60)),
            kaggle_max_vram_gb=max(0.0, _float("KAGGLE_MAX_VRAM_GB", 16.0)),
            remote_gpu_url=os.getenv("REMOTE_GPU_WORKER_URL") or None,
            remote_gpu_token=os.getenv("REMOTE_GPU_WORKER_TOKEN") or None,
            remote_gpu_vram_gb=max(0.0, _float("REMOTE_GPU_VRAM_GB", 0.0)),
            remote_gpu_ram_gb=max(0.0, _float("REMOTE_GPU_RAM_GB", 0.0)),
            gpu_vm_url=os.getenv("GPU_VM_WORKER_URL") or None,
            gpu_vm_token=os.getenv("GPU_VM_WORKER_TOKEN") or None,
            gpu_vm_vram_gb=max(0.0, _float("GPU_VM_VRAM_GB", 0.0)),
            gpu_vm_ram_gb=max(0.0, _float("GPU_VM_RAM_GB", 0.0)),
            discord_inbox_channel_id=os.getenv("DISCORD_AGENT_INBOX_CHANNEL_ID") or None,
            discord_work_forum_id=os.getenv("DISCORD_WORK_SESSIONS_FORUM_ID") or None,
            discord_approval_channel_id=os.getenv("DISCORD_APPROVAL_CHANNEL_ID") or None,
            discord_ops_channel_id=os.getenv("DISCORD_AGENT_OPS_CHANNEL_ID") or None,
            discord_allowed_user_ids=_csv("DISCORD_ALLOWED_USER_IDS"),
            discord_allowed_guild_ids=_csv("DISCORD_ALLOWED_GUILD_IDS"),
            discord_progress_min_interval_seconds=max(
                1.0,
                _float("DISCORD_PROGRESS_MIN_INTERVAL_SECONDS", 10.0),
            ),
        )

    def prepare_directories(self) -> None:
        for path in (
            self.runtime_root,
            self.database_path.parent,
            self.projects_root,
            self.jobs_root,
            self.kaggle_packages_root,
            self.kaggle_outputs_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def public_summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "database_path": str(self.database_path),
            "projects_root": str(self.projects_root),
            "jobs_root": str(self.jobs_root),
            "scheduler_workers": self.scheduler_workers,
            "scheduler_queue_size": self.scheduler_queue_size,
            "requeue_interrupted_jobs": self.requeue_interrupted_jobs,
            "local_cpu_backend": self.enable_local_cpu_backend,
            "kaggle_backend": self.enable_kaggle_backend,
            "remote_gpu_backend": bool(self.remote_gpu_url and self.remote_gpu_token),
            "gpu_vm_backend": bool(self.gpu_vm_url and self.gpu_vm_token),
            "discord_threading": bool(
                self.discord_inbox_channel_id and self.discord_work_forum_id
            ),
            "allowed_user_count": len(self.discord_allowed_user_ids),
            "allowed_guild_count": len(self.discord_allowed_guild_ids),
        }
