from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.compute import BackendCapabilities, ComputeBackend, ComputeBroker, FakeComputeBackend
from harness.control_plane import ControlPlaneRegistry
from harness.control_plane_config import ControlPlaneConfig
from harness.job_scheduler import JobScheduler
from harness.kaggle_compute import KaggleCliTransport, KaggleNotebookBackend
from harness.kaggle_domain import KaggleStore
from harness.kaggle_gateway import (
    KaggleCliSubmissionTransport,
    KaggleSubmissionGateway,
)
from harness.kaggle_service import KaggleApplicationService
from harness.local_compute import LocalProcessBackend
from harness.remote_compute import HttpRemoteWorkerClient, RemoteComputeBackend
from harness.work_sessions import WorkSessionService, WorkSessionStore


@dataclass
class ApplicationRuntime:
    config: ControlPlaneConfig
    registry: ControlPlaneRegistry
    work_session_store: WorkSessionStore
    kaggle_store: KaggleStore
    broker: ComputeBroker
    scheduler: JobScheduler
    work_sessions: WorkSessionService
    kaggle: KaggleApplicationService
    submission_gateway: KaggleSubmissionGateway | None

    @classmethod
    def build(
        cls,
        config: ControlPlaneConfig,
        *,
        extra_backends: list[ComputeBackend] | None = None,
        start_scheduler: bool = True,
    ) -> "ApplicationRuntime":
        config.prepare_directories()
        registry = ControlPlaneRegistry(config.database_path)
        work_store = WorkSessionStore(config.database_path)
        kaggle_store = KaggleStore(config.database_path)
        backends: list[ComputeBackend] = []

        if config.enable_local_cpu_backend:
            backends.append(
                LocalProcessBackend(
                    workspace_root=config.jobs_root / "local",
                    max_ram_gb=config.local_cpu_max_ram_gb,
                )
            )

        kaggle_transport = None
        if config.enable_kaggle_backend and shutil.which(config.kaggle_executable):
            kaggle_transport = KaggleCliTransport(
                executable=config.kaggle_executable,
                timeout_seconds=config.kaggle_cli_timeout_seconds,
            )
            backends.append(
                KaggleNotebookBackend(
                    kaggle_transport,
                    package_root=config.kaggle_packages_root,
                    output_root=config.kaggle_outputs_root,
                    poll_interval_seconds=config.kaggle_poll_interval_seconds,
                    max_poll_seconds=config.kaggle_max_poll_seconds,
                    max_vram_gb=config.kaggle_max_vram_gb,
                )
            )

        if config.remote_gpu_url and config.remote_gpu_token:
            backends.append(
                RemoteComputeBackend(
                    "remote_gpu",
                    HttpRemoteWorkerClient(
                        config.remote_gpu_url,
                        token=config.remote_gpu_token,
                    ),
                    BackendCapabilities(
                        name="remote_gpu",
                        domains=("research", "kaggle"),
                        accelerators=("cpu", "gpu"),
                        max_vram_gb=config.remote_gpu_vram_gb,
                        max_ram_gb=config.remote_gpu_ram_gb,
                        supports_cancel=True,
                        supports_live_events=True,
                        supports_network=True,
                        metadata={"remote": True, "kind": "owned_gpu"},
                    ),
                )
            )

        if config.gpu_vm_url and config.gpu_vm_token:
            backends.append(
                RemoteComputeBackend(
                    "gpu_vm",
                    HttpRemoteWorkerClient(
                        config.gpu_vm_url,
                        token=config.gpu_vm_token,
                    ),
                    BackendCapabilities(
                        name="gpu_vm",
                        domains=("research", "kaggle"),
                        accelerators=("cpu", "gpu"),
                        max_vram_gb=config.gpu_vm_vram_gb,
                        max_ram_gb=config.gpu_vm_ram_gb,
                        supports_cancel=True,
                        supports_live_events=True,
                        supports_network=True,
                        metadata={"remote": True, "kind": "rented_gpu"},
                    ),
                )
            )

        if config.enable_fake_backend:
            backends.append(FakeComputeBackend())
        backends.extend(extra_backends or [])
        if not backends:
            # The control plane can still start without credentials or compute.
            # Jobs will fail closed at backend selection rather than at boot.
            backends.append(
                FakeComputeBackend(
                    name="unavailable",
                    accelerators=(),
                    domains=(),
                )
            )

        broker = ComputeBroker(backends)
        scheduler = JobScheduler(
            registry,
            broker,
            worker_count=config.scheduler_workers,
            queue_size=config.scheduler_queue_size,
            requeue_interrupted=config.requeue_interrupted_jobs,
        )
        work_sessions = WorkSessionService(registry, work_store, scheduler)

        submission_gateway = None
        if shutil.which(config.kaggle_executable):
            submission_gateway = KaggleSubmissionGateway(
                kaggle_store,
                KaggleCliSubmissionTransport(
                    executable=config.kaggle_executable,
                    timeout_seconds=config.kaggle_cli_timeout_seconds,
                ),
            )
        kaggle = KaggleApplicationService(
            registry=registry,
            work_sessions=work_sessions,
            scheduler=scheduler,
            kaggle_store=kaggle_store,
            submission_gateway=submission_gateway,
        )
        runtime = cls(
            config=config,
            registry=registry,
            work_session_store=work_store,
            kaggle_store=kaggle_store,
            broker=broker,
            scheduler=scheduler,
            work_sessions=work_sessions,
            kaggle=kaggle,
            submission_gateway=submission_gateway,
        )
        if start_scheduler:
            scheduler.start()
        return runtime

    def close(self, *, cancel_running: bool = False) -> None:
        self.scheduler.close(cancel_running=cancel_running)

    def status(self) -> dict[str, Any]:
        return {
            "config": self.config.public_summary(),
            "scheduler": self.scheduler.snapshot(),
            "backends": self.broker.status(),
        }

    def project_path(self, project_id: str) -> Path:
        project = self.registry.get_project(project_id)
        if project is None:
            raise KeyError(project_id)
        return Path(project.root_dir)
