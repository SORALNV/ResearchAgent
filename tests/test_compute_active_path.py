from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness.compute_backends import FakeComputeBackend
from harness.compute_bundle import build_source_bundle
from harness.compute_models import BackendCapabilities
from harness.compute_worker import PortableComputeWorker
from harness.config import HarnessConfig
from harness.control_plane import ControlPlaneStore, Domain, JobSpec, ResourceRequirements
from main import build_routed_discord_service


def test_routed_service_disables_core_process_backends_by_default(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("DISCORD_RESEARCH_CHANNEL_IDS", "100")
    monkeypatch.setenv("CONTROL_PLANE_DIR", str(tmp_path / "control-default"))
    monkeypatch.delenv("LOCAL_PROCESS_COMPUTE_ENABLED", raising=False)

    service = build_routed_discord_service(HarnessConfig(project_root=tmp_path))
    assert "local_gpu" not in service.compute.broker.backends
    assert "local_cpu" not in service.compute.broker.backends
    assert "kaggle_notebook" in service.compute.broker.backends


def test_core_process_backends_require_explicit_trusted_opt_in(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("DISCORD_RESEARCH_CHANNEL_IDS", "100")
    monkeypatch.setenv("CONTROL_PLANE_DIR", str(tmp_path / "control-opt-in"))
    monkeypatch.setenv("LOCAL_PROCESS_COMPUTE_ENABLED", "true")

    service = build_routed_discord_service(HarnessConfig(project_root=tmp_path))
    assert "local_gpu" in service.compute.broker.backends
    assert "local_cpu" in service.compute.broker.backends


def test_local_gpu_compose_uses_secret_minimal_worker_sidecar():
    overlay = Path("compose.local-gpu.yaml").read_text(encoding="utf-8")
    assert "local-gpu-worker:" in overlay
    assert "REMOTE_GPU_WORKER_NAME: local_gpu_worker" in overlay
    assert "gpus: all" in overlay
    assert "CODEX_HOME" not in overlay
    assert "OPENAI_API_KEY" not in overlay
    assert "KAGGLE_API_TOKEN" not in overlay
    assert "DISCORD_BOT_TOKEN" not in overlay
    assert "LOCAL_GPU_WORKER_TOKEN" in overlay


def test_discord_adapter_exposes_compute_operational_commands():
    source = Path("harness/routed_discord_adapter.py").read_text(encoding="utf-8")
    assert 'name="compute_backends"' in source
    assert 'name="approve_compute"' in source
    assert 'name="cancel_job"' in source


def test_worker_serializes_duplicate_submit_for_the_same_job(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("print('ok')\n", encoding="utf-8")
    bundle = build_source_bundle(source)

    store = ControlPlaneStore(tmp_path / "control")
    project = store.create_project("worker", Domain.RESEARCH)
    session = store.create_work_session(project.project_id, "thread")
    job = store.create_job(
        JobSpec(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain=Domain.RESEARCH,
            kind="experiment",
            payload={"entrypoint": ["python", "run.py"]},
            resources=ResourceRequirements(accelerator="cpu"),
        )
    )

    worker = PortableComputeWorker(
        data_dir=tmp_path / "worker-runtime",
        capabilities=BackendCapabilities(
            accelerators=("cpu",),
            domains=(Domain.RESEARCH,),
            cpu_cores=4,
        ),
    )

    class CountingBackend(FakeComputeBackend):
        def __init__(self):
            super().__init__(
                name="worker_process",
                capabilities=worker.capabilities,
                complete_after_polls=10,
            )
            self.submit_count = 0

        def submit(self, job, workspace):
            self.submit_count += 1
            return super().submit(job, workspace)

    backend = CountingBackend()
    worker.backend = backend
    body = {"job": job.to_dict(), "source_bundle": bundle}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: worker.submit(body), range(8)))

    assert backend.submit_count == 1
    assert {item["backend_job_id"] for item in results} == {
        f"worker_process-{job.job_id}"
    }
