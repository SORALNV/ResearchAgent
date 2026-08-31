from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from harness.compute_backends import (
    CommandResult,
    FakeComputeBackend,
    KaggleNotebookBackend,
    LocalGpuInventory,
    LocalProcessBackend,
    RemoteGpuBackend,
    RemoteWorkerDescriptor,
)
from harness.compute_bundle import build_source_bundle, extract_source_bundle
from harness.compute_discord import AutonomousRoutedDiscordService
from harness.compute_feedback import (
    ResultFeedbackEngine,
    find_hypothesis_proposal,
)
from harness.compute_materializer import MaterializationResult
from harness.compute_models import (
    BackendCapabilities,
    BackendState,
    ComputeRuntimeRecord,
)
from harness.compute_scheduler import (
    ComputeBroker,
    ComputeRuntimeStore,
    ComputeScheduler,
    ComputeStack,
)
from harness.compute_worker import PortableComputeWorker
from harness.control_plane import (
    ControlPlaneStore,
    Domain,
    EventLane,
    JobSpec,
    JobStatus,
    ResourceRequirements,
)
from harness.discord_thread_router import (
    ChannelDomainMap,
    DiscordChannelDispatcher,
    DiscordLocation,
    DiscordThreadRouter,
)
from harness.human_decision_policy import (
    HumanDecisionKind,
    HumanDecisionVerdict,
)


class PassthroughMaterializer:
    def materialize(self, job, workspace: Path) -> MaterializationResult:
        workspace.mkdir(parents=True, exist_ok=True)
        payload = {**job.spec.payload, "workspace": str(workspace)}
        effective = replace(job, spec=replace(job.spec, payload=payload))
        return MaterializationResult(
            job=effective,
            workspace=str(workspace),
            provider="test",
            smoke_command=(),
            smoke_stdout="",
            smoke_stderr="",
            generated_at="2026-01-01T00:00:00Z",
        )


def _scope(tmp_path: Path, domain: Domain = Domain.RESEARCH):
    store = ControlPlaneStore(tmp_path / "control-plane")
    project = store.create_project("compute", domain)
    session = store.create_work_session(project.project_id, "thread")
    return store, project, session


def _spec(project, session, *, domain=Domain.RESEARCH, backend_preferences=("fake",)):
    return JobSpec(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        domain=domain,
        kind="experiment",
        payload={
            "hypothesis": "test hypothesis",
            "entrypoint": [sys.executable, "run.py"],
            "outputs": ["result.json", "metrics.json"],
        },
        resources=ResourceRequirements(
            cpu_cores=1,
            memory_mb=512,
            gpu_count=0,
            accelerator="cpu",
        ),
        backend_preferences=backend_preferences,
        max_runtime_seconds=60,
    )


def _scheduler(tmp_path: Path, store: ControlPlaneStore, backend):
    root = tmp_path / "compute"
    runtime = ComputeRuntimeStore(root / "state")
    feedback = ResultFeedbackEngine(store, root)
    broker = ComputeBroker(
        [backend],
        research_order=(backend.name,),
        kaggle_order=(backend.name,),
    )
    scheduler = ComputeScheduler(
        store=store,
        broker=broker,
        runtime_store=runtime,
        materializer=PassthroughMaterializer(),
        feedback=feedback,
        root_dir=root,
        poll_interval_seconds=0.01,
        max_unknown_polls=2,
    )
    return scheduler, runtime, feedback, broker


def test_compute_broker_routes_by_domain_resources_and_availability():
    kaggle = FakeComputeBackend(
        name="kaggle_notebook",
        capabilities=BackendCapabilities(
            accelerators=("gpu",),
            domains=(Domain.KAGGLE,),
            gpu_count=1,
            labels=("training",),
        ),
    )
    remote = FakeComputeBackend(
        name="remote_gpu",
        capabilities=BackendCapabilities(
            accelerators=("gpu",),
            domains=(Domain.RESEARCH, Domain.KAGGLE),
            gpu_count=2,
            gpu_memory_mb=48000,
            labels=("training",),
        ),
    )
    local = FakeComputeBackend(
        name="local_cpu",
        capabilities=BackendCapabilities(
            accelerators=("cpu",),
            domains=(Domain.RESEARCH, Domain.KAGGLE),
            cpu_cores=16,
        ),
    )
    broker = ComputeBroker(
        [kaggle, remote, local],
        research_order=("remote_gpu", "local_cpu"),
        kaggle_order=("kaggle_notebook", "remote_gpu", "local_cpu"),
    )
    research_spec = JobSpec(
        project_id="p",
        work_session_id="s",
        domain=Domain.RESEARCH,
        kind="train",
        resources=ResourceRequirements(
            gpu_count=1,
            gpu_memory_mb=24000,
            accelerator="gpu",
            labels=("training",),
        ),
    )
    kaggle_spec = replace(research_spec, domain=Domain.KAGGLE)
    cpu_spec = replace(
        research_spec,
        resources=ResourceRequirements(cpu_cores=2, accelerator="cpu"),
    )

    assert broker.decide(research_spec).selected == "remote_gpu"
    assert broker.decide(kaggle_spec).selected == "kaggle_notebook"
    assert broker.decide(cpu_spec).selected == "local_cpu"


def test_scheduler_executes_collects_and_proposes_next_hypothesis(tmp_path: Path):
    store, project, session = _scope(tmp_path)
    result = {
        "summary": "score improved",
        "metrics": {"score": 0.82},
        "primary_metric": {
            "name": "score",
            "value": 0.82,
            "direction": "maximize",
        },
        "next_hypotheses": [
            {
                "subject_ref": "hypothesis:child-1",
                "title": "single-factor child",
                "hypothesis": "changing one factor improves score",
                "implementation_prompt": "implement the child experiment",
                "resources": {
                    "cpu_cores": 1,
                    "memory_mb": 512,
                    "gpu_count": 0,
                    "accelerator": "cpu",
                },
                "backend_preferences": ["fake"],
                "outputs": ["result.json", "metrics.json"],
            }
        ],
    }
    backend = FakeComputeBackend(result=result, complete_after_polls=2)
    scheduler, runtime_store, _, _ = _scheduler(tmp_path, store, backend)
    job = store.create_job(_spec(project, session))
    try:
        scheduler.start(recover=False)
        scheduler.enqueue(job.job_id)
        scheduler.run_until_idle(timeout_seconds=10)
        completed = store.get_job(job.job_id)
        assert completed.status == JobStatus.SUCCEEDED
        assert completed.backend_id == "fake"
        assert completed.checkpoint_ref.startswith(f"result:{job.job_id}:")
        assert completed.artifact_refs
        runtime = runtime_store.load(job.job_id)
        assert runtime is not None and runtime.collection_complete
        assert runtime.result_ref == completed.checkpoint_ref
        proposal = find_hypothesis_proposal(
            store,
            work_session_id=session.work_session_id,
            subject_ref="hypothesis:child-1",
        )
        assert proposal is not None
        assert proposal.parent_job_id == job.job_id
        assert proposal.parent_result_ref == completed.checkpoint_ref
        event_types = [
            event.event_type
            for event in store.latest_events(
                work_session_id=session.work_session_id,
                limit=100,
            )
        ]
        assert "compute.backend.selected" in event_types
        assert "compute.smoke.passed" in event_types
        assert "experiment.result.collected" in event_types
        assert "experiment.hypothesis.proposed" in event_types
        assert "compute.job.completed" in event_types
    finally:
        scheduler.stop(wait=True)


def test_paid_backend_waits_for_operational_approval(tmp_path: Path):
    store, project, session = _scope(tmp_path)
    backend = FakeComputeBackend(approval_required=True, complete_after_polls=1)
    scheduler, runtime_store, _, _ = _scheduler(tmp_path, store, backend)
    job = store.create_job(_spec(project, session))
    try:
        scheduler.start(recover=False)
        scheduler.enqueue(job.job_id)
        scheduler.run_until_idle(timeout_seconds=5)
        waiting = store.get_job(job.job_id)
        assert waiting.status == JobStatus.WAITING_APPROVAL
        runtime = runtime_store.load(job.job_id)
        assert runtime is not None and runtime.approval_required
        assert runtime.approved is False

        scheduler.approve_job(job.job_id, actor="human:42")
        scheduler.run_until_idle(timeout_seconds=10)
        assert store.get_job(job.job_id).status == JobStatus.SUCCEEDED
    finally:
        scheduler.stop(wait=True)


def test_source_bundle_rejects_traversal_and_round_trips(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("print('ok')\n", encoding="utf-8")
    bundle = build_source_bundle(source)
    destination = tmp_path / "destination"
    result = extract_source_bundle(bundle, destination)
    assert result["file_count"] == 1
    assert (destination / "run.py").read_text(encoding="utf-8") == "print('ok')\n"

    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("../escape.txt", b"no")
    raw = memory.getvalue()
    malicious = {
        "encoding": "base64+zip",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(raw).decode("ascii"),
    }
    with pytest.raises(ValueError, match="unsafe source bundle path"):
        extract_source_bundle(malicious, tmp_path / "unsafe")


def test_local_process_backend_runs_and_collects_result(tmp_path: Path):
    store, project, session = _scope(tmp_path)
    workspace = tmp_path / "local-workspace"
    workspace.mkdir()
    (workspace / "run.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "Path('progress.json').write_text(json.dumps({'progress': 1, 'stage': 'done'}))\n"
        "Path('metrics.json').write_text(json.dumps({'score': 0.9}))\n"
        "Path('result.json').write_text(json.dumps({'summary': 'done', 'metrics': {'score': 0.9}}))\n",
        encoding="utf-8",
    )
    spec = _spec(project, session, backend_preferences=("local_cpu",))
    job = store.create_job(spec)
    backend = LocalProcessBackend(
        name="local_cpu",
        gpu=False,
        inventory=LocalGpuInventory(),
    )
    handle = backend.submit(job, workspace)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        handle = backend.poll(job, handle)
        if handle.terminal:
            break
        time.sleep(0.02)
    assert handle.state == BackendState.SUCCEEDED
    collected = backend.collect(job, handle, tmp_path / "local-artifacts")
    assert collected.result["metrics"]["score"] == 0.9
    assert "result.json" in collected.artifact_paths


def test_kaggle_notebook_backend_pushes_polls_and_collects(tmp_path: Path):
    store, project, session = _scope(tmp_path, Domain.KAGGLE)
    workspace = tmp_path / "kaggle-workspace"
    workspace.mkdir()
    (workspace / "run.py").write_text("print('kaggle')\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def runner(command: Sequence[str], cwd: Path, env: Mapping[str, str]):
        calls.append(tuple(command))
        if "output" in command:
            destination = Path(command[command.index("-p") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "result.json").write_text(
                json.dumps({"summary": "kaggle done", "metrics": {"auc": 0.7}}),
                encoding="utf-8",
            )
        if "status" in command:
            return CommandResult(0, "complete", "")
        return CommandResult(0, "ok", "")

    backend = KaggleNotebookBackend(
        username="owner",
        api_token="token",
        command_runner=runner,
    )
    spec = JobSpec(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        domain=Domain.KAGGLE,
        kind="experiment",
        payload={
            "entrypoint": [sys.executable, "run.py"],
            "code_file": "run.py",
            "kaggle_owner": "owner",
            "kaggle_kernel_slug": "test-kernel",
        },
        resources=ResourceRequirements(gpu_count=1, accelerator="gpu"),
    )
    job = store.create_job(spec)
    handle = backend.submit(job, workspace)
    handle = backend.poll(job, handle)
    assert handle.state == BackendState.SUCCEEDED
    collected = backend.collect(job, handle, tmp_path / "kaggle-artifacts")
    assert collected.result["metrics"]["auc"] == 0.7
    assert any("push" in call for call in calls)
    assert any("status" in call for call in calls)
    assert any("output" in call for call in calls)


class FakeRemoteTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Mapping[str, Any] | None]] = []

    def request(self, method, url, *, payload, headers, timeout_seconds):
        self.requests.append((method, url, payload))
        if url.endswith("/health"):
            return {"ok": True, "detail": "worker"}
        if method == "POST" and url.endswith("/v1/jobs"):
            assert payload and payload.get("source_bundle")
            return {
                "job_id": "remote-1",
                "backend_job_id": "remote-1",
                "status": "running",
                "stage": "training",
                "progress": 0.2,
            }
        if url.endswith("/artifacts"):
            data = b'{"summary":"remote done","metrics":{"score":0.88}}'
            return {
                "result": {"summary": "remote done", "metrics": {"score": 0.88}},
                "artifacts": [
                    {
                        "path": "result.json",
                        "url": "/download/result.json",
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ],
            }
        return {
            "status": "succeeded",
            "stage": "completed",
            "progress": 1.0,
            "result": {"summary": "remote done", "metrics": {"score": 0.88}},
        }

    def download(self, url, destination, *, headers, timeout_seconds):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            b'{"summary":"remote done","metrics":{"score":0.88}}'
        )


def test_remote_gpu_backend_bundles_source_and_verifies_artifacts(tmp_path: Path):
    store, project, session = _scope(tmp_path)
    workspace = tmp_path / "remote-workspace"
    workspace.mkdir()
    (workspace / "run.py").write_text("print('remote')\n", encoding="utf-8")
    job = store.create_job(
        JobSpec(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain=Domain.RESEARCH,
            kind="experiment",
            payload={"entrypoint": [sys.executable, "run.py"]},
            resources=ResourceRequirements(gpu_count=1, accelerator="gpu"),
        )
    )
    transport = FakeRemoteTransport()
    backend = RemoteGpuBackend(
        RemoteWorkerDescriptor(
            name="remote_gpu",
            base_url="https://worker.example",
            token="secret",
            capabilities=BackendCapabilities(
                accelerators=("gpu",),
                domains=(Domain.RESEARCH,),
                gpu_count=1,
            ),
        ),
        transport=transport,
    )
    assert backend.available()[0]
    handle = backend.submit(job, workspace)
    handle = backend.poll(job, handle)
    assert handle.state == BackendState.SUCCEEDED
    collected = backend.collect(job, handle, tmp_path / "remote-artifacts")
    assert collected.result["metrics"]["score"] == 0.88
    assert (tmp_path / "remote-artifacts" / "result.json").is_file()


def test_portable_worker_executes_uploaded_bundle(tmp_path: Path):
    source = tmp_path / "worker-source"
    source.mkdir()
    (source / "run.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "Path('result.json').write_text(json.dumps({'summary': 'worker', 'metrics': {'x': 1}}))\n",
        encoding="utf-8",
    )
    store, project, session = _scope(tmp_path / "scope")
    job = store.create_job(
        JobSpec(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain=Domain.RESEARCH,
            kind="experiment",
            payload={
                "entrypoint": [sys.executable, "run.py"],
                "outputs": ["result.json"],
            },
            resources=ResourceRequirements(accelerator="cpu"),
        )
    )
    worker = PortableComputeWorker(
        data_dir=tmp_path / "worker-runtime",
        capabilities=BackendCapabilities(
            accelerators=("cpu",),
            domains=(Domain.RESEARCH,),
            cpu_cores=4,
            labels=(),
        ),
    )
    submitted = worker.submit(
        {"job": job.to_dict(), "source_bundle": build_source_bundle(source)}
    )
    assert submitted["status"] in {"running", "queued"}
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = worker.status(job.job_id)
        if status["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.02)
    assert status["status"] == "succeeded"
    collected = worker.collect(job.job_id)
    assert collected["result"]["metrics"]["x"] == 1
    assert any(item["path"] == "result.json" for item in collected["artifacts"])


def test_accepted_hypothesis_creates_job_and_child_requires_interpretation(
    tmp_path: Path,
):
    store = ControlPlaneStore(tmp_path / "control-plane")
    router = DiscordThreadRouter(
        store,
        ChannelDomainMap({"100": Domain.RESEARCH}),
    )
    dispatcher = DiscordChannelDispatcher(
        router,
        {
            Domain.RESEARCH: lambda ingress: "research",
            Domain.KAGGLE: lambda ingress: "kaggle",
        },
    )
    result = {
        "summary": "parent done",
        "metrics": {"score": 0.8},
        "next_hypotheses": [
            {
                "subject_ref": "hypothesis:child",
                "title": "child",
                "hypothesis": "child improves score",
                "resources": {"gpu_count": 0, "accelerator": "cpu"},
                "backend_preferences": ["fake"],
            }
        ],
    }
    backend = FakeComputeBackend(result=result, complete_after_polls=1)
    scheduler, runtime, feedback, broker = _scheduler(tmp_path, store, backend)
    service = AutonomousRoutedDiscordService(
        router,
        dispatcher,
        ComputeStack(broker, scheduler, feedback, runtime),
    )
    location = DiscordLocation(guild_id="1", channel_id="100")
    route = router.resolve_work_session(location, title="Research")
    store.append_event(
        event_type=ResultFeedbackEngine.PROPOSAL_EVENT,
        lane=EventLane.DATA,
        project_id=route.project.project_id,
        work_session_id=route.work_session.work_session_id,
        actor="agent:test",
        payload={
            "proposal": {
                "subject_ref": "hypothesis:parent",
                "domain": "research",
                "title": "parent",
                "hypothesis": "run parent",
                "resources": {"gpu_count": 0, "accelerator": "cpu"},
                "backend_preferences": ["fake"],
                "outputs": ["result.json"],
            }
        },
    )
    try:
        service.start()
        decision = service.record_decision(
            location,
            title="Research",
            kind=HumanDecisionKind.HYPOTHESIS,
            verdict=HumanDecisionVerdict.ACCEPT,
            subject_ref="hypothesis:parent",
            note="try it",
            actor_id="42",
            message_id="1000",
            actor_is_human=True,
        )
        scheduler.run_until_idle(timeout_seconds=10)
        jobs = store.list_jobs(work_session_id=route.work_session.work_session_id)
        assert len(jobs) == 1
        assert jobs[0].status == JobStatus.SUCCEEDED
        child = find_hypothesis_proposal(
            store,
            work_session_id=route.work_session.work_session_id,
            subject_ref="hypothesis:child",
        )
        assert child is not None and child.parent_result_ref

        with pytest.raises(PermissionError, match="result interpretation"):
            service.record_decision(
                location,
                title="Research",
                kind=HumanDecisionKind.HYPOTHESIS,
                verdict=HumanDecisionVerdict.ACCEPT,
                subject_ref="hypothesis:child",
                note="try child",
                actor_id="42",
                message_id="1001",
                actor_is_human=True,
            )
        service.record_decision(
            location,
            title="Research",
            kind=HumanDecisionKind.RESULT_INTERPRETATION,
            verdict=HumanDecisionVerdict.ACCEPT,
            subject_ref=child.parent_result_ref,
            note="interpretation confirmed",
            actor_id="42",
            message_id="1002",
            actor_is_human=True,
        )
        service.record_decision(
            location,
            title="Research",
            kind=HumanDecisionKind.HYPOTHESIS,
            verdict=HumanDecisionVerdict.ACCEPT,
            subject_ref="hypothesis:child",
            note="try child",
            actor_id="42",
            message_id="1003",
            actor_is_human=True,
        )
        scheduler.run_until_idle(timeout_seconds=10)
        jobs = store.list_jobs(work_session_id=route.work_session.work_session_id)
        assert len(jobs) == 2
        assert all(job.status == JobStatus.SUCCEEDED for job in jobs)
        assert decision.work_session_id == route.work_session.work_session_id
    finally:
        service.stop(wait=True)
