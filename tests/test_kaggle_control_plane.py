import json
from pathlib import Path

import pytest

from harness.compute import ComputeBroker
from harness.control_plane import ControlPlaneRegistry, JobSpec
from harness.job_scheduler import JobScheduler
from harness.kaggle_compute import (
    FakeKaggleTransport,
    KaggleNotebookBackend,
    KaggleNotebookPackageBuilder,
)
from harness.kaggle_domain import KaggleStore
from harness.kaggle_gateway import (
    FakeKaggleSubmissionTransport,
    KaggleSubmissionGateway,
)
from harness.kaggle_service import KaggleApplicationService, parse_competition_slug
from harness.kaggle_validation import validate_submission, verify_submission_hash
from harness.work_sessions import WorkSessionService, WorkSessionStore


def make_services(tmp_path, backend=None):
    database = tmp_path / "control.sqlite3"
    registry = ControlPlaneRegistry(database)
    messages = WorkSessionStore(database)
    kaggle_store = KaggleStore(database)
    backend = backend or KaggleNotebookBackend(
        FakeKaggleTransport(
            outputs={
                "result.json": json.dumps({"status": "success"}),
                "metrics.json": json.dumps({"cv_mean": 0.82}),
            }
        ),
        package_root=tmp_path / "packages",
        output_root=tmp_path / "outputs",
        poll_interval_seconds=0.01,
        max_poll_seconds=5,
    )
    scheduler = JobScheduler(
        registry,
        ComputeBroker([backend]),
        worker_count=1,
        queue_size=8,
    )
    sessions = WorkSessionService(registry, messages, scheduler)
    submit_transport = FakeKaggleSubmissionTransport()
    gateway = KaggleSubmissionGateway(kaggle_store, submit_transport)
    kaggle = KaggleApplicationService(
        registry=registry,
        work_sessions=sessions,
        scheduler=scheduler,
        kaggle_store=kaggle_store,
        submission_gateway=gateway,
    )
    return registry, messages, kaggle_store, scheduler, sessions, kaggle, submit_transport


def test_kaggle_package_builder_excludes_secrets_and_rejects_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (source / ".env").write_text("SECRET=x\n", encoding="utf-8")
    spec = JobSpec.new(
        project_id="PRJ-X",
        work_session_id="WS-X",
        domain="kaggle",
        task_type="train",
        payload={
            "source_dir": str(source),
            "code_file": "run.py",
            "kernel_ref": "owner/research-agent-test",
            "competition_sources": ["sample-comp"],
        },
        resources={"accelerator": "gpu"},
    )
    package, kernel_ref = KaggleNotebookPackageBuilder(tmp_path / "packages").build(spec)
    assert kernel_ref == "owner/research-agent-test"
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True
    assert metadata["competition_sources"] == ["sample-comp"]
    assert (package / "run.py").exists()
    assert not (package / ".env").exists()

    if hasattr(Path, "symlink_to"):
        source2 = tmp_path / "source2"
        source2.mkdir()
        (source2 / "run.py").write_text("print('ok')\n", encoding="utf-8")
        try:
            (source2 / "link.py").symlink_to(source2 / "run.py")
        except OSError:
            return
        spec2 = JobSpec.new(
            project_id="PRJ-X",
            work_session_id="WS-X",
            domain="kaggle",
            task_type="train",
            payload={
                "source_dir": str(source2),
                "code_file": "run.py",
                "kernel_ref": "owner/symlink-test",
            },
            resources={"accelerator": "gpu"},
        )
        with pytest.raises(ValueError, match="symlink"):
            KaggleNotebookPackageBuilder(tmp_path / "packages2").build(spec2)


def test_kaggle_notebook_backend_pushes_polls_and_collects(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("print('ok')\n", encoding="utf-8")
    transport = FakeKaggleTransport(
        statuses=["queued", "running", "complete"],
        outputs={
            "result.json": json.dumps({"status": "success"}),
            "metrics.json": json.dumps({"cv_mean": 0.8123}),
        },
    )
    backend = KaggleNotebookBackend(
        transport,
        package_root=tmp_path / "packages",
        output_root=tmp_path / "outputs",
        poll_interval_seconds=0.01,
        max_poll_seconds=5,
    )
    spec = JobSpec.new(
        project_id="PRJ-X",
        work_session_id="WS-X",
        domain="kaggle",
        task_type="train_cv",
        payload={
            "source_dir": str(source),
            "code_file": "run.py",
            "kernel_ref": "owner/backend-test",
        },
        resources={"accelerator": "gpu"},
        outputs=("result.json", "metrics.json"),
    )
    events = []
    import threading

    result = backend.run(
        spec,
        emit=lambda kind, payload: events.append((kind, payload)),
        cancel_event=threading.Event(),
    )
    assert result.status == "completed"
    assert result.result["metrics"]["cv_mean"] == 0.8123
    assert transport.status_calls == 3
    assert any(kind == "kaggle_package_ready" for kind, _ in events)
    assert any(kind == "kaggle_output_finished" for kind, _ in events)


def test_submission_validation_and_exact_hash_approval(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    candidate = tmp_path / "submission.csv"
    sample.write_text("id,prediction\n1,0\n2,0\n", encoding="utf-8")
    candidate.write_text("id,prediction\n1,0.1\n2,0.9\n", encoding="utf-8")
    report = validate_submission(
        candidate,
        sample,
        id_column="id",
        prediction_ranges={"prediction": (0.0, 1.0)},
    )
    assert report.valid
    assert verify_submission_hash(candidate, report.sha256).ok

    registry, _, store, scheduler, sessions, kaggle, transport = make_services(tmp_path)
    try:
        competition, root_session = kaggle.create_competition(
            "https://www.kaggle.com/competitions/test-comp",
            title="Test Comp",
            project_root=tmp_path / "project",
            metric="auc",
            id_column="id",
        )
        kaggle.confirm_rules(competition.project_id, confirmed_by="sora")
        store.update_competition(
            competition.project_id,
            sample_submission_path=str(sample),
            dataset_fingerprint="abc123",
        )
        experiment = store.create_experiment(
            project_id=competition.project_id,
            work_session_id=root_session.work_session_id,
            hypothesis="baseline",
            experiment_id="EXP-TEST-0001",
        )
        candidate_record, candidate_report = kaggle.prepare_submission(
            experiment.experiment_id,
            submission_path=candidate,
            message="baseline",
            prediction_ranges={"prediction": (0.0, 1.0)},
            cv_score=0.8,
        )
        assert candidate_report.valid
        with pytest.raises(PermissionError):
            kaggle.submit_candidate(candidate_record.candidate_id)
        approved = kaggle.approve_submission(
            candidate_record.candidate_id,
            "DISCORD-123",
        )
        assert approved.status == "approved"
        submitted = kaggle.submit_candidate(candidate_record.candidate_id)
        assert submitted.status == "submitted"
        assert len(transport.calls) == 1

        candidate.write_text("id,prediction\n1,0.2\n2,0.8\n", encoding="utf-8")
        assert not verify_submission_hash(candidate, report.sha256).ok
    finally:
        scheduler.close(cancel_running=True)


def test_competition_slug_parser():
    assert parse_competition_slug("https://www.kaggle.com/competitions/foo-bar/") == "foo-bar"
    assert parse_competition_slug("foo_bar") == "foo_bar"
    with pytest.raises(ValueError):
        parse_competition_slug("https://example.com/nope")
