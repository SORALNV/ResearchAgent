from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from harness.compute_feedback import ResultFeedbackEngine
from harness.config import HarnessConfig
from harness.control_plane import (
    ControlPlaneStore,
    Domain,
    EventLane,
    JobSpec,
)
from harness.discord_thread_router import (
    ChannelDomainMap,
    DiscordLocation,
    DiscordThreadRouter,
    HumanDecisionKind,
    HumanDecisionVerdict,
)
from harness.final_actions import FinalActionCoordinator, FinalActionState
from harness.kaggle_submission import (
    KaggleCliTransport,
    KaggleCommandResult,
    KaggleSubmissionPipeline,
    SubmissionBlockedError,
    SubmissionState,
)
from harness.paper_pipeline import PaperGenerationPipeline, PaperPipelineBlockedError
from harness.paper_search import FakePaperSearchProvider


def _router(tmp_path: Path) -> DiscordThreadRouter:
    return DiscordThreadRouter(
        ControlPlaneStore(tmp_path / "control-plane"),
        ChannelDomainMap({"100": Domain.RESEARCH, "200": Domain.KAGGLE}),
    )


def _route(router: DiscordThreadRouter, domain: Domain):
    if domain == Domain.KAGGLE:
        location = DiscordLocation(
            guild_id="1",
            channel_id="201",
            parent_channel_id="200",
            thread_id="201",
        )
        title = "Kaggle experiment"
    else:
        location = DiscordLocation(
            guild_id="1",
            channel_id="101",
            parent_channel_id="100",
            thread_id="101",
        )
        title = "Research experiment"
    return router.resolve_work_session(location, title=title)


def _result_event(
    router: DiscordThreadRouter,
    route,
    artifact_root: Path,
    *,
    result: Mapping[str, object],
    suffix: str,
):
    job = router.store.create_job(
        JobSpec(
            project_id=route.project.project_id,
            work_session_id=route.work_session.work_session_id,
            domain=route.domain,
            kind="experiment",
            payload={
                "title": f"Experiment {suffix}",
                "hypothesis": f"Hypothesis {suffix}",
                "competition_slug": "demo-competition",
            },
            experiment_id=f"experiment:{suffix}",
        ),
        job_id=f"JOB-{suffix}",
    )
    result_ref = f"result:{job.job_id}:{suffix.lower()}"
    event = router.store.append_event(
        event_type=ResultFeedbackEngine.RESULT_EVENT,
        lane=EventLane.DATA,
        project_id=route.project.project_id,
        work_session_id=route.work_session.work_session_id,
        job_id=job.job_id,
        actor="compute:fake",
        payload={
            "result_ref": result_ref,
            "backend": "fake",
            "result": dict(result),
            "artifact_refs": [],
            "artifacts_dir": str(artifact_root),
            "requires_human_interpretation": True,
        },
        idempotency_key=f"test-result:{suffix}",
    )
    return job, result_ref, event


class _FakeKaggleRunner:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.submitted_message = ""
        self.environments: list[dict[str, str]] = []

    def __call__(
        self,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> KaggleCommandResult:
        self.environments.append(dict(environment))
        if "submit" in command:
            self.submit_calls += 1
            self.submitted_message = command[command.index("-m") + 1]
            return KaggleCommandResult(
                0,
                "Successfully submitted to https://www.kaggle.com/competitions/demo/submissions",
                "",
            )
        if "submissions" in command:
            header = "ref,date,description,status,publicScore,privateScore\n"
            if not self.submitted_message:
                return KaggleCommandResult(0, header, "")
            row = (
                "123,2026-09-01T00:00:00Z,"
                + self.submitted_message
                + ",complete,0.8123,\n"
            )
            return KaggleCommandResult(0, header + row, "")
        raise AssertionError(command)


def _submission_pipeline(
    router: DiscordThreadRouter,
    tmp_path: Path,
    runner: _FakeKaggleRunner,
) -> KaggleSubmissionPipeline:
    return KaggleSubmissionPipeline(
        router=router,
        root_dir=tmp_path / "final-actions" / "kaggle",
        transport=KaggleCliTransport(
            command="kaggle",
            command_runner=runner,
        ),
        rules_acknowledged=("demo-competition",),
        history_poll_seconds=0,
        history_timeout_seconds=0,
    )


def test_exact_sha_approval_submits_once_and_records_lb_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "must-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "core-only-token")
    router = _router(tmp_path)
    route = _route(router, Domain.KAGGLE)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    submission = artifacts / "submission.csv"
    submission.write_text("id,target\n1,0.7\n2,0.2\n", encoding="utf-8")
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    _result_event(
        router,
        route,
        artifacts,
        result={
            "metrics": {"cv": 0.81},
            "competition_slug": "demo-competition",
            "submission_candidate": {
                "path": "submission.csv",
                "message": "cv=0.81",
                "expected_columns": ["id", "target"],
            },
        },
        suffix="K1",
    )
    router.record_human_decision(
        route,
        kind=HumanDecisionKind.KAGGLE_SUBMISSION,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=digest,
        text="submit this exact file",
        actor_id="42",
        message_id="9001",
        actor_is_human=True,
    )

    runner = _FakeKaggleRunner()
    pipeline = _submission_pipeline(router, tmp_path, runner)
    completed = pipeline.execute(route, subject_ref=digest)

    assert completed.state == SubmissionState.SUBMITTED
    assert completed.public_score == pytest.approx(0.8123)
    assert completed.subject_ref == f"sha256:{digest}"
    assert runner.submit_calls == 1
    assert completed.marker in runner.submitted_message
    assert all("DISCORD_BOT_TOKEN" not in item for item in runner.environments)
    assert all("OPENAI_API_KEY" not in item for item in runner.environments)
    assert any(item.get("KAGGLE_API_TOKEN") == "core-only-token" for item in runner.environments)

    duplicate = pipeline.execute(route, subject_ref=f"sha256:{digest}")
    assert duplicate.state == SubmissionState.SUBMITTED
    assert runner.submit_calls == 1

    events = router.store.latest_events(
        work_session_id=route.work_session.work_session_id,
        limit=100,
    )
    assert any(event.event_type == "kaggle.submission.completed" for event in events)
    assert any(event.event_type == "kaggle.submission.history_updated" for event in events)


def test_submission_refuses_changed_file_and_missing_rules(tmp_path: Path):
    router = _router(tmp_path)
    route = _route(router, Domain.KAGGLE)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    submission = artifacts / "submission.csv"
    submission.write_text("id,target\n1,0.7\n", encoding="utf-8")
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    _result_event(
        router,
        route,
        artifacts,
        result={
            "competition_slug": "demo-competition",
            "submission_path": "submission.csv",
        },
        suffix="K2",
    )
    router.record_human_decision(
        route,
        kind=HumanDecisionKind.KAGGLE_SUBMISSION,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=digest,
        text="approved",
        actor_id="42",
        message_id="9002",
        actor_is_human=True,
    )
    runner = _FakeKaggleRunner()
    no_rules = KaggleSubmissionPipeline(
        router=router,
        root_dir=tmp_path / "no-rules",
        transport=KaggleCliTransport(command="kaggle", command_runner=runner),
        history_timeout_seconds=0,
    )
    with pytest.raises(SubmissionBlockedError, match="rules"):
        no_rules.execute(route, subject_ref=digest)

    pipeline = _submission_pipeline(router, tmp_path, runner)
    pipeline.discover_work_session(route.work_session.work_session_id)
    submission.write_text("id,target\n1,0.1\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="changed"):
        pipeline.execute(route, subject_ref=digest)
    assert runner.submit_calls == 0


def test_paper_decision_generates_reviewed_markdown_latex_bib_and_manifest(
    tmp_path: Path,
):
    router = _router(tmp_path)
    route = _route(router, Domain.RESEARCH)
    artifacts = tmp_path / "research-artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text(
        json.dumps({"accuracy": 0.91}),
        encoding="utf-8",
    )
    _, result_ref, _ = _result_event(
        router,
        route,
        artifacts,
        result={
            "summary": "The tested method improved the recorded accuracy.",
            "metrics": {"accuracy": 0.91},
            "method": "fixed train/validation split",
        },
        suffix="R1",
    )
    router.record_human_decision(
        route,
        kind=HumanDecisionKind.RESULT_INTERPRETATION,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=result_ref,
        text="The improvement is meaningful only for the tested split.",
        actor_id="42",
        message_id="9101",
        actor_is_human=True,
    )
    router.record_human_decision(
        route,
        kind=HumanDecisionKind.RESEARCH_PAPER,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=result_ref,
        text="prepare a paper",
        actor_id="42",
        message_id="9102",
        actor_is_human=True,
    )
    pipeline = PaperGenerationPipeline(
        config=HarnessConfig(project_root=tmp_path, paper_provider="fake"),
        router=router,
        root_dir=tmp_path / "papers",
        search_provider=FakePaperSearchProvider(),
        max_revisions=0,
        compile_pdf=False,
    )

    result = pipeline.execute(route, subject_ref=result_ref)
    assert Path(result.markdown_path).is_file()
    assert Path(result.latex_path).is_file()
    assert Path(result.bibtex_path).is_file()
    assert Path(result.evidence_path).is_file()
    assert Path(result.review_path).is_file()
    assert Path(result.manifest_path).is_file()
    markdown = Path(result.markdown_path).read_text(encoding="utf-8")
    assert result_ref in markdown
    for section in PaperGenerationPipeline.REQUIRED_SECTIONS:
        assert f"## {section}" in markdown
    review = json.loads(Path(result.review_path).read_text(encoding="utf-8"))
    assert review["external_publication_performed"] is False

    repeated = pipeline.execute(route, subject_ref=result_ref)
    assert repeated.paper_id == result.paper_id
    completed_events = [
        event
        for event in router.store.latest_events(
            work_session_id=route.work_session.work_session_id,
            limit=100,
        )
        if event.event_type == "research.paper.completed"
    ]
    assert len(completed_events) == 1


def test_paper_pipeline_requires_accepted_result_interpretation(tmp_path: Path):
    router = _router(tmp_path)
    route = _route(router, Domain.RESEARCH)
    artifacts = tmp_path / "research-artifacts"
    artifacts.mkdir()
    _, result_ref, _ = _result_event(
        router,
        route,
        artifacts,
        result={"metrics": {"score": 1.0}},
        suffix="R2",
    )
    router.record_human_decision(
        route,
        kind=HumanDecisionKind.RESEARCH_PAPER,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=result_ref,
        text="prepare",
        actor_id="42",
        message_id="9201",
        actor_is_human=True,
    )
    pipeline = PaperGenerationPipeline(
        config=HarnessConfig(project_root=tmp_path, paper_provider="fake"),
        router=router,
        root_dir=tmp_path / "papers",
        search_provider=FakePaperSearchProvider(),
        max_revisions=0,
    )
    with pytest.raises(PaperPipelineBlockedError, match="interpretation"):
        pipeline.execute(route, subject_ref=result_ref)


def test_final_action_coordinator_recovers_decisions_and_executes_both_domains(
    tmp_path: Path,
):
    router = _router(tmp_path)
    kaggle = _route(router, Domain.KAGGLE)
    research = _route(router, Domain.RESEARCH)

    kaggle_artifacts = tmp_path / "kaggle-artifacts"
    kaggle_artifacts.mkdir()
    submission_path = kaggle_artifacts / "submission.csv"
    submission_path.write_text("id,target\n1,0.5\n", encoding="utf-8")
    digest = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    _result_event(
        router,
        kaggle,
        kaggle_artifacts,
        result={
            "competition_slug": "demo-competition",
            "submission_path": "submission.csv",
        },
        suffix="K3",
    )
    router.record_human_decision(
        kaggle,
        kind=HumanDecisionKind.KAGGLE_SUBMISSION,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=digest,
        text="submit",
        actor_id="42",
        message_id="9301",
        actor_is_human=True,
    )

    research_artifacts = tmp_path / "paper-artifacts"
    research_artifacts.mkdir()
    _, result_ref, _ = _result_event(
        router,
        research,
        research_artifacts,
        result={"summary": "scoped result", "metrics": {"score": 0.7}},
        suffix="R3",
    )
    router.record_human_decision(
        research,
        kind=HumanDecisionKind.RESULT_INTERPRETATION,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=result_ref,
        text="scoped interpretation",
        actor_id="42",
        message_id="9302",
        actor_is_human=True,
    )
    router.record_human_decision(
        research,
        kind=HumanDecisionKind.RESEARCH_PAPER,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=result_ref,
        text="write paper",
        actor_id="42",
        message_id="9303",
        actor_is_human=True,
    )

    runner = _FakeKaggleRunner()
    coordinator = FinalActionCoordinator(
        router=router,
        submission=_submission_pipeline(router, tmp_path, runner),
        paper=PaperGenerationPipeline(
            config=HarnessConfig(project_root=tmp_path, paper_provider="fake"),
            router=router,
            root_dir=tmp_path / "papers",
            search_provider=FakePaperSearchProvider(),
            max_revisions=0,
        ),
        root_dir=tmp_path / "final-actions",
        scan_interval_seconds=60,
        retry_interval_seconds=0.2,
        max_failure_attempts=2,
        max_concurrent_actions=2,
    )
    try:
        coordinator.run_until_idle(timeout_seconds=15)
        records = coordinator.actions.list()
        assert len(records) == 2
        assert {item.state for item in records} == {FinalActionState.SUCCEEDED}
        assert runner.submit_calls == 1
    finally:
        coordinator.stop(wait=True)
