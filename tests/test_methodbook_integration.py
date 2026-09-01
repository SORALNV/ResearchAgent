from pathlib import Path
from types import SimpleNamespace

from harness.channel_sessions import ChannelSessionRegistry
from harness.compute_feedback import ResultFeedbackEngine
from harness.compute_scheduler import ComputeStack
from harness.config import HarnessConfig
from harness.control_plane import (
    ControlPlaneStore,
    Domain,
    JobSpec,
    ResourceRequirements,
)
from harness.iteration_memo import IterationMemoEngine
from harness.kaggle_methodbook import (
    EvidenceKind,
    MethodCandidate,
    MethodCardStatus,
    MethodCardStore,
    MethodEvidence,
    MethodScope,
    ValidationKind,
)
from harness.learning_feedback import LearningResultFeedbackAdapter
from harness.learning_integration import attach_iteration_learning
from harness.methodbook_natural import (
    MethodBookConversationHandler,
)
from harness.methodbook_planner import MethodBookAwareMemoPlanner
from harness.natural_channel_service_v2 import NaturalConversationHandler


class MinimalMemoPlanner:
    def summarize(self, *, job, result, result_ref, proposals, backend):
        return {
            "outcome": "improved",
            "lesson_summary": "referenced method was evaluated",
            "lessons": [],
            "anti_patterns": [],
            "quality_gates": [],
            "method_candidates": [],
        }


def _candidate() -> MethodCandidate:
    return MethodCandidate(
        claim="Target encoding improves tabular CV when fitted inside each fold",
        scope=MethodScope(
            task_family="tabular",
            modality="structured",
            metric_family="auc",
            conditions=("fit encoder inside fold", "fixed CVSpec"),
            tags=("target-encoding",),
        ),
        evidence_kind=EvidenceKind.SUPPORT,
        next_falsification="repeat on another seed and competition",
    )


def _job(
    store: ControlPlaneStore,
    *,
    competition: str,
    method_id: str,
    suffix: str,
):
    project = store.create_project(f"project-{suffix}", Domain.KAGGLE)
    session = store.create_work_session(project.project_id, f"session-{suffix}")
    job = store.create_job(
        JobSpec(
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            domain=Domain.KAGGLE,
            kind="experiment",
            payload={
                "title": "reuse target encoding",
                "hypothesis": "referenced MethodCard improves the fixed CV",
                "competition_slug": competition,
                "task_family": "tabular",
                "modality": "structured",
                "metric_family": "auc",
                "proposal_metadata": {
                    "method_card_ids": [method_id],
                    "cv_spec": "CV-001",
                },
            },
            resources=ResourceRequirements(
                cpu_cores=1,
                memory_mb=512,
                accelerator="cpu",
            ),
            backend_preferences=("fake",),
            experiment_id=f"EXP-{suffix}",
        )
    )
    return job


def _result(name: str = "cv_auc", value: float = 0.844) -> dict:
    return {
        "status": "succeeded",
        "primary_metric": {
            "name": name,
            "value": value,
            "baseline": 0.840,
            "direction": "maximize",
        },
        "metrics": {name: value},
        "task_family": "tabular",
        "modality": "structured",
        "metric_family": "auc",
    }


def test_reused_method_card_receives_cross_competition_evidence(tmp_path: Path) -> None:
    store = ControlPlaneStore(tmp_path / "control-plane")
    method_store = MethodCardStore(tmp_path / "knowledge")
    initial = method_store.record(
        _candidate(),
        MethodEvidence(
            result_ref="result:comp-a:seed-a",
            competition="comp-a",
            memo_id="MEMO-A",
            outcome="improved",
            validation_kind=ValidationKind.CV,
            metric_name="cv_auc",
            metric_value=0.843,
            metric_delta=0.003,
            independent_key="seed-a",
        ),
    )
    assert initial.status == MethodCardStatus.LOCAL

    aware = MethodBookAwareMemoPlanner(MinimalMemoPlanner(), method_store)
    engine = IterationMemoEngine(
        store,
        method_store.root,
        method_store,
        planner=aware,
    )
    job = _job(
        store,
        competition="comp-b",
        method_id=initial.method_id,
        suffix="b",
    )
    memo = engine.integrate(
        job=job,
        result=_result(),
        result_ref=f"result:{job.job_id}:b",
        backend="fake",
    )

    assert memo is not None
    assert initial.method_id in memo.method_card_ids
    updated = method_store.get(initial.method_id)
    assert updated is not None
    assert updated.status == MethodCardStatus.VERIFIED
    assert {item.competition for item in updated.evidence} == {"comp-a", "comp-b"}


def test_public_lb_reuse_is_recorded_but_cannot_promote(tmp_path: Path) -> None:
    store = ControlPlaneStore(tmp_path / "control-plane")
    method_store = MethodCardStore(tmp_path / "knowledge")
    initial = method_store.record(
        _candidate(),
        MethodEvidence(
            result_ref="result:comp-a:seed-a",
            competition="comp-a",
            memo_id="MEMO-A",
            outcome="improved",
            validation_kind=ValidationKind.CV,
            metric_name="cv_auc",
            metric_value=0.843,
            metric_delta=0.003,
            independent_key="seed-a",
        ),
    )
    engine = IterationMemoEngine(
        store,
        method_store.root,
        method_store,
        planner=MethodBookAwareMemoPlanner(MinimalMemoPlanner(), method_store),
    )
    job = _job(
        store,
        competition="comp-b",
        method_id=initial.method_id,
        suffix="public-lb",
    )
    engine.integrate(
        job=job,
        result=_result(name="public_lb", value=0.85),
        result_ref=f"result:{job.job_id}:lb",
        backend="kaggle_notebook",
    )
    updated = method_store.get(initial.method_id)
    assert updated is not None
    assert updated.status == MethodCardStatus.LOCAL
    assert any(
        item.validation_kind == ValidationKind.PUBLIC_LB
        for item in updated.evidence
    )


def test_learning_attachment_updates_shared_compute_stack_and_kaggle_handler(
    tmp_path: Path,
) -> None:
    config = HarnessConfig(project_root=tmp_path)
    store = ControlPlaneStore(tmp_path / "control-plane")
    registry = ChannelSessionRegistry(tmp_path / "control-plane")
    base_feedback = ResultFeedbackEngine(store, tmp_path / "compute")
    scheduler = SimpleNamespace(feedback=base_feedback)
    compute = ComputeStack(
        broker=SimpleNamespace(),
        scheduler=scheduler,
        feedback=base_feedback,
        runtime_store=SimpleNamespace(),
    )
    handler = NaturalConversationHandler(
        config,
        registry,
        Domain.KAGGLE,
        store,
        executor=SimpleNamespace(),
    )
    service = SimpleNamespace(
        compute=compute,
        router=SimpleNamespace(store=store),
        dispatcher=SimpleNamespace(handlers={Domain.KAGGLE: handler}),
    )

    result = attach_iteration_learning(
        service,
        config,
        environ={
            "METHODBOOK_ENABLED": "true",
            "METHODBOOK_DIR": str(tmp_path / "knowledge"),
            "ITERATION_MEMO_PROVIDER_ENABLED": "false",
        },
    )

    assert result is service
    assert isinstance(scheduler.feedback, LearningResultFeedbackAdapter)
    assert compute.feedback is scheduler.feedback
    assert isinstance(
        service.dispatcher.handlers[Domain.KAGGLE],
        MethodBookConversationHandler,
    )
    assert service.method_store.markdown_path.is_file()
