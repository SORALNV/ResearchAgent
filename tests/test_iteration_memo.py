from pathlib import Path
from types import SimpleNamespace

from harness.channel_sessions import ChannelSessionRegistry
from harness.compute_feedback import ResultFeedbackEngine
from harness.config import HarnessConfig
from harness.control_plane import (
    ControlPlaneStore,
    Domain,
    JobSpec,
    ResourceRequirements,
)
from harness.discord_channel_map import ChannelDomainMap, DiscordLocation
from harness.discord_thread_router import DiscordThreadRouter
from harness.iteration_memo import (
    IterationMemoEngine,
    IterationOutcome,
)
from harness.kaggle_methodbook import MethodCardStatus, MethodCardStore
from harness.learning_feedback import LearningResultFeedbackAdapter
from harness.methodbook_natural import MethodBookConversationHandler


class StaticMemoPlanner:
    def summarize(self, *, job, result, result_ref, proposals, backend):
        return {
            "outcome": "improved",
            "lesson_summary": "native categorical処理は固定CVで再現可能な改善を示した。",
            "lessons": ["カテゴリ列を文字列のまま保持する"],
            "anti_patterns": ["事前one-hotとnative categoricalを混在させない"],
            "quality_gates": ["CVSpecを固定しseed再現を確認する"],
            "reusable_assets": ["catboost_pipeline.py"],
            "discard": ["重複one-hot前処理"],
            "next_best_action": "別seedで同じ差分を再検証する",
            "method_candidates": [
                {
                    "claim": "CatBoost native categorical improves tabular AUC when raw categories are preserved",
                    "scope": {
                        "task_family": "tabular",
                        "modality": "structured",
                        "metric_family": "auc",
                        "conditions": ["fixed CVSpec", "raw categorical columns"],
                        "tags": ["catboost", "categorical"],
                    },
                    "evidence_kind": "support",
                    "next_falsification": "別seedと別コンペで再現する",
                }
            ],
            "planner": "static-test",
        }


def _job(
    store: ControlPlaneStore,
    *,
    competition: str,
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
                "title": "CatBoost native categorical",
                "hypothesis": "native categorical処理でAUCが改善する",
                "competition_slug": competition,
                "task_family": "tabular",
                "modality": "structured",
                "metric_family": "auc",
                "baseline_metric": 0.8398,
                "metric_direction": "maximize",
                "proposal_metadata": {
                    "success_condition": "cv_auc >= 0.8430",
                    "cv_spec": "CV-001",
                },
            },
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=2048,
                accelerator="cpu",
            ),
            backend_preferences=("fake",),
            experiment_id=f"P-021-{suffix}",
        )
    )
    return project, session, job


def _result(value: float = 0.8440):
    return {
        "status": "succeeded",
        "summary": "score improved",
        "primary_metric": {
            "name": "cv_auc",
            "value": value,
            "baseline": 0.8398,
            "direction": "maximize",
        },
        "metrics": {"cv_auc": value},
        "task_family": "tabular",
        "modality": "structured",
        "metric_family": "auc",
    }


def test_iteration_memo_persists_lessons_events_and_method_card(tmp_path: Path) -> None:
    store = ControlPlaneStore(tmp_path / "control-plane")
    _, session, job = _job(store, competition="comp-a", suffix="a")
    method_store = MethodCardStore(tmp_path / "knowledge")
    engine = IterationMemoEngine(
        store,
        method_store.root,
        method_store,
        planner=StaticMemoPlanner(),
    )

    memo = engine.integrate(
        job=job,
        result=_result(),
        result_ref=f"result:{job.job_id}:abc",
        proposals=[{"subject_ref": "hypothesis:next", "title": "next"}],
        artifact_refs=("artifact#sha256=abc",),
        backend="fake",
    )
    assert memo is not None
    assert memo.outcome == IterationOutcome.IMPROVED
    assert memo.method_card_ids
    assert memo.metric.delta is not None and memo.metric.delta > 0
    assert (method_store.root / "competitions" / "comp-a" / "MEMO.md").is_file()
    assert method_store.markdown_path.is_file()

    card = method_store.get(memo.method_card_ids[0])
    assert card is not None
    assert card.status == MethodCardStatus.LOCAL
    event_types = {
        event.event_type
        for event in store.latest_events(
            work_session_id=session.work_session_id,
            limit=200,
        )
    }
    assert IterationMemoEngine.MEMO_EVENT in event_types
    assert IterationMemoEngine.METHOD_EVENT in event_types

    repeated = engine.integrate(
        job=job,
        result=_result(),
        result_ref=f"result:{job.job_id}:abc",
        proposals=(),
        artifact_refs=(),
        backend="fake",
    )
    assert repeated == memo
    assert method_store.get(card.method_id).revision == card.revision


def test_same_method_across_two_competitions_becomes_verified(tmp_path: Path) -> None:
    store = ControlPlaneStore(tmp_path / "control-plane")
    method_store = MethodCardStore(tmp_path / "knowledge")
    engine = IterationMemoEngine(
        store,
        method_store.root,
        method_store,
        planner=StaticMemoPlanner(),
    )
    _, _, job_a = _job(store, competition="comp-a", suffix="a")
    _, _, job_b = _job(store, competition="comp-b", suffix="b")

    memo_a = engine.integrate(
        job=job_a,
        result=_result(),
        result_ref=f"result:{job_a.job_id}:a",
        backend="fake",
    )
    memo_b = engine.integrate(
        job=job_b,
        result=_result(0.845),
        result_ref=f"result:{job_b.job_id}:b",
        backend="fake",
    )
    assert memo_a is not None and memo_b is not None
    assert memo_a.method_card_ids == memo_b.method_card_ids
    card = method_store.get(memo_b.method_card_ids[0])
    assert card is not None
    assert card.status == MethodCardStatus.VERIFIED


def test_learning_feedback_failure_does_not_erase_experiment_result(tmp_path: Path) -> None:
    store = ControlPlaneStore(tmp_path / "control-plane")
    _, session, job = _job(store, competition="comp-a", suffix="a")
    base = ResultFeedbackEngine(store, tmp_path / "compute")

    class ExplodingMemoEngine:
        MEMO_FAILED_EVENT = IterationMemoEngine.MEMO_FAILED_EVENT
        method_store = MethodCardStore(tmp_path / "knowledge")

        def integrate(self, **kwargs):
            raise RuntimeError("intentional memo failure")

    adapter = LearningResultFeedbackAdapter(base, ExplodingMemoEngine())
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    outcome = adapter.integrate(
        job=job,
        collected_result=_result(),
        artifacts_dir=artifacts,
        backend="fake",
    )
    assert outcome.result_ref.startswith(f"result:{job.job_id}:")
    failures = [
        event
        for event in store.latest_events(
            work_session_id=session.work_session_id,
            limit=200,
        )
        if event.event_type == IterationMemoEngine.MEMO_FAILED_EVENT
    ]
    assert len(failures) == 1
    assert failures[0].payload["experiment_completion_preserved"] is True


def test_kaggle_conversation_prompt_receives_relevant_method_cards(tmp_path: Path) -> None:
    config = HarnessConfig(project_root=tmp_path)
    store = ControlPlaneStore(tmp_path / "control-plane")
    registry = ChannelSessionRegistry(tmp_path / "control-plane")
    location = DiscordLocation(guild_id="100", channel_id="200")
    channel = registry.setup(
        location,
        domain="kaggle",
        subject="tabular competition",
        target_ref="comp-a",
        actor_id="human",
    )
    router = DiscordThreadRouter(store, ChannelDomainMap({"200": Domain.KAGGLE}))
    ingress = router.ingest_message(
        location,
        message_id="300",
        actor_id="400",
        text="CatBoostを検討して",
        title=channel.subject,
    )
    method_store = MethodCardStore(tmp_path / "knowledge")
    _, _, job = _job(store, competition="comp-b", suffix="method")
    memo_engine = IterationMemoEngine(
        store,
        method_store.root,
        method_store,
        planner=StaticMemoPlanner(),
    )
    memo_engine.integrate(
        job=job,
        result=_result(),
        result_ref=f"result:{job.job_id}:method",
        backend="fake",
    )

    handler = MethodBookConversationHandler(
        config,
        registry,
        Domain.KAGGLE,
        store,
        executor=SimpleNamespace(),
        method_store=method_store,
    )
    prompt = handler._build_prompt(ingress, channel)
    assert "<UNTRUSTED_METHODBOOK>" in prompt
    assert "CatBoost native categorical" in prompt
    assert "method_card_ids" in prompt
