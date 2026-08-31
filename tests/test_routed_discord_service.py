from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from harness.config import HarnessConfig
from harness.control_plane import ControlPlaneStore, Domain, EventLane
from harness.discord_thread_router import (
    ChannelDomainMap,
    DiscordChannelDispatcher,
    DiscordLocation,
    DiscordThreadRouter,
)
from harness.domain_consultation import (
    DomainConsultationHandler,
    DomainConsultationResponse,
)
from harness.human_decision_policy import (
    ControlledAction,
    HumanDecisionKind,
    HumanDecisionVerdict,
)
from harness.routed_discord_adapter import RoutedDiscordService
from main import (
    _domain_routing_is_configured,
    build_routed_discord_service,
)


@dataclass
class FakeInvocation:
    output: str
    ok: bool = True
    stderr: str = ""
    command: tuple[str, ...] = ("provider:fake",)
    returncode: int = 0
    estimated_input_tokens: int = 12
    estimated_output_tokens: int = 7


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        stage = str(kwargs["stage"])
        return FakeInvocation(output=f"{stage}: consultation response")


def _service(
    tmp_path: Path,
) -> tuple[RoutedDiscordService, FakeExecutor, FakeExecutor]:
    store_root = tmp_path / "control-plane"
    router = DiscordThreadRouter(
        ControlPlaneStore(store_root),
        ChannelDomainMap(
            {
                "100": Domain.RESEARCH,
                "200": Domain.KAGGLE,
            }
        ),
    )
    config = HarnessConfig(project_root=tmp_path)
    research_executor = FakeExecutor()
    kaggle_executor = FakeExecutor()
    dispatcher = DiscordChannelDispatcher(
        router,
        {
            Domain.RESEARCH: DomainConsultationHandler(
                config,
                router.store,
                Domain.RESEARCH,
                executor=research_executor,
            ),
            Domain.KAGGLE: DomainConsultationHandler(
                config,
                router.store,
                Domain.KAGGLE,
                executor=kaggle_executor,
            ),
        },
    )
    return (
        RoutedDiscordService(router, dispatcher),
        research_executor,
        kaggle_executor,
    )


def _research_location() -> DiscordLocation:
    return DiscordLocation(
        guild_id="1",
        channel_id="101",
        parent_channel_id="100",
        thread_id="101",
    )


def _kaggle_location() -> DiscordLocation:
    return DiscordLocation(
        guild_id="1",
        channel_id="201",
        parent_channel_id="200",
        thread_id="201",
    )


def test_service_routes_real_messages_to_domain_specific_consultation_handlers(
    tmp_path: Path,
):
    service, research_executor, kaggle_executor = _service(tmp_path)

    research = service.handle_message(
        _research_location(),
        message_id="1000",
        actor_id="42",
        text="画像認識の新しい仮説を比較したい",
        title="Research thread",
    )
    kaggle = service.handle_message(
        _kaggle_location(),
        message_id="1001",
        actor_id="42",
        text="CVを固定して次の特徴量を相談したい",
        title="Kaggle thread",
    )

    assert research.domain == Domain.RESEARCH
    assert research.message.startswith("discord_research_consultation")
    assert kaggle.domain == Domain.KAGGLE
    assert kaggle.message.startswith("discord_kaggle_consultation")
    assert len(research_executor.calls) == 1
    assert len(kaggle_executor.calls) == 1

    research_prompt = str(research_executor.calls[0]["prompt"])
    kaggle_prompt = str(kaggle_executor.calls[0]["prompt"])
    assert "現在のDomainは research" in research_prompt
    assert "論文としてまとめ始めるかは人間" in research_prompt
    assert "現在のDomainは kaggle" in kaggle_prompt
    assert "対象CSVのSHA-256に対する人間の最終承認" in kaggle_prompt


def test_duplicate_discord_delivery_reuses_response_without_second_model_call(
    tmp_path: Path,
):
    service, _, kaggle_executor = _service(tmp_path)
    arguments = {
        "message_id": "2000",
        "actor_id": "42",
        "text": "same delivery",
        "title": "Kaggle",
    }

    first = service.handle_message(_kaggle_location(), **arguments)
    duplicate = service.handle_message(_kaggle_location(), **arguments)

    assert first.correlation_id == duplicate.correlation_id
    assert first.message == duplicate.message
    assert duplicate.cached is True
    assert len(kaggle_executor.calls) == 1

    events = service.router.store.list_events(
        work_session_id=first.work_session_id,
        lanes=[EventLane.CONTROL],
        limit=50,
    )
    assert sum(
        event.event_type == "discord.message.received"
        for event in events
    ) == 1
    assert sum(
        event.event_type == "discord.assistant.responded"
        for event in events
    ) == 1
    assert not any(
        event.event_type.startswith("human.decision.")
        for event in events
    )


def test_human_decision_commands_drive_exact_domain_gates(tmp_path: Path):
    service, _, _ = _service(tmp_path)
    location = _kaggle_location()
    reply = service.handle_message(
        location,
        message_id="3000",
        actor_id="42",
        text="submission candidateを確認したい",
        title="Kaggle",
    )
    digest = "ab" * 32

    before = service.check_gate(
        location,
        title="Kaggle",
        action=ControlledAction.SUBMIT_KAGGLE,
        subject_ref=digest,
    )
    assert before.allowed is False

    decision = service.record_decision(
        location,
        title="Kaggle",
        kind=HumanDecisionKind.KAGGLE_SUBMISSION,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=digest,
        note="検証結果を確認したので提出可",
        actor_id="42",
        message_id="3001",
        actor_is_human=True,
    )
    assert decision.subject_ref == f"sha256:{digest}"

    exact = service.check_gate(
        location,
        title="Kaggle",
        action=ControlledAction.SUBMIT_KAGGLE,
        subject_ref=digest,
    )
    changed = service.check_gate(
        location,
        title="Kaggle",
        action=ControlledAction.SUBMIT_KAGGLE,
        subject_ref="cd" * 32,
    )
    assert exact.allowed is True
    assert changed.allowed is False
    assert exact.event_id == decision.event_id
    assert reply.work_session_id == decision.work_session_id

    with pytest.raises(
        ValueError,
        match="is not a human decision in kaggle mode",
    ):
        service.record_decision(
            location,
            title="Kaggle",
            kind=HumanDecisionKind.RESEARCH_PAPER,
            verdict=HumanDecisionVerdict.ACCEPT,
            subject_ref="result:R1",
            note="paper",
            actor_id="42",
            message_id="3002",
            actor_is_human=True,
        )


def test_research_paper_and_result_interpretation_remain_human_only(
    tmp_path: Path,
):
    service, _, _ = _service(tmp_path)
    location = _research_location()
    service.handle_message(
        location,
        message_id="4000",
        actor_id="42",
        text="結果を見て次を決めたい",
        title="Research",
    )

    for kind, action, subject in (
        (
            HumanDecisionKind.RESULT_INTERPRETATION,
            ControlledAction.CONTINUE_FROM_RESULT,
            "result:R1",
        ),
        (
            HumanDecisionKind.RESEARCH_PAPER,
            ControlledAction.START_PAPER_DRAFT,
            "result-bundle:R1",
        ),
    ):
        blocked = service.check_gate(
            location,
            title="Research",
            action=action,
            subject_ref=subject,
        )
        assert blocked.allowed is False
        decision = service.record_decision(
            location,
            title="Research",
            kind=kind,
            verdict=HumanDecisionVerdict.ACCEPT,
            subject_ref=subject,
            note="人間が確認",
            actor_id="42",
            message_id=f"4{len(subject)}",
            actor_is_human=True,
        )
        allowed = service.check_gate(
            location,
            title="Research",
            action=action,
            subject_ref=subject,
        )
        assert allowed.allowed is True
        assert allowed.event_id == decision.event_id

    with pytest.raises(PermissionError, match="human-authenticated"):
        service.record_decision(
            location,
            title="Research",
            kind=HumanDecisionKind.HYPOTHESIS,
            verdict=HumanDecisionVerdict.ACCEPT,
            subject_ref="hypothesis:H1",
            note="agent must not approve",
            actor_id="agent",
            message_id="4999",
            actor_is_human=False,
        )


def test_status_exposes_mode_and_only_three_human_direction_decisions(
    tmp_path: Path,
):
    service, _, _ = _service(tmp_path)
    service.handle_message(
        _kaggle_location(),
        message_id="5000",
        actor_id="42",
        text="status",
        title="Kaggle",
    )
    service.record_decision(
        _kaggle_location(),
        title="Kaggle",
        kind=HumanDecisionKind.HYPOTHESIS,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref="hypothesis:H1",
        note="試す",
        actor_id="42",
        message_id="5001",
        actor_is_human=True,
    )

    status = service.status(
        _kaggle_location(),
        title="Kaggle",
    )

    assert "mode: kaggle" in status
    assert "hypothesis: accept (hypothesis:H1)" in status
    assert status.count("\n- ") >= 3
    assert "提出してよいかの最終判断" in status
    assert "論文としてまとめるかの判断" not in status


def test_main_activates_routed_edge_only_for_explicit_domain_configuration(
    tmp_path: Path,
    monkeypatch,
):
    for name in (
        "DISCORD_CHANNEL_DOMAIN_MAP",
        "DISCORD_RESEARCH_CHANNEL_IDS",
        "DISCORD_KAGGLE_CHANNEL_IDS",
        "DISCORD_CHANNEL_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    assert _domain_routing_is_configured() is False
    monkeypatch.setenv("DISCORD_CHANNEL_ID", "999")
    assert _domain_routing_is_configured() is False
    monkeypatch.setenv("DISCORD_KAGGLE_CHANNEL_IDS", "200")
    assert _domain_routing_is_configured() is True

    monkeypatch.setenv("CONTROL_PLANE_DIR", "routed-control")
    config = HarnessConfig(project_root=tmp_path)
    service = build_routed_discord_service(config)
    assert service.router.channel_domains.to_dict() == {"200": "kaggle"}
    assert (tmp_path / "routed-control").is_dir()
