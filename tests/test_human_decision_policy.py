from __future__ import annotations

from pathlib import Path

import pytest

from harness.control_plane import ControlPlaneStore, Domain
from harness.discord_thread_router import (
    ChannelDomainMap,
    ControlledAction,
    DiscordLocation,
    DiscordThreadRouter,
    HumanDecisionKind,
    HumanDecisionVerdict,
    HumanResponsibilityPolicy,
)


def _router(tmp_path: Path) -> DiscordThreadRouter:
    return DiscordThreadRouter(
        ControlPlaneStore(tmp_path / "control-plane"),
        ChannelDomainMap({"100": Domain.RESEARCH, "200": Domain.KAGGLE}),
    )


def _kaggle_route(router: DiscordThreadRouter):
    return router.resolve_work_session(
        DiscordLocation(
            guild_id="1",
            channel_id="201",
            parent_channel_id="200",
            thread_id="201",
        ),
        title="Kaggle",
    )


def test_each_domain_keeps_exactly_three_human_owned_decisions():
    assert HumanResponsibilityPolicy.required_decisions(Domain.KAGGLE) == (
        HumanDecisionKind.HYPOTHESIS,
        HumanDecisionKind.RESULT_INTERPRETATION,
        HumanDecisionKind.KAGGLE_SUBMISSION,
    )
    assert HumanResponsibilityPolicy.required_decisions(Domain.RESEARCH) == (
        HumanDecisionKind.HYPOTHESIS,
        HumanDecisionKind.RESULT_INTERPRETATION,
        HumanDecisionKind.RESEARCH_PAPER,
    )
    assert len(HumanResponsibilityPolicy.human_tasks(Domain.KAGGLE)) == 3
    assert len(HumanResponsibilityPolicy.human_tasks(Domain.RESEARCH)) == 3

    with pytest.raises(ValueError, match="only valid in the kaggle"):
        HumanResponsibilityPolicy.decision_for_action(
            Domain.RESEARCH,
            ControlledAction.SUBMIT_KAGGLE,
        )
    with pytest.raises(ValueError, match="only valid in the research"):
        HumanResponsibilityPolicy.decision_for_action(
            Domain.KAGGLE,
            ControlledAction.START_PAPER_DRAFT,
        )


def test_agent_or_bot_cannot_satisfy_a_human_decision_gate(tmp_path: Path):
    router = _router(tmp_path)
    route = _kaggle_route(router)

    with pytest.raises(PermissionError, match="cannot satisfy"):
        router.record_human_decision(
            route,
            kind=HumanDecisionKind.HYPOTHESIS,
            verdict=HumanDecisionVerdict.ACCEPT,
            subject_ref="hypothesis-1",
            text="agent says yes",
            actor_id="bot-1",
            message_id="1200",
            actor_is_human=False,
        )

    gate = router.check_human_gate(
        route,
        action=ControlledAction.START_EXPERIMENT,
        subject_ref="hypothesis-1",
    )
    assert gate.allowed is False
    assert gate.required_decision == HumanDecisionKind.HYPOTHESIS


def test_hypothesis_and_result_interpretation_are_subject_specific(tmp_path: Path):
    router = _router(tmp_path)
    route = _kaggle_route(router)

    router.record_human_decision(
        route,
        kind=HumanDecisionKind.HYPOTHESIS,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref="hypothesis-1",
        text="try it",
        actor_id="42",
        message_id="1300",
        actor_is_human=True,
    )
    assert router.check_human_gate(
        route,
        action=ControlledAction.START_EXPERIMENT,
        subject_ref="hypothesis-1",
    ).allowed
    assert not router.check_human_gate(
        route,
        action=ControlledAction.START_EXPERIMENT,
        subject_ref="hypothesis-2",
    ).allowed

    router.record_human_decision(
        route,
        kind=HumanDecisionKind.RESULT_INTERPRETATION,
        verdict=HumanDecisionVerdict.DEFER,
        subject_ref="experiment-1-result",
        text="not enough evidence",
        actor_id="42",
        message_id="1301",
        actor_is_human=True,
    )
    deferred = router.check_human_gate(
        route,
        action=ControlledAction.CONTINUE_FROM_RESULT,
        subject_ref="experiment-1-result",
    )
    assert deferred.allowed is False
    assert deferred.verdict == HumanDecisionVerdict.DEFER


def test_kaggle_submission_approval_is_bound_to_exact_sha256_and_latest_verdict(
    tmp_path: Path,
):
    router = _router(tmp_path)
    route = _kaggle_route(router)
    first_sha = "a" * 64
    changed_sha = "b" * 64

    with pytest.raises(ValueError, match="exact SHA-256"):
        router.record_human_decision(
            route,
            kind=HumanDecisionKind.KAGGLE_SUBMISSION,
            verdict=HumanDecisionVerdict.ACCEPT,
            subject_ref="submission.csv",
            text="invalid reference",
            actor_id="42",
            message_id="1400",
            actor_is_human=True,
        )

    router.record_human_decision(
        route,
        kind=HumanDecisionKind.KAGGLE_SUBMISSION,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref=first_sha,
        text="submit this exact file",
        actor_id="42",
        message_id="1401",
        actor_is_human=True,
    )
    accepted = router.check_human_gate(
        route,
        action=ControlledAction.SUBMIT_KAGGLE,
        subject_ref=f"sha256:{first_sha}",
    )
    assert accepted.allowed is True
    assert accepted.subject_ref == f"sha256:{first_sha}"
    assert not router.check_human_gate(
        route,
        action=ControlledAction.SUBMIT_KAGGLE,
        subject_ref=changed_sha,
    ).allowed

    router.record_human_decision(
        route,
        kind=HumanDecisionKind.KAGGLE_SUBMISSION,
        verdict=HumanDecisionVerdict.REJECT,
        subject_ref=first_sha,
        text="withdraw approval",
        actor_id="42",
        message_id="1402",
        actor_is_human=True,
    )
    rejected = router.check_human_gate(
        route,
        action=ControlledAction.SUBMIT_KAGGLE,
        subject_ref=first_sha,
    )
    assert rejected.allowed is False
    assert rejected.verdict == HumanDecisionVerdict.REJECT


def test_research_paper_decision_is_not_available_in_kaggle_mode(tmp_path: Path):
    router = _router(tmp_path)
    kaggle = _kaggle_route(router)
    with pytest.raises(ValueError, match="not a human decision"):
        router.record_human_decision(
            kaggle,
            kind=HumanDecisionKind.RESEARCH_PAPER,
            verdict=HumanDecisionVerdict.ACCEPT,
            subject_ref="paper-plan-1",
            text="write paper",
            actor_id="42",
            message_id="1500",
            actor_is_human=True,
        )

    research = router.resolve_work_session(
        DiscordLocation(guild_id="1", channel_id="100"),
        title="Research",
    )
    router.record_human_decision(
        research,
        kind=HumanDecisionKind.RESEARCH_PAPER,
        verdict=HumanDecisionVerdict.ACCEPT,
        subject_ref="paper-plan-1",
        text="turn this into a paper",
        actor_id="42",
        message_id="1501",
        actor_is_human=True,
    )
    assert router.check_human_gate(
        research,
        action=ControlledAction.START_PAPER_DRAFT,
        subject_ref="paper-plan-1",
    ).allowed
