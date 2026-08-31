from __future__ import annotations

from pathlib import Path

import pytest

from harness.control_plane import ConflictError, ControlPlaneStore, Domain, EventLane
from harness.discord_thread_router import (
    ChannelDomainMap,
    DiscordChannelDispatcher,
    DiscordLocation,
    DiscordThreadRouter,
    MissingDomainHandlerError,
    UnmappedDiscordChannelError,
)


def _router(tmp_path: Path) -> DiscordThreadRouter:
    return DiscordThreadRouter(
        ControlPlaneStore(tmp_path / "control-plane"),
        ChannelDomainMap({"100": Domain.RESEARCH, "200": Domain.KAGGLE}),
    )


def _kaggle_location(thread_id: str = "201") -> DiscordLocation:
    return DiscordLocation(
        guild_id="1",
        channel_id=thread_id,
        parent_channel_id="200",
        thread_id=thread_id,
    )


def test_router_reads_control_plane_path_and_channel_map_from_environment(tmp_path: Path):
    root = tmp_path / "custom-control-plane"
    router = DiscordThreadRouter.from_environment(
        environ={
            "CONTROL_PLANE_DIR": str(root),
            "DISCORD_RESEARCH_CHANNEL_IDS": "100",
        }
    )
    result = router.ingest_message(
        DiscordLocation(guild_id="1", channel_id="100"),
        message_id="999",
        actor_id="42",
        text="hello",
        title="Research",
    )
    assert result.route.domain == Domain.RESEARCH
    assert (root / "projects").is_dir()


def test_channel_map_supports_explicit_json_csv_and_parent_inheritance():
    mapping = ChannelDomainMap.from_environment(
        {
            "DISCORD_CHANNEL_DOMAIN_MAP": '{"100":"research","201":"research"}',
            "DISCORD_KAGGLE_CHANNEL_IDS": "200,300",
        }
    )

    assert mapping.resolve("100").domain == Domain.RESEARCH
    assert mapping.resolve("201", parent_channel_id="200").domain == Domain.RESEARCH
    inherited = mapping.resolve("202", parent_channel_id="200")
    assert inherited.domain == Domain.KAGGLE
    assert inherited.route_channel_id == "200"
    assert inherited.inherited_from_parent is True
    assert mapping.to_dict() == {
        "100": "research",
        "200": "kaggle",
        "201": "research",
        "300": "kaggle",
    }


def test_channel_map_rejects_conflicts_hybrid_and_unknown_channels():
    with pytest.raises(ValueError, match="mapped to both"):
        ChannelDomainMap.from_environment(
            {
                "DISCORD_CHANNEL_DOMAIN_MAP": "100=research",
                "DISCORD_KAGGLE_CHANNEL_IDS": "100",
            }
        )
    with pytest.raises(ValueError, match="not hybrid"):
        ChannelDomainMap.parse("100=hybrid")

    mapping = ChannelDomainMap.parse("100=research")
    with pytest.raises(UnmappedDiscordChannelError):
        mapping.resolve("999")


def test_legacy_single_channel_is_research_only_without_explicit_routes():
    mapping = ChannelDomainMap.from_environment({"DISCORD_CHANNEL_ID": "100"})
    assert mapping.to_dict() == {"100": "research"}

    explicit = ChannelDomainMap.from_environment(
        {
            "DISCORD_CHANNEL_ID": "999",
            "DISCORD_KAGGLE_CHANNEL_IDS": "200",
        }
    )
    assert explicit.to_dict() == {"200": "kaggle"}


def test_router_creates_domain_scoped_projects_and_one_session_per_conversation(
    tmp_path: Path,
):
    router = _router(tmp_path)
    research_location = DiscordLocation(guild_id="1", channel_id="100")
    kaggle_location = _kaggle_location()

    research = router.ingest_message(
        research_location,
        message_id="1000",
        actor_id="42",
        text="compare two research methods",
        title="Research discussion",
    )
    kaggle = router.ingest_message(
        kaggle_location,
        message_id="1001",
        actor_id="42",
        text="try a stronger validation split",
        title="Kaggle experiment",
    )
    duplicate = router.ingest_message(
        kaggle_location,
        message_id="1001",
        actor_id="42",
        text="try a stronger validation split",
        title="ignored duplicate title",
    )

    assert research.route.domain == Domain.RESEARCH
    assert kaggle.route.domain == Domain.KAGGLE
    assert research.route.project.project_id != kaggle.route.project.project_id
    assert research.route.work_session.work_session_id != (
        kaggle.route.work_session.work_session_id
    )
    assert duplicate.route.work_session == kaggle.route.work_session
    assert duplicate.event == kaggle.event

    events = router.store.list_events(
        work_session_id=kaggle.route.work_session.work_session_id,
        lanes=[EventLane.CONTROL],
        limit=20,
    )
    assert [event.event_type for event in events] == [
        "discord.route.bound",
        "discord.message.received",
    ]
    assert kaggle.route.work_session.metadata["domain"] == "kaggle"
    assert len(
        kaggle.route.work_session.metadata["human_responsibility_policy"][
            "human_only"
        ]
    ) == 3


def test_router_refuses_to_rebind_an_existing_conversation_to_another_domain(
    tmp_path: Path,
):
    store = ControlPlaneStore(tmp_path / "control-plane")
    location = _kaggle_location()
    first = DiscordThreadRouter(
        store,
        ChannelDomainMap({"200": Domain.KAGGLE}),
    ).resolve_work_session(location, title="Kaggle")
    assert first.domain == Domain.KAGGLE

    changed = DiscordThreadRouter(
        store,
        ChannelDomainMap({"200": Domain.RESEARCH}),
    )
    with pytest.raises(ConflictError, match="already bound"):
        changed.resolve_work_session(location, title="Research")


def test_dispatcher_selects_handlers_only_from_channel_mapping(tmp_path: Path):
    router = _router(tmp_path)
    called: list[tuple[str, str]] = []

    def research_handler(ingress):
        called.append(("research", ingress.event.event_id))
        return "research-result"

    def kaggle_handler(ingress):
        called.append(("kaggle", ingress.event.event_id))
        return "kaggle-result"

    dispatcher = DiscordChannelDispatcher(
        router,
        {
            Domain.RESEARCH: research_handler,
            Domain.KAGGLE: kaggle_handler,
        },
    )
    research = dispatcher.dispatch_message(
        DiscordLocation(guild_id="1", channel_id="100"),
        message_id="1100",
        actor_id="42",
        text="research",
        title="Research",
    )
    kaggle = dispatcher.dispatch_message(
        _kaggle_location(),
        message_id="1101",
        actor_id="42",
        text="kaggle",
        title="Kaggle",
    )

    assert research.domain == Domain.RESEARCH
    assert research.handler_result == "research-result"
    assert kaggle.domain == Domain.KAGGLE
    assert kaggle.handler_result == "kaggle-result"
    assert called == [
        ("research", research.correlation_id),
        ("kaggle", kaggle.correlation_id),
    ]

    incomplete = DiscordChannelDispatcher(
        router,
        {Domain.RESEARCH: research_handler},
    )
    with pytest.raises(MissingDomainHandlerError):
        incomplete.dispatch_message(
            _kaggle_location("202"),
            message_id="1102",
            actor_id="42",
            text="must not fall back to research",
            title="Kaggle",
        )
