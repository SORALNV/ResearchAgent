from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from harness.channel_sessions_compat import ChannelSessionStatus
from harness.config import HarnessConfig
from harness.control_plane import ConflictError, ControlPlaneStore, Domain, ProjectStatus, WorkSessionStatus
from harness.discord_channel_map import DiscordLocation
from harness.discord_thread_router import ChannelDomainMap, DiscordThreadRouter
from harness.natural_channel_service_v2 import build_natural_channel_service


class _BaseService:
    def __init__(self, router: DiscordThreadRouter) -> None:
        self.router = router
        self.dispatcher = SimpleNamespace(router=router)
        self.compute = None
        self.final_actions = None
        self.codex_app_server = None
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self, *, wait: bool = False) -> None:
        self.started = False


def _location(channel_id: str) -> DiscordLocation:
    return DiscordLocation(
        guild_id="1000",
        channel_id=channel_id,
        parent_channel_id=None,
        thread_id=None,
    )


def test_setup_creates_isolated_project_and_work_session_per_channel(tmp_path: Path) -> None:
    router = DiscordThreadRouter(
        ControlPlaneStore(tmp_path / "control-plane"),
        ChannelDomainMap({}),
    )
    service = build_natural_channel_service(
        HarnessConfig(project_root=tmp_path),
        _BaseService(router),
        environ={"CONTROL_PLANE_DIR": str(tmp_path / "control-plane")},
    )

    research = service.setup_channel(
        _location("2001"),
        mode="research",
        subject="知識蒸留の研究",
        target_ref="CIFAR-100",
        actor_id="human-1",
    )
    kaggle = service.setup_channel(
        _location("2002"),
        mode="kaggle",
        subject="House Prices",
        target_ref="house-prices-advanced-regression-techniques",
        actor_id="human-1",
    )

    assert research.config.domain == Domain.RESEARCH
    assert kaggle.config.domain == Domain.KAGGLE
    assert research.route.project.project_id != kaggle.route.project.project_id
    assert (
        research.route.work_session.work_session_id
        != kaggle.route.work_session.work_session_id
    )
    assert service.registry.active("2001").work_session_id == (
        research.route.work_session.work_session_id
    )
    assert service.registry.active("2002").target_ref == (
        "house-prices-advanced-regression-techniques"
    )

    with pytest.raises(ConflictError):
        service.setup_channel(
            _location("2001"),
            mode="research",
            subject="別の研究",
            target_ref="ImageNet",
            actor_id="human-1",
        )


def test_finish_archives_internal_state_and_blocks_channel_reuse(tmp_path: Path) -> None:
    store = ControlPlaneStore(tmp_path / "control-plane")
    router = DiscordThreadRouter(store, ChannelDomainMap({}))
    service = build_natural_channel_service(
        HarnessConfig(project_root=tmp_path),
        _BaseService(router),
        environ={"CONTROL_PLANE_DIR": str(tmp_path / "control-plane")},
    )
    setup = service.setup_channel(
        _location("3001"),
        mode="research",
        subject="再現性検証",
        target_ref="benchmark-A",
        actor_id="human-1",
    )

    archived = service.finish_channel(_location("3001"), actor_id="human-1")

    assert archived.status == ChannelSessionStatus.ARCHIVED
    assert store.get_work_session(setup.route.work_session.work_session_id).status == (
        WorkSessionStatus.CLOSED
    )
    assert store.get_project(setup.route.project.project_id).status == (
        ProjectStatus.ARCHIVED
    )
    with pytest.raises(ConflictError):
        service.setup_channel(
            _location("3001"),
            mode="research",
            subject="新しい案件",
            target_ref="benchmark-B",
            actor_id="human-1",
        )
