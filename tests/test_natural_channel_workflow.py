from __future__ import annotations

from pathlib import Path

import pytest

from harness.channel_sessions import (
    ChannelSessionDomainMap,
    ChannelSessionRegistry,
    ChannelSessionStatus,
)
from harness.control_plane import ConflictError, Domain
from harness.discord_channel_map import ChannelDomainMap, DiscordLocation
from harness.discord_markdown import compact_discord_markdown
from harness.natural_channel_service import (
    _explicit_paper_intent,
    _explicit_run_intent,
    _explicit_submit_intent,
)


def _location(channel_id: str) -> DiscordLocation:
    return DiscordLocation(
        guild_id="1000",
        channel_id=channel_id,
        parent_channel_id=None,
        thread_id=None,
    )


def test_compact_discord_markdown_joins_vertical_label_cards() -> None:
    source = """### 次の候補

期待:
カテゴリ変数の処理改善

成功条件:
CV >= 0.8430

現在best:
0.8398
"""
    result = compact_discord_markdown(source)

    assert "**次の候補**" in result
    assert "**期待:** カテゴリ変数の処理改善" in result
    assert "**成功条件:** CV >= 0.8430" in result
    assert "**現在best:** 0.8398" in result
    assert "\n\n\n" not in result


def test_channel_registry_keeps_one_persistent_topic_per_channel(tmp_path: Path) -> None:
    registry = ChannelSessionRegistry(tmp_path)
    research = registry.setup(
        _location("2001"),
        domain="research",
        subject="画像認識の蒸留研究",
        actor_id="human-1",
    )
    kaggle = registry.setup(
        _location("2002"),
        domain="kaggle",
        subject="Titanic",
        target_ref="titanic",
        actor_id="human-1",
    )
    registry.bind_runtime(
        research.conversation_id,
        project_id=research.project_id,
        work_session_id="WS-RESEARCH",
        codex_thread_id="thr-research",
    )

    loaded = ChannelSessionRegistry(tmp_path)
    assert loaded.active("2001").subject == "画像認識の蒸留研究"
    assert loaded.active("2002").target_ref == "titanic"
    assert loaded.active("2001").codex_thread_id == "thr-research"
    assert len(loaded.list()) == 2

    archived = loaded.archive("2001")
    assert archived.status == ChannelSessionStatus.ARCHIVED
    assert loaded.active("2001") is None
    with pytest.raises(ConflictError):
        loaded.setup(
            _location("2001"),
            domain="research",
            subject="別の研究",
            actor_id="human-1",
        )


def test_dynamic_channel_sessions_override_static_domain_map(tmp_path: Path) -> None:
    registry = ChannelSessionRegistry(tmp_path)
    registry.setup(
        _location("3001"),
        domain=Domain.KAGGLE,
        subject="House Prices",
        target_ref="house-prices-advanced-regression-techniques",
    )
    mapping = ChannelSessionDomainMap(
        ChannelDomainMap({"3001": Domain.RESEARCH, "3002": Domain.RESEARCH}),
        registry,
    )

    assert mapping.resolve("3001").domain == Domain.KAGGLE
    assert mapping.resolve("3002").domain == Domain.RESEARCH


def test_natural_intents_are_explicit_and_negation_is_respected() -> None:
    assert _explicit_run_intent("P-021を実装して試して") is True
    assert _explicit_run_intent("P-021の利点を教えて") is False
    assert _explicit_run_intent("まだ実行しないで") is False
    assert _explicit_run_intent("このリポジトリを改修して") is True
    assert _explicit_run_intent("origin/mainを--ff-onlyで取り込んで") is True
    assert _explicit_run_intent("修正する必要がありますか？") is False
    assert _explicit_submit_intent("このCSVで提出しよう") is True
    assert _explicit_submit_intent("まだ提出しない") is False
    assert _explicit_paper_intent("この結果を論文にまとめて") is True


def test_active_discord_entrypoint_has_no_strategy_mode_commands() -> None:
    root = Path(__file__).resolve().parents[1]
    main = (root / "main.py").read_text(encoding="utf-8")
    adapter = (root / "harness" / "natural_channel_discord.py").read_text(
        encoding="utf-8"
    )

    assert "NaturalChannelDiscordBotAdapter" in main
    assert 'name="setup"' in adapter
    assert 'name="finish"' in adapter
    assert 'name="hypothesis"' not in adapter
    assert 'name="interpret"' not in adapter
    assert 'name="submit"' not in adapter
    assert 'name="paper"' not in adapter
    assert 'name="gate"' not in adapter
    assert "create_threads=_bool_env(\"DISCORD_EXECUTION_THREADS\", True)" in main
    assert 'name="help"' in adapter
    assert 'name="readiness"' in adapter
    assert 'name="job"' in adapter and 'name="list"' in adapter
