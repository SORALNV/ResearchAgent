from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from harness.discord_channel_map import DiscordLocation
from harness.discord_execution_ui import (
    ExecutionThreadRegistry,
    attach_execution_narration_prompt,
    build_help_message,
    build_job_list_message,
    build_readiness_message,
    execution_opening_message,
    execution_thread_name,
    format_job_progress,
    job_is_terminal,
)


def _location(channel_id: str = "2001") -> DiscordLocation:
    return DiscordLocation(
        guild_id="1000",
        channel_id=channel_id,
        parent_channel_id=None,
        thread_id=None,
    )


def _channel(tmp_path: Path):
    return SimpleNamespace(
        subject="House Prices",
        domain=SimpleNamespace(value="kaggle"),
        status=SimpleNamespace(value="active"),
        work_session_id="WS-1",
        created_by="123456789012345678",
        target_ref="house-prices-advanced-regression-techniques",
    )


def test_execution_thread_registry_round_trip_and_parent_route(tmp_path: Path) -> None:
    registry = ExecutionThreadRegistry(tmp_path)
    record = registry.bind(
        thread_id="3001",
        location=_location(),
        work_session_id="WS-1",
        source_message_id="4001",
        action_kind="experiment",
        subject="House Prices",
    )
    registry.bind_jobs("3001", ["JOB-1", "JOB-2", "JOB-1"])
    registry.set_status("3001", "watching")

    loaded = ExecutionThreadRegistry(tmp_path).get("3001")
    assert loaded is not None
    assert loaded.work_session_id == "WS-1"
    assert loaded.job_ids == ("JOB-1", "JOB-2")
    assert loaded.status == "watching"
    assert loaded.parent_location() == _location()
    assert record.parent_conversation_id == "2001"


def test_help_and_opening_messages_are_compact_and_explicit(tmp_path: Path) -> None:
    channel = _channel(tmp_path)
    service = SimpleNamespace(
        registry=SimpleNamespace(get=lambda location: channel),
    )

    help_text = build_help_message(service, _location())
    assert "/agent help" in help_text
    assert "/agent readiness" in help_text
    assert "/agent job list" in help_text
    assert "専用Thread" in help_text

    opening = execution_opening_message(
        subject="ResearchAgent",
        action_kind="experiment",
        request_text="origin/mainを取り込んで修正して",
    )
    assert "現在状態" in opening
    assert "未マージ変更" in opening
    assert "非公開の思考そのものではなく" in opening
    assert len(execution_thread_name("House Prices", "experiment", "123456789")) <= 95


def test_readiness_reports_active_dependencies_without_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "secret-not-rendered")
    channel = _channel(tmp_path)
    service = SimpleNamespace(
        registry=SimpleNamespace(get=lambda location: channel),
        discord_access_policy=SimpleNamespace(
            required=True,
            global_user_ids=frozenset({"123456789012345678"}),
        ),
        codex_status=lambda location, title: {"running": True},
        compute=SimpleNamespace(
            broker=SimpleNamespace(
                snapshot=lambda: {
                    "kaggle_notebook": {"available": True},
                    "remote_gpu": {"available": False},
                }
            )
        ),
        config=SimpleNamespace(project_root=tmp_path),
        final_actions=SimpleNamespace(submission=object()),
    )

    result = build_readiness_message(service, _location())
    assert "Readiness: READY" in result
    assert "App Server running" in result
    assert "kaggle_notebook" in result
    assert "secret-not-rendered" not in result


def test_job_list_and_progress_use_current_work_session(tmp_path: Path) -> None:
    channel = _channel(tmp_path)
    running = SimpleNamespace(
        job_id="JOB-1",
        status=SimpleNamespace(value="running"),
        backend_id="kaggle_notebook",
        checkpoint_ref=None,
        error="",
        spec=SimpleNamespace(
            payload={"title": "CatBoost native categorical"},
            experiment_id="hypothesis:P-021",
        ),
    )
    succeeded = SimpleNamespace(
        job_id="JOB-2",
        status=SimpleNamespace(value="succeeded"),
        backend_id="local_gpu_worker",
        checkpoint_ref="result:JOB-2:abc",
        error="",
        spec=SimpleNamespace(payload={}, experiment_id="hypothesis:P-022"),
    )
    store = SimpleNamespace(list_jobs=lambda work_session_id: [running, succeeded])
    service = SimpleNamespace(
        registry=SimpleNamespace(get=lambda location: channel),
        router=SimpleNamespace(store=store),
    )

    result = build_job_list_message(service, _location())
    assert "JOB-1" in result and "running" in result
    assert "JOB-2" in result and "result:JOB-2:abc" in result
    assert "CatBoost native categorical" in result
    assert job_is_terminal(running) is False
    assert job_is_terminal(succeeded) is True
    assert "local_gpu_worker" in format_job_progress(succeeded)


def test_execution_narration_prompt_is_attached_once() -> None:
    class Handler:
        def _build_prompt(self, value: str) -> str:
            return value

    handler = Handler()
    service = SimpleNamespace(
        dispatcher=SimpleNamespace(handlers={"kaggle": handler})
    )
    attach_execution_narration_prompt(service)
    first = handler._build_prompt("base")
    attach_execution_narration_prompt(service)
    second = handler._build_prompt("base")

    assert "次に何をするか" in first
    assert "chain-of-thought" in first
    assert first == second


def test_active_adapter_registers_requested_commands_and_execution_threads() -> None:
    root = Path(__file__).resolve().parents[1]
    adapter = (root / "harness" / "natural_channel_discord.py").read_text(
        encoding="utf-8"
    )
    service = (root / "harness" / "codex_app_server_service.py").read_text(
        encoding="utf-8"
    )

    assert 'name="help"' in adapter
    assert 'name="readiness"' in adapter
    assert 'name="job"' in adapter
    assert 'name="list"' in adapter
    assert "message.create_thread" in adapter
    assert "DISCORD_EXECUTION_THREADS" in (
        root / "main.py"
    ).read_text(encoding="utf-8")
    assert 'payload.get("phase") or ""' in service
    assert '== "commentary"' in service
