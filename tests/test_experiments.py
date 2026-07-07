import json
import sys

from harness.agent_runner import SubAgentCommandRunner
from harness.command_parser import parse_research_command
from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.experiments import ExperimentEntry, ExperimentStore
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.state import ResearchSession, utc_timestamp


def _entry(exp_id="EXP-001", name="baseline", status="planned", round_id=1, metrics=None):
    now = utc_timestamp()
    return ExperimentEntry(
        exp_id=exp_id,
        name=name,
        hypothesis="baseline improves accuracy",
        status=status,
        round_id=round_id,
        metrics=metrics or {"accuracy": 0.9},
        dir=f"experiments/{exp_id}_{name}",
        created_at=now,
        updated_at=now,
    )


def test_experiment_store_append_upsert_next_id_and_skips_broken_lines(tmp_path):
    path = tmp_path / "experiments.jsonl"
    path.write_text("{broken\n", encoding="utf-8")
    store = ExperimentStore(path)

    first = _entry("EXP-001", "baseline")
    store.append(first)
    assert store.next_exp_id() == "EXP-002"

    updated = _entry("EXP-001", "baseline", status="done", metrics={"accuracy": 0.93})
    store.upsert(updated)
    store.upsert(_entry("EXP-002", "ablation", metrics={"accuracy": 0.88}))

    entries = store.read_all()
    assert [entry.exp_id for entry in entries] == ["EXP-001", "EXP-002"]
    assert entries[0].status == "done"
    assert entries[0].metrics == {"accuracy": 0.93}
    assert store.next_exp_id() == "EXP-003"


def test_new_session_creates_experiment_and_paper_skeleton(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)

    discord.inject(orchestrator, "/re new")

    session = orchestrator.store.load()
    assert session is not None
    assert config.experiments_dir(session.session_id).is_dir()
    assert (config.paper_dir(session.session_id) / "figures").is_dir()


def test_re_exps_returns_empty_message_and_list(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    discord.inject(orchestrator, "/re new")

    empty = orchestrator.handle(parse_research_command("/re exps"))
    assert empty.ok
    assert empty.message == "実験はまだ登録されていません。"

    session = orchestrator.store.load()
    assert session is not None
    store = ExperimentStore(config.experiments_path(session.session_id))
    store.append(_entry("EXP-001", "baseline", status="done", metrics={"accuracy": 0.93}))
    store.append(_entry("EXP-002", "ablation", status="failed", metrics={"error": "timeout"}))

    listed = orchestrator.handle(parse_research_command("/re exps"))
    assert listed.ok
    assert "EXP-001 [done] baseline" in listed.message
    assert "accuracy=0.93" in listed.message
    assert "EXP-002 [failed] ablation" in listed.message
    assert "experiments/EXP-002_ablation" in listed.message

    entries = orchestrator._journal(session).read_entries()
    assert any(entry.get("event_type") == "experiments_list_requested" for entry in entries)


def test_re_task_rejects_in_planning(tmp_path):
    config = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command=f"{sys.executable} -c \"print('should-not-run')\"",
    )
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    discord.inject(orchestrator, "/re new")

    result = orchestrator.handle(parse_research_command("/re task run baseline"))

    assert result.ok
    assert "PLANNING中は /re task を使えません" in result.message


def test_re_task_rejects_without_sub_agent_command(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    discord.inject(orchestrator, "/re new")
    session = orchestrator.store.load()
    assert session is not None
    session.transition_to(Mode.RESEARCH)
    orchestrator.store.save(session)

    result = orchestrator.handle(parse_research_command("/re task run baseline"))

    assert result.ok
    assert result.message == "実サブエージェント未設定のため /re task は使えません。"


def test_re_task_runs_and_records_round_ledger_and_journal(tmp_path):
    config = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command=(
            f"{sys.executable} -c "
            "\"import sys; data=sys.stdin.read(); print('task-ok'); print('run baseline' in data)\""
        ),
        max_command_seconds=10,
    )
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    discord.inject(orchestrator, "/re new")
    session = orchestrator.store.load()
    assert session is not None
    session.transition_to(Mode.RESEARCH)
    orchestrator.store.save(session)

    result = orchestrator.handle(parse_research_command("/re task run baseline"))

    assert result.ok
    assert "task-ok" in result.message
    assert "True" in result.message
    session = orchestrator.store.load()
    assert session is not None
    assert session.round_id == 1
    assert session.mode == Mode.RESEARCH

    ledger_entries = orchestrator._research_ledger(session).read_entries()
    assert len(ledger_entries) == 1
    assert ledger_entries[0].round_id == 1
    assert ledger_entries[0].action == "run baseline"
    assert "task-ok" in ledger_entries[0].result

    journal_entries = orchestrator._journal(session).read_entries()
    assert any(entry.get("event_type") == "research_round_completed" for entry in journal_entries)


def test_re_task_keeps_done_mode_when_no_approval(tmp_path):
    config = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command=f"{sys.executable} -c \"print('done-task')\"",
        max_command_seconds=10,
    )
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    discord.inject(orchestrator, "/re new")
    session = orchestrator.store.load()
    assert session is not None
    session.transition_to(Mode.DONE)
    orchestrator.store.save(session)

    result = orchestrator.handle(parse_research_command("/re task run after done"))

    assert result.ok
    assert "done-task" in result.message
    session = orchestrator.store.load()
    assert session is not None
    assert session.round_id == 1
    assert session.mode == Mode.DONE


def test_sub_prompt_includes_experiment_ledger_summary(tmp_path):
    session = ResearchSession.new("experiment prompt")
    session.research_dir = str(tmp_path)
    config = HarnessConfig(project_root=tmp_path)
    session_dir = config.session_dir(session.session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    config.plan_path(session.session_id).write_text("plan", encoding="utf-8")
    ExperimentStore(config.experiments_path(session.session_id)).append(
        _entry("EXP-001", "baseline", status="done", metrics={"accuracy": 0.93})
    )

    prompt = SubAgentCommandRunner(config)._build_prompt(session, round_number=2, task="next")

    assert "実験台帳サマリ:" in prompt
    assert "EXP-001 [done] baseline" in prompt
    assert "accuracy" in prompt
