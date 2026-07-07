import json

from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator


def make_orchestrator(tmp_path):
    config = HarnessConfig(project_root=tmp_path, max_rounds=3, fresh_interval=2)
    discord = FakeDiscordAdapter()
    return ResearchOrchestrator(config=config, discord=discord), discord, config


class QueryingPlanningDialogue:
    def propose_search_queries(self, session, user_text, purpose):
        return ["research harness agent evaluation"]

    def respond(self, session, user_text, purpose):
        return "## PLANNING壁打ち\nLLMが類似研究検索結果を踏まえて壁打ちします。"


def test_goal_to_start_approval_approve_stop_e2e(tmp_path):
    orchestrator, discord, config = make_orchestrator(tmp_path)
    orchestrator.planning_dialogue = QueryingPlanningDialogue()

    discord.inject(orchestrator, "/re new")


    discord.inject(orchestrator, "/re plan")



    planning_result = discord.inject_message(orchestrator, "研究ハーネスMVPを検証する")
    assert planning_result.ok
    planning = planning_result.message
    session = orchestrator.store.load()
    assert planning_result.mode == "PLANNING"
    assert "PLANNING壁打ち" in planning
    assert session.version_label == "V001.0"
    assert session.research_dir
    assert "research_runs" in session.research_dir
    assert config.artifacts_dir(session.session_id).exists()
    assert config.research_brief_path(session.session_id).exists()
    assert config.journal_path(session.session_id).exists()
    assert session.planning_scout
    assert "PLANNING壁打ち" in planning

    brief = orchestrator.plan()
    assert "# Research Brief" in brief
    assert "研究ハーネスMVPを検証する" in brief

    accepted_gate = discord.inject(orchestrator, "/re accept PG-1").message
    assert "acceptしました" in accepted_gate
    first = discord.inject(orchestrator, "/re start").message
    assert "モード: APPROVAL_BLOCKED" in first
    assert "@Sora AP-1" in first
    assert "承認依頼 [id: AP-1]" in first
    session = orchestrator.store.load()
    assert session.mode == Mode.APPROVAL_BLOCKED
    assert session.round_id == 1

    continued = discord.inject(orchestrator, "/re approve AP-1").message
    assert "モード: DONE" in continued
    session = orchestrator.store.load()
    assert session.mode == Mode.DONE
    assert session.round_id == 3
    assert "AP-1" in session.approvals_received

    stopped = discord.inject(orchestrator, "/re stop").message
    assert "journal entries:" in stopped
    assert "モード: DONE" in stopped
    assert config.research_ledger_path(session.session_id).exists()
    assert config.report_path(session.session_id).exists()
    assert config.run_summary_path(session.session_id).exists()
    session = orchestrator.store.load()
    assert any(gate.phase == "review" and gate.status == "pending" for gate in session.phase_gates.values())
    assert any("モード: PLANNING" in message for message in discord.messages)
    assert any("モード: APPROVAL_BLOCKED" in message for message in discord.messages)
    assert any("モード: RESEARCH" in message for message in discord.messages)
    assert any("モード: DONE" in message for message in discord.messages)
    assert any("モード: APPROVAL_BLOCKED" in message for message in discord.important_messages)
    assert any("event_type" not in message for message in discord.important_messages)
    assert any("approval_requested" in message for message in discord.log_messages)
    assert any(message.channel == "important" for message in discord.sent_messages)
    assert any(message.channel == "log" for message in discord.sent_messages)

    journal_lines = config.journal_path(session.session_id).read_text(encoding="utf-8").splitlines()
    assert len(journal_lines) >= 6
    entries = [json.loads(line) for line in journal_lines]
    assert all("timestamp" in entry for entry in entries)
    assert all("event_type" in entry for entry in entries)
    assert any(entry["event_type"] == "approval_requested" for entry in entries)
    assert any(entry["event_type"] == "approval_received" for entry in entries)
    assert any(entry["event_type"] == "ledger_entry_appended" for entry in entries)
    assert any(entry["event_type"] == "report_generated" for entry in entries)
    assert any(entry["event_type"] == "report_review_completed" for entry in entries)
    assert any(entry["conversation_sessions"] for entry in entries)
    assert any(entry["fresh_agent_output"] for entry in entries)
    assert any(entry["claude_consultation"] for entry in entries)

    ledger_entries = [
        json.loads(line)
        for line in config.research_ledger_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert ledger_entries[0]["node_id"] == "N-001"
    assert ledger_entries[0]["parent_node_id"] is None
    assert ledger_entries[1]["parent_node_id"] == "N-001"
    report = config.report_path(session.session_id).read_text(encoding="utf-8")
    assert "## AI Provenance" in report
    assert "# Run Summary" in config.run_summary_path(session.session_id).read_text(encoding="utf-8")


def test_start_is_blocked_before_goal(tmp_path):
    orchestrator, _, _ = make_orchestrator(tmp_path)
    try:
        orchestrator.start()
    except ValueError as exc:
        assert "Run /re new first" in str(exc)
    else:
        raise AssertionError("start without /goal should fail")


def test_re_new_finishes_previous_session_and_creates_next(tmp_path):
    orchestrator, discord, _ = make_orchestrator(tmp_path)
    discord.inject(orchestrator, "/re new", user_id="sora", channel_id="lab")

    discord.inject(orchestrator, "/re plan", user_id="sora", channel_id="lab")


    first = discord.inject_message(orchestrator, "first goal", user_id="sora", channel_id="lab")
    assert first.ok
    assert discord.input_events[0].user_id == "sora"
    assert discord.input_events[0].channel_id == "lab"
    assert discord.input_events[0].timestamp
    session = orchestrator.store.load()
    created = discord.inject(orchestrator, "/re new")
    assert created.ok
    next_session = orchestrator.store.load()
    assert next_session.session_id != session.session_id
    assert next_session.research_goal == "未設定"
    assert "前テーマの終了処理" in created.message


def test_research_reject_path_through_fake_discord(tmp_path):
    orchestrator, discord, config = make_orchestrator(tmp_path)
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "reject path")
    discord.inject(orchestrator, "/re accept PG-1")
    discord.inject(orchestrator, "/re start")
    rejected = discord.inject(orchestrator, "/re reject AP-1 許可しない")
    assert rejected.ok
    assert rejected.mode == "PLANNING"
    session = orchestrator.store.load()
    entries = [
        json.loads(line)
        for line in config.journal_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["event_type"] == "approval_rejected" for entry in entries)


def test_research_discuss_continues_planning_dialogue(tmp_path):
    orchestrator, discord, config = make_orchestrator(tmp_path)
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "planning dialogue")
    result = discord.inject_message(orchestrator, "それは違う。評価軸はコストを重視したい")
    assert result.ok
    assert "既存研究と違う" in result.message
    session = orchestrator.store.load()
    assert any("評価軸はコスト" in item for item in session.accepted_ideas)
    entries = [
        json.loads(line)
        for line in config.journal_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["event_type"] == "planning_dialogue_completed" for entry in entries)


def test_re_mode_plan_then_plain_message_dialogue(tmp_path):
    orchestrator, discord, config = make_orchestrator(tmp_path)
    created = discord.inject(orchestrator, "/re new")
    assert created.ok
    session = orchestrator.store.load()
    assert session.research_goal == "未設定"

    discord.inject(orchestrator, "/re plan")


    planned = discord.inject_message(orchestrator, "小型LLMの研究支援エージェント")
    assert planned.ok
    assert "既存研究と違う" in planned.message
    session = orchestrator.store.load()
    assert session.research_goal == "小型LLMの研究支援エージェント"

    reply = discord.inject_message(orchestrator, "既存研究との差分を運用コストに置きたい")
    assert reply.ok
    assert "既存研究と違う" in reply.message
    session = orchestrator.store.load()
    assert any("運用コスト" in item for item in session.accepted_ideas)
    entries = [
        json.loads(line)
        for line in config.journal_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["event_type"] == "planning_dialogue_started" for entry in entries)
    assert any(entry["event_type"] == "planning_dialogue_completed" for entry in entries)


def test_research_discuss_runs_llm_chosen_search_query(tmp_path):
    orchestrator, discord, config = make_orchestrator(tmp_path)
    orchestrator.planning_dialogue = QueryingPlanningDialogue()
    discord.inject(orchestrator, "/re new")

    discord.inject(orchestrator, "/re plan")


    discord.inject_message(orchestrator, "planning dialogue")
    result = discord.inject_message(orchestrator, "類似論文を踏まえて方向性を決めたい")
    assert result.ok
    assert "類似研究も少し見ました" in result.message
    session = orchestrator.store.load()
    assert session.planning_scout["query"] == "research harness agent evaluation"
    entries = [
        json.loads(line)
        for line in config.journal_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["event_type"] == "planning_search_queries_generated" for entry in entries)


def test_custom_research_archive_dir_is_used(tmp_path):
    archive_dir = tmp_path / "my-research-archive"
    config = HarnessConfig(project_root=tmp_path / "runtime", research_archive_dir=archive_dir)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)

    discord.inject(orchestrator, "/re new")


    discord.inject(orchestrator, "/re plan")



    result = discord.inject_message(orchestrator, "archive location")
    assert result.ok
    session = orchestrator.store.load()
    assert session.version_label == "V001.0"
    assert session.research_dir.startswith(str(archive_dir))
    assert (archive_dir / f"{session.version_label}_{session.session_id}_untitled-research").exists()
    assert config.artifacts_dir(session.session_id).exists()
