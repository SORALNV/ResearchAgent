import json
import shlex
import sys

from harness.command_parser import parse_research_command
from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.state import ResearchSession


def test_re_paper_routes_p_ids_to_paper_detail_and_others_to_write():
    detail = parse_research_command("/re paper P-001")
    assert detail.name == "paper"
    assert detail.args == {"paper_id": "P-001"}

    empty = parse_research_command("/re paper")
    assert empty.name == "paper_write"
    assert empty.args == {"instruction": ""}

    write = parse_research_command("/re paper 考察を厚くして")
    assert write.name == "paper_write"
    assert write.args == {"instruction": "考察を厚くして"}


def test_write_paper_rejects_without_sub_agent_command(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    session = ResearchSession.new("paper goal")
    session.transition_to(Mode.RESEARCH)
    orchestrator.store.save(session)

    message = orchestrator.write_paper("")

    assert "実サブエージェント未設定" in message


def test_write_paper_creates_thesis_records_journal_and_increments_round(tmp_path):
    script = (
        "from pathlib import Path; "
        "p=Path('paper/thesis.md'); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text('# タイトル\\n\\n## 概要\\n本文\\n', encoding='utf-8'); "
        "print('paper written')"
    )
    config = HarnessConfig(
        project_root=tmp_path,
        sub_agent_command=f"{sys.executable} -c {shlex.quote(script)}",
    )
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    session = ResearchSession.new("paper goal")
    session.transition_to(Mode.RESEARCH)
    orchestrator.store.save(session)

    message = orchestrator.write_paper("概要を更新")
    updated = orchestrator.store.load()
    assert updated is not None

    thesis_path = config.paper_dir(updated.session_id) / "thesis.md"
    assert thesis_path.exists()
    assert "保存先:" in message
    assert "文字数:" in message
    assert "# タイトル" in message
    assert updated.round_id == 1

    journal_path = config.journal_path(updated.session_id)
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("event_type") == "paper_written" for event in events)


class _RoundRunner:
    def run_round(self, session):
        from harness.agent_runner import RoundOutput

        return RoundOutput(
            main_agent_summary="mock round",
            subtask="continue",
            sub_agent_output="continued",
            review_output="reviewed",
            claude_consultation=None,
            fresh_agent_output=None,
            conversation_sessions=[],
            proposed_operation=None,
            accepted_ideas=[],
            rejected_ideas=[],
            decision="continue",
            confidence="high",
            next_action="next",
        )


def test_start_from_done_resumes_additional_rounds_until_done(tmp_path):
    config = HarnessConfig(project_root=tmp_path, max_rounds=2)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    orchestrator.runner = _RoundRunner()
    session = ResearchSession.new("resume goal")
    session.transition_to(Mode.RESEARCH)
    session.transition_to(Mode.DONE)
    session.round_id = 3
    session.completed_reason = "MAX_ROUNDS reached"
    orchestrator.store.save(session)

    message = orchestrator.start()
    updated = orchestrator.store.load()
    assert updated is not None

    assert updated.mode == Mode.DONE
    assert updated.round_budget == 5
    assert updated.round_id == 5
    assert updated.completed_reason == "MAX_ROUNDS reached"
    assert "研究ラウンドを完了した" in message

    journal_path = config.journal_path(updated.session_id)
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("event_type") == "research_resumed" for event in events)


class _DialogueResult:
    text = "了解しました。必要なら /re task <内容> を使います。"
    thread_id = "thread-1"
    error = ""
    searches = []


class _Dialogue:
    def __init__(self):
        self.calls = []

    def converse(self, session, text, search_callback, image_paths=None):
        self.calls.append((session.mode, text, image_paths))
        return _DialogueResult()


def test_plan_text_in_research_and_done_uses_dialogue(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    dialogue = _Dialogue()
    orchestrator.planning_dialogue = dialogue
    session = ResearchSession.new("dialogue goal")
    session.transition_to(Mode.RESEARCH)
    orchestrator.store.save(session)

    response = orchestrator.plan_text("今の状況は？")
    assert "了解しました" in response
    assert dialogue.calls
    assert "[研究状況]" in dialogue.calls[-1][1]

    session = orchestrator.store.load()
    assert session is not None
    session.transition_to(Mode.DONE)
    orchestrator.store.save(session)

    response = orchestrator.plan_text("論文は？")
    assert "了解しました" in response
    assert dialogue.calls[-1][0] == Mode.DONE


def test_plan_text_rejects_during_approval_blocked(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    session = ResearchSession.new("blocked goal")
    session.transition_to(Mode.RESEARCH)
    session.transition_to(Mode.APPROVAL_BLOCKED)
    orchestrator.store.save(session)

    response = orchestrator.plan_text("相談したい")

    assert "承認待ち" in response
