from __future__ import annotations

import json
from pathlib import Path

from harness.agent_runner import SubAgentCommandRunner
from harness.command_parser import parse_research_command
from harness.config import HarnessConfig
from harness.commands import Command
from harness.memory import AgentMemory
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.planning_dialogue import PlanningDialogueRunner, parse_plan_update
from harness.state import ResearchSession, SessionStore


class DummyDiscord:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str, channel: str = "default") -> None:
        self.messages.append(message)


class StubPlanningDialogue:
    def converse(self, session, user_text, search_callback, image_paths=None):
        from harness.planning_dialogue import DialogueTurnResult

        return DialogueTurnResult(
            text="了解です。\n\nPLAN_UPDATE:\nREMEMBER: ユーザーは日本語の短い返答を好む",
            thread_id="thread-1",
            searches=[],
            error="",
        )


def make_config(tmp_path: Path) -> HarnessConfig:
    return HarnessConfig(
        project_root=tmp_path,
        research_archive_dir=tmp_path / "research_runs",
        agent_memory_dir=tmp_path / "agent_memory",
        sub_agent_command="mock",
    )


def make_orchestrator(tmp_path: Path) -> ResearchOrchestrator:
    config = make_config(tmp_path)
    return ResearchOrchestrator(config=config, discord=DummyDiscord())


def make_session(tmp_path: Path, config: HarnessConfig | None = None) -> ResearchSession:
    config = config or make_config(tmp_path)
    session = ResearchSession.new("memory test goal")
    version, research_dir = config.allocate_research_dir(session.session_id, session.research_goal)
    session.version_label = version
    session.research_dir = str(research_dir)
    SessionStore(config.state_path).save(session)
    return session


def test_agent_memory_save_and_digest(tmp_path: Path) -> None:
    memory = AgentMemory(tmp_path / "agent_memory")

    assert memory.digest() == "記憶はまだありません。"

    entry = memory.save("Gemini embedding video", "動画埋め込みではGeminiを比較対象にする。")

    assert entry.exists()
    assert entry.parent.name == "entries"
    assert "動画埋め込み" in entry.read_text(encoding="utf-8")

    index = (tmp_path / "agent_memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "- [Gemini embedding video](entries/" in index
    assert "動画埋め込みではGeminiを比較対象にする。" in memory.digest()


def test_re_remember_creates_memory_and_journal_event(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(tmp_path)
    session = make_session(tmp_path, orchestrator.config)

    command = parse_research_command("/re remember 常に日本語で短く返す")
    result = orchestrator.handle(command)

    assert result.ok
    assert "記憶しました" in result.message
    entries = list((tmp_path / "agent_memory" / "entries").glob("*.md"))
    assert len(entries) == 1

    journal = Path(session.research_dir) / "journal.jsonl"
    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert any(event.get("event_type") == "memory_saved" for event in events)


def test_plan_update_remember_is_saved(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(tmp_path)
    session = make_session(tmp_path, orchestrator.config)
    session.phase = "plan"
    SessionStore(orchestrator.config.state_path).save(session)
    orchestrator.planning_dialogue = StubPlanningDialogue()

    result = orchestrator.handle(Command("plan_text", {"text": "好みを覚えて", "attachments": []}))

    assert result.ok
    assert "記憶+1" in result.message
    entries = list((tmp_path / "agent_memory" / "entries").glob("*.md"))
    assert len(entries) == 1
    assert "ユーザーは日本語の短い返答を好む" in entries[0].read_text(encoding="utf-8")


def test_parse_plan_update_remember() -> None:
    text, updates = parse_plan_update(
        "本文\n\nPLAN_UPDATE:\nREMEMBER: ユーザーは日本語を好む\nGOAL_CRITERIA: 成功条件"
    )

    assert text == "本文"
    assert updates["remembers"] == ["ユーザーは日本語を好む"]
    assert updates["goal_criteria"] == ["成功条件"]


def test_stop_saves_session_summary_once(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(tmp_path)
    session = make_session(tmp_path, orchestrator.config)
    session.mode = Mode.RESEARCH
    session.goal_criteria.append("再現可能な実験ログがある")
    session.round_id = 2
    SessionStore(orchestrator.config.state_path).save(session)

    experiments_path = orchestrator.config.experiments_path(session.session_id)
    experiments_path.write_text(
        json.dumps(
            {
                "exp_id": "EXP-001",
                "name": "baseline",
                "status": "done",
                "metrics": {"score": 0.9},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    orchestrator.stop(quiet=True)
    orchestrator.stop(quiet=True)

    entries = list((tmp_path / "agent_memory" / "entries").glob("*.md"))
    assert len(entries) == 1
    body = entries[0].read_text(encoding="utf-8")
    assert "goal: memory test goal" in body
    assert "EXP-001" in body


def test_plan_initial_prompt_includes_memory_digest(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    AgentMemory(config.agent_memory_path).save("pref", "ユーザーは比較表を好む")
    session = ResearchSession.new("prompt memory goal")

    prompt = PlanningDialogueRunner(config)._build_initial_prompt(session, "相談したい")

    assert "長期記憶（過去の研究・ユーザーの好み）:" in prompt
    assert "ユーザーは比較表を好む" in prompt


def test_sub_prompt_includes_memory_digest(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    session = make_session(tmp_path, config)
    AgentMemory(config.agent_memory_path).save("env", "GPUはJetson上で使う")
    runner = SubAgentCommandRunner(config)

    prompt = runner._build_prompt(session, "実験計画を作る", 1)

    assert "長期記憶（過去の研究・ユーザーの好み）:" in prompt
    assert "GPUはJetson上で使う" in prompt
