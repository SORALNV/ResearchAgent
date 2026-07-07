import json

from harness.config import HarnessConfig
from harness.conversation import ConversationSession
from harness.journal import Journal
from harness.state import ResearchSession


def test_bounded_conversation_stops_on_stagnation_and_returns_summary_only():
    session = ConversationSession(
        topic="想定外結果の原因分析",
        participants=["main", "sub"],
        max_turns=6,
        timeout_seconds=60,
    ).run_scripted(["A", "B", "B", "B", "C"])

    assert session.stop_condition == "stagnation"
    exported = session.to_journal_dict()
    assert exported["scratchpad"]
    assert exported["final_summary"]["問い"] == "想定外結果の原因分析"
    assert "結論" in exported["final_summary"]


def test_bounded_conversation_stops_on_max_turns():
    session = ConversationSession(
        topic="実験設計",
        participants=["main", "claude"],
        max_turns=2,
        timeout_seconds=60,
    ).run_scripted(["one", "two", "three"])

    assert session.stop_condition == "max_turns"
    assert len(session.scratchpad) == 2


def test_journal_masks_secret_like_values(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    session = ResearchSession.new("secret masking")
    journal = Journal(config.journal_path(session.session_id))
    journal.append(
        session,
        event_type="discord_command_received",
        user_instruction="api_key=abcdef1234567890abcdef1234567890",
    )
    entry = json.loads(config.journal_path(session.session_id).read_text(encoding="utf-8").strip())
    assert "abcdef1234567890abcdef1234567890" not in entry["user_instruction"]
    assert "***" in entry["user_instruction"]
