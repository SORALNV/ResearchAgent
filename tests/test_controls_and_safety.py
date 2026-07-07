from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
import json


def make_orchestrator(tmp_path):
    config = HarnessConfig(project_root=tmp_path, max_rounds=2, fresh_interval=1)
    return ResearchOrchestrator(config=config, discord=FakeDiscordAdapter()), config


def test_pause_resume_status_redirect_idea(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)
    orchestrator.goal("制御コマンドを検証する")

    paused = orchestrator.pause()
    assert "モード: PAUSED" in paused
    resumed = orchestrator.resume()
    assert "モード: PLANNING" in resumed

    idea = orchestrator.idea("失敗時再開を早めに検証する")
    assert "ideaを記録しました" in idea
    status = orchestrator.status()
    assert "承認待ち: なし" in status

    plan = orchestrator.redirect("外部ネットワークはMVPで使わない")
    assert "外部ネットワークはMVPで使わない" in plan
    session = orchestrator.store.load()
    assert session.mode == Mode.PLANNING
    assert "失敗時再開を早めに検証する" in session.accepted_ideas


def test_reject_returns_to_planning(tmp_path):
    orchestrator, _ = make_orchestrator(tmp_path)
    orchestrator.goal("承認却下を検証する")
    orchestrator.accept_phase_gate("PG-1")
    orchestrator.start()
    rejected = orchestrator.reject("AP-1", "削除系操作はまだ許可しない")
    assert "モード: PLANNING" in rejected
    session = orchestrator.store.load()
    assert session.mode == Mode.PLANNING
    assert session.approval_requests["AP-1"].status.startswith("rejected")
    entries = [
        json.loads(line)
        for line in orchestrator.config.journal_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["event_type"] == "approval_rejected" for entry in entries)
