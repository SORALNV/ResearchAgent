from harness.agent_runner import SubAgentCommandRunner
from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.planning_dialogue import parse_plan_update
from harness.state import ResearchSession


def _session_with_dir(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    session = ResearchSession.new("動画埋め込み評価")
    version_label, research_dir = config.allocate_research_dir(session.session_id, session.research_goal)
    session.version_label = version_label
    session.research_dir = str(research_dir)
    return config, session


def test_goal_criteria_plan_update_replaces_full_list(tmp_path):
    orchestrator = ResearchOrchestrator(
        config=HarnessConfig(project_root=tmp_path),
        discord=FakeDiscordAdapter(),
    )
    session = ResearchSession.new("goal")
    session.goal_criteria = ["古い条件", "消える条件"]

    _, updates = parse_plan_update(
        """本文
PLAN_UPDATE:
GOAL_CRITERIA: C条件A
GOAL_CRITERIA: C条件B
"""
    )
    orchestrator._apply_plan_update(session, updates)

    assert session.goal_criteria == ["C条件A", "C条件B"]


def test_method_plan_update_replaces_full_list(tmp_path):
    orchestrator = ResearchOrchestrator(
        config=HarnessConfig(project_root=tmp_path),
        discord=FakeDiscordAdapter(),
    )
    session = ResearchSession.new("goal")
    session.research_methods = ["古い方法"]

    _, updates = parse_plan_update(
        """本文
PLAN_UPDATE:
METHOD: 新しい方法A
METHOD: 新しい方法B
"""
    )
    orchestrator._apply_plan_update(session, updates)

    assert session.research_methods == ["新しい方法A", "新しい方法B"]


def test_constraint_plan_update_appends_without_exact_duplicates(tmp_path):
    orchestrator = ResearchOrchestrator(
        config=HarnessConfig(project_root=tmp_path),
        discord=FakeDiscordAdapter(),
    )
    session = ResearchSession.new("goal")

    _, updates = parse_plan_update(
        """本文
PLAN_UPDATE:
CONSTRAINT: 認証はADCを使う
CONSTRAINT: 認証はADCを使う
CONSTRAINT: GCPプロジェクトID: embedding-2-490304
"""
    )
    orchestrator._apply_plan_update(session, updates)
    orchestrator._apply_plan_update(session, updates)

    assert session.constraints == [
        "認証はADCを使う",
        "GCPプロジェクトID: embedding-2-490304",
    ]


def test_project_name_updates_default_only(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    session = ResearchSession.new("goal")
    session.version_label = "V001.0"
    untitled = tmp_path / "research_runs" / f"{session.version_label}_{session.session_id}_untitled-research"
    session.research_dir = str(untitled)
    untitled.mkdir(parents=True)
    orchestrator = ResearchOrchestrator(
        config=config,
        discord=FakeDiscordAdapter(),
    )

    _, updates = parse_plan_update(
        """本文
PLAN_UPDATE:
PROJECT_NAME: 動画埋め込み評価
PROJECT_NAME: 最終研究名
"""
    )
    orchestrator._apply_plan_update(session, updates)
    assert session.project_name == "最終研究名"
    assert not session.research_dir.endswith("_untitled-research")
    assert session.research_dir.endswith("_research")

    _, later_updates = parse_plan_update(
        """本文
PLAN_UPDATE:
PROJECT_NAME: 勝手に変えない
"""
    )
    orchestrator._apply_plan_update(session, later_updates)
    assert session.project_name == "最終研究名"


def test_constraints_render_in_plan_md(tmp_path):
    config, session = _session_with_dir(tmp_path)
    session.project_name = "動画埋め込み評価"
    session.goal_criteria = ["比較表がある", "再現手順がある"]
    session.constraints = ["認証はADCを使う", "課金はGoogle Cloudクレジット範囲内"]
    orchestrator = ResearchOrchestrator(config=config, discord=FakeDiscordAdapter())

    plan_text = orchestrator._render_plan_md(session)

    assert "- project_name: 動画埋め込み評価" in plan_text
    assert "C1. 比較表がある" in plan_text
    assert "C2. 再現手順がある" in plan_text
    assert "制約:" in plan_text
    assert "- 認証はADCを使う" in plan_text
    assert "- 課金はGoogle Cloudクレジット範囲内" in plan_text


def test_constraints_render_in_sub_prompt(tmp_path):
    config, session = _session_with_dir(tmp_path)
    session.transition_to(Mode.RESEARCH)
    session.constraints = ["GCPプロジェクトID: embedding-2-490304"]
    config.plan_path(session.session_id).write_text("plan", encoding="utf-8")

    prompt = SubAgentCommandRunner(config)._build_prompt(
        session,
        round_number=1,
        task="実験する",
    )

    assert "実行時制約（PLANで確定。必ず従う）:" in prompt
    assert "- GCPプロジェクトID: embedding-2-490304" in prompt


def test_old_state_without_constraints_loads():
    session = ResearchSession.new("old state")
    data = session.to_dict()
    data.pop("constraints")

    loaded = ResearchSession.from_dict(data)

    assert loaded.constraints == []
