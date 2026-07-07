import json

from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.modes import Mode
from harness.orchestrator import ResearchOrchestrator
from harness.papers import make_paper


def test_planning_scout_finds_similar_research_and_updates_brief(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)

    discord.inject(orchestrator, "/re new")


    discord.inject(orchestrator, "/re plan")



    discord.inject_message(orchestrator, "research harness reliability")
    message = orchestrator.scout_planning()
    assert "類似研究スカウト完了" in message
    assert "[P-001]" in message
    assert "Questions For Sora" in message

    session = orchestrator.store.load()
    assert session.mode == Mode.PLANNING
    assert session.planning_scout["similar_research"]
    assert session.planning_scout["novelty_status"] == "needs_human_decision"
    assert session.planning_scout["blocking"] is True
    assert session.planning_scout["primary_comparison"]["paper_id"] == "P-001"
    assert session.planning_scout["overlap_points"]
    assert session.planning_scout["differentiation_hypotheses"]
    assert session.planning_scout["weakness_points"]
    assert session.planning_scout["required_decisions"]
    brief = config.research_brief_path(session.session_id).read_text(encoding="utf-8")
    assert "## Similar Research Scout" in brief
    assert "Novelty Status" in brief
    assert "Primary Comparison" in brief
    assert "Required Decisions" in brief
    assert "[P-001]" in brief
    assert "差分" in brief

    entries = [
        json.loads(line)
        for line in config.journal_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["event_type"] == "planning_scout_started" for entry in entries)
    assert any(entry["event_type"] == "novelty_gate_evaluated" for entry in entries)
    assert any(entry["event_type"] == "phase_gate_requested" for entry in entries)
    assert any(entry["event_type"] == "planning_scout_completed" for entry in entries)
    assert any(entry.get("planning_scout") for entry in entries)


def test_planning_scout_is_planning_only(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)

    discord.inject(orchestrator, "/re new")


    discord.inject(orchestrator, "/re plan")



    discord.inject_message(orchestrator, "scout mode")
    session = orchestrator.store.load()
    session.mode = Mode.RESEARCH
    orchestrator.store.save(session)
    message = orchestrator.scout_planning()
    assert "PLANNING中だけ有効" in message


def test_novelty_gate_blocks_start_when_scout_is_uncertain(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)

    discord.inject(orchestrator, "/re new")


    discord.inject(orchestrator, "/re plan")



    discord.inject_message(orchestrator, "research harness reliability")
    orchestrator.scout_planning()
    result = discord.inject(orchestrator, "/re start")

    assert result.ok
    assert "Phase gateがpending" in result.message
    session = orchestrator.store.load()
    assert session.mode == Mode.PLANNING
    assert session.round_id == 0
    assert session.phase_gates["PG-1"].status == "pending"
    assert any("Phase gate" in message for message in discord.important_messages)


def test_phase_gate_accept_allows_blocking_scout_to_start(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)

    discord.inject(orchestrator, "/re new")


    discord.inject(orchestrator, "/re plan")



    discord.inject_message(orchestrator, "research harness reliability")
    orchestrator.scout_planning()
    accepted = discord.inject(orchestrator, "/re accept PG-1")
    assert accepted.ok
    assert "acceptしました" in accepted.message
    session = orchestrator.store.load()
    assert session.phase_gates["PG-1"].status == "accepted"

    started = discord.inject(orchestrator, "/re start")
    assert started.ok
    assert "Novelty gateで研究開始を保留" not in started.message


def test_phase_gate_revise_records_reason(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)

    discord.inject(orchestrator, "/re new")


    discord.inject(orchestrator, "/re plan")



    discord.inject_message(orchestrator, "research harness reliability")
    orchestrator.scout_planning()
    revised = discord.inject(orchestrator, "/re revise PG-1 比較対象を変える")
    assert revised.ok
    session = orchestrator.store.load()
    assert session.phase_gates["PG-1"].status == "revised"
    assert any("比較対象を変える" in item for item in session.redirects)
    entries = [
        json.loads(line)
        for line in config.journal_path(session.session_id).read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["event_type"] == "phase_revision_requested" for entry in entries)


def test_novelty_gate_allows_start_when_real_evidence_is_supported(tmp_path):
    config = HarnessConfig(project_root=tmp_path)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    orchestrator.paper_provider = RealEnoughProvider()

    discord.inject(orchestrator, "/re new")


    discord.inject(orchestrator, "/re plan")



    discord.inject_message(orchestrator, "agent novelty gates")
    scout_message = orchestrator.scout_planning()
    assert "類似研究スカウト完了" in scout_message
    session = orchestrator.store.load()
    assert session.planning_scout["novelty_status"] == "supported"
    assert session.planning_scout["blocking"] is False

    started = discord.inject(orchestrator, "/re start")
    assert started.ok
    session = orchestrator.store.load()
    assert session.mode in {Mode.RESEARCH, Mode.APPROVAL_BLOCKED, Mode.DONE}
    assert "Novelty gateで研究開始を保留" not in started.message


class RealEnoughProvider:
    name = "test-real"

    def search(self, query: str, max_results: int = 5):
        return [
            make_paper(
                title=f"Grounded Research Planning {index}",
                authors=["A. Researcher"],
                year=2024,
                venue="TestConf",
                url=f"https://example.test/paper-{index}",
                doi=f"10.0000/real-{index}",
                arxiv_id=None,
                abstract=(
                    "This study discusses research planning systems, literature review, "
                    "and human-in-the-loop novelty assessment."
                ),
                source=self.name,
                relevance_score=0.75,
                confidence="mid",
            )
            for index in range(1, 4)
        ][:max_results]
