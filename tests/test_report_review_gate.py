from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.orchestrator import ResearchOrchestrator
from harness.papers import make_paper
from harness.reporting import review_report
from harness.state import ResearchSession


def test_review_gate_is_not_created_when_report_review_has_no_warnings(tmp_path):
    config = HarnessConfig(project_root=tmp_path, max_rounds=1)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    orchestrator.paper_provider = RealEnoughProvider()

    discord.inject(orchestrator, "/re new")


    discord.inject(orchestrator, "/re plan")



    discord.inject_message(orchestrator, "sourced report")
    orchestrator.scout_planning()
    discord.inject(orchestrator, "/re start")
    discord.inject(orchestrator, "/re approve AP-1")
    discord.inject(orchestrator, "/re stop")

    session = orchestrator.store.load()
    assert not any(gate.phase == "review" for gate in session.phase_gates.values())


def test_stop_preserves_existing_report_and_writes_harness_report(tmp_path):
    config = HarnessConfig(project_root=tmp_path, max_rounds=1)
    discord = FakeDiscordAdapter()
    orchestrator = ResearchOrchestrator(config=config, discord=discord)
    orchestrator.paper_provider = RealEnoughProvider()

    discord.inject(orchestrator, "/re new")
    discord.inject(orchestrator, "/re plan")
    discord.inject_message(orchestrator, "preserve report")
    orchestrator.scout_planning()
    discord.inject(orchestrator, "/re start")
    discord.inject(orchestrator, "/re approve AP-1")

    session = orchestrator.store.load()
    report_path = config.report_path(session.session_id)
    figures_dir = report_path.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "ok.png").write_bytes(b"png")
    existing_report = "# Existing Report\n\n![ok](figures/ok.png)\n\n未確認\n"
    report_path.write_text(existing_report, encoding="utf-8")

    discord.inject(orchestrator, "/re stop")

    assert report_path.read_text(encoding="utf-8") == existing_report
    harness_report_path = report_path.parent / "harness_report.md"
    assert harness_report_path.exists()
    assert "# Research Report" in harness_report_path.read_text(encoding="utf-8")
    journal = config.journal_path(session.session_id).read_text(encoding="utf-8")
    assert "report_preserved" in journal


def test_review_report_warns_for_missing_image_reference(tmp_path):
    session = ResearchSession.new("image review")
    result = review_report(
        "# Report\n\n![plot](figures/missing.png)\n\n未確認\n",
        [],
        session,
        research_dir=tmp_path,
    )
    assert "存在しない画像を参照しています: figures/missing.png" in result["warnings"]


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
                doi=f"10.0000/report-real-{index}",
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
