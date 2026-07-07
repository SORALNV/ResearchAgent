from harness.config import HarnessConfig
from harness.discord_adapter import FakeDiscordAdapter
from harness.orchestrator import ResearchOrchestrator
from harness.papers import make_paper


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
