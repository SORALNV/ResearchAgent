from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from harness.config import HarnessConfig
from harness.kaggle_cli_transport import CurrentKaggleCliTransport
from harness.kaggle_submission import KaggleCommandResult
from main import build_routed_discord_service


def test_current_kaggle_cli_uses_positional_competition_reference(tmp_path: Path):
    calls: list[tuple[str, ...]] = []

    def runner(
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> KaggleCommandResult:
        del cwd, environment
        calls.append(tuple(command))
        if "submissions" in command:
            return KaggleCommandResult(
                0,
                "ref,date,description,status,publicScore,privateScore\n",
                "",
            )
        return KaggleCommandResult(0, "Submission ref: 123", "")

    submission = tmp_path / "submission.csv"
    submission.write_text("id,target\n1,0.5\n", encoding="utf-8")
    transport = CurrentKaggleCliTransport(
        command="kaggle",
        command_runner=runner,
    )

    transport.submission_history("demo-competition", cwd=tmp_path)
    transport.submit(
        competition_slug="demo-competition",
        file_path=submission,
        message="test",
        cwd=tmp_path,
    )

    assert calls[0] == (
        "kaggle",
        "competitions",
        "submissions",
        "demo-competition",
        "-v",
        "-q",
    )
    assert calls[1][:4] == (
        "kaggle",
        "competitions",
        "submit",
        "demo-competition",
    )
    assert "-c" not in calls[0]
    assert "-c" not in calls[1]


def test_routed_bot_wires_current_kaggle_transport(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("DISCORD_RESEARCH_CHANNEL_IDS", "100")
    monkeypatch.setenv("CONTROL_PLANE_DIR", str(tmp_path / "control-plane"))
    monkeypatch.setenv("COMPUTE_RUNTIME_DIR", str(tmp_path / "compute"))
    monkeypatch.setenv("FINAL_ACTION_RUNTIME_DIR", str(tmp_path / "final"))
    monkeypatch.setenv("PAPER_OUTPUT_DIR", str(tmp_path / "papers"))
    monkeypatch.setenv("LOCAL_PROCESS_COMPUTE_ENABLED", "false")

    service = build_routed_discord_service(
        HarnessConfig(project_root=tmp_path, paper_provider="fake")
    )

    assert isinstance(
        service.final_actions.submission.transport,
        CurrentKaggleCliTransport,
    )
