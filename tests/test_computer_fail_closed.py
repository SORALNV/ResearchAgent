from __future__ import annotations

import threading
from pathlib import Path

from harness.config import HarnessConfig
from harness.planning_dialogue import PlanningDialogueRunner
from harness.process_manager import ProcessCancellationController
from harness.provider_runtime import ProviderAwareAgentCommandExecutor
from harness.state import ResearchSession


class NeverCalledFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, settings):
        self.calls += 1
        raise AssertionError("OpenAI client must not be created for invalid safety settings")


def _session(tmp_path: Path) -> ResearchSession:
    session = ResearchSession.new("computer fail closed")
    research_dir = tmp_path / "run"
    research_dir.mkdir()
    session.research_dir = str(research_dir)
    return session


def _configure_computer(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_RUNTIME_ORDER", "openai_computer")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "text-model")
    monkeypatch.setenv("OPENAI_COMPUTER_ENABLED", "true")
    monkeypatch.setenv("OPENAI_COMPUTER_MODEL", "computer-model")
    monkeypatch.setenv("OPENAI_COMPUTER_BRIDGE_URL", "http://bridge")


def _executor(tmp_path: Path, factory: NeverCalledFactory):
    return ProviderAwareAgentCommandExecutor(
        HarnessConfig(project_root=tmp_path),
        threading.RLock(),
        ProcessCancellationController(0.1),
        openai_client_factory=factory,
    )


def test_computer_runtime_rejects_empty_stage_allowlist(tmp_path, monkeypatch):
    _configure_computer(monkeypatch)
    monkeypatch.delenv("OPENAI_COMPUTER_ALLOWED_STAGES", raising=False)
    monkeypatch.setenv("OPENAI_COMPUTER_REQUIRE_APPROVAL", "true")
    factory = NeverCalledFactory()
    executor = _executor(tmp_path, factory)

    invocation = executor.run(
        session=_session(tmp_path),
        role="main",
        stage="main_plan",
        prompt="inspect a GUI",
        command_text=None,
        sandbox="read-only",
    )

    assert invocation.skipped
    assert invocation.returncode == 126
    assert "OPENAI_COMPUTER_ALLOWED_STAGES" in invocation.output
    assert factory.calls == 0


def test_computer_runtime_rejects_disabled_approval_enforcement(
    tmp_path,
    monkeypatch,
):
    _configure_computer(monkeypatch)
    monkeypatch.setenv("OPENAI_COMPUTER_ALLOWED_STAGES", "main_plan")
    monkeypatch.setenv("OPENAI_COMPUTER_REQUIRE_APPROVAL", "false")
    factory = NeverCalledFactory()
    executor = _executor(tmp_path, factory)

    invocation = executor.run(
        session=_session(tmp_path),
        role="main",
        stage="main_plan",
        prompt="inspect a GUI",
        command_text=None,
        sandbox="read-only",
    )

    assert invocation.skipped
    assert invocation.returncode == 126
    assert "approval" in invocation.output.lower()
    assert factory.calls == 0


def test_planning_dialogue_uses_the_strict_provider_runtime(tmp_path):
    runner = PlanningDialogueRunner(HarnessConfig(project_root=tmp_path))

    assert isinstance(runner.executor, ProviderAwareAgentCommandExecutor)
