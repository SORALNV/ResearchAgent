from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.approval import ApprovalGate, ProposedOperation
from harness.config import HarnessConfig
from harness.process_manager import ProcessCancellationController
from harness.provider_runtime import ProviderAwareAgentCommandExecutor as Executor
from harness.state import ResearchSession


@dataclass
class FakeUsage:
    input_tokens: int = 11
    output_tokens: int = 7


class FakeAction:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def model_dump(self, exclude_none: bool = True):
        return dict(self.value)


class FakeComputerCall:
    type = "computer_call"

    def __init__(self) -> None:
        self.call_id = "call-1"
        self.pending_safety_checks = []
        self.action = FakeAction({"type": "screenshot"})
        self.actions = None


class FakeResponse:
    def __init__(
        self,
        *,
        response_id: str,
        output_text: str = "",
        output: list[object] | None = None,
        usage: FakeUsage | None = None,
    ) -> None:
        self.id = response_id
        self.output_text = output_text
        self.output = output or []
        self.usage = usage or FakeUsage()

    def model_dump_json(self) -> str:
        return '{"id":"%s"}' % self.id


class FakeResponses:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = FakeResponses(responses)


def make_session(tmp_path: Path) -> ResearchSession:
    session = ResearchSession.new("provider routing")
    research_dir = tmp_path / "run"
    research_dir.mkdir()
    session.research_dir = str(research_dir)
    return session


def make_executor(
    tmp_path: Path,
    client: FakeClient,
    *,
    bridge_request=None,
):
    config = HarnessConfig(
        project_root=tmp_path,
        max_command_seconds=2,
        agent_allow_unsandboxed_generic=True,
    )
    return Executor(
        config,
        __import__("threading").RLock(),
        ProcessCancellationController(0.1),
        openai_client_factory=lambda settings: client,
        bridge_request=bridge_request,
    )


def configure_openai(monkeypatch, order: str = "openai_responses") -> None:
    monkeypatch.setenv("AGENT_RUNTIME_ORDER", order)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")


def test_openai_responses_provider_records_usage_and_runtime_events(
    tmp_path,
    monkeypatch,
):
    configure_openai(monkeypatch)
    client = FakeClient(
        [FakeResponse(response_id="resp-1", output_text='{"result":"ok"}')]
    )
    executor = make_executor(tmp_path, client)
    session = make_session(tmp_path)

    invocation = executor.run(
        session=session,
        role="review",
        stage="review",
        prompt="review this",
        command_text=None,
        sandbox="read-only",
    )

    assert invocation.ok
    assert invocation.output == '{"result":"ok"}'
    assert invocation.command[0] == "provider:openai_responses"
    assert session.cost.agent_calls == 1
    assert session.cost.estimated_tokens == 18
    assert client.responses.calls[0]["model"] == "test-model"
    event_log = Path(session.research_dir) / "artifacts" / "runtime_events.jsonl"
    assert event_log.exists()
    assert "provider_finished" in event_log.read_text(encoding="utf-8")


def test_missing_codex_falls_back_to_openai_responses(
    tmp_path,
    monkeypatch,
):
    configure_openai(monkeypatch, "codex_cli,openai_responses")
    monkeypatch.setenv("PATH", "")
    client = FakeClient(
        [FakeResponse(response_id="resp-2", output_text="fallback-ok")]
    )
    executor = make_executor(tmp_path, client)
    session = make_session(tmp_path)

    invocation = executor.run(
        session=session,
        role="main",
        stage="main_plan",
        prompt="make a plan",
        command_text="codex",
        sandbox="read-only",
    )

    assert invocation.ok
    assert invocation.output == "fallback-ok"
    assert invocation.command[0] == "provider:openai_responses"
    assert "provider=codex_cli returncode=127" in invocation.stderr
    assert session.cost.agent_calls == 2


def test_computer_provider_requires_harness_approval(
    tmp_path,
    monkeypatch,
):
    configure_openai(monkeypatch, "openai_computer")
    monkeypatch.setenv("OPENAI_COMPUTER_ENABLED", "true")
    monkeypatch.setenv("OPENAI_COMPUTER_MODEL", "computer-model")
    monkeypatch.setenv("OPENAI_COMPUTER_BRIDGE_URL", "http://bridge")
    client = FakeClient([])
    executor = make_executor(tmp_path, client)
    session = make_session(tmp_path)

    invocation = executor.run(
        session=session,
        role="main",
        stage="main_plan",
        prompt="inspect a web application",
        command_text=None,
        sandbox="read-only",
    )

    assert not invocation.ok
    assert "APPROVAL_REQUIRED:" in invocation.output
    assert "openai_computer_use:main:main_plan:global" in invocation.output
    assert not client.responses.calls


def test_approved_computer_provider_executes_bridge_loop(
    tmp_path,
    monkeypatch,
):
    configure_openai(monkeypatch, "openai_computer")
    monkeypatch.setenv("OPENAI_COMPUTER_ENABLED", "true")
    monkeypatch.setenv("OPENAI_COMPUTER_MODEL", "computer-model")
    monkeypatch.setenv("OPENAI_COMPUTER_BRIDGE_URL", "http://bridge")
    client = FakeClient(
        [
            FakeResponse(
                response_id="resp-cu-1",
                output=[FakeComputerCall()],
            ),
            FakeResponse(
                response_id="resp-cu-2",
                output_text="computer-finished",
            ),
        ]
    )
    bridge_calls: list[dict[str, object]] = []

    def bridge(settings, payload):
        bridge_calls.append(dict(payload))
        return {"screenshot_data_url": "data:image/png;base64,AA=="}

    executor = make_executor(tmp_path, client, bridge_request=bridge)
    session = make_session(tmp_path)
    operation = ProposedOperation(
        operation="openai_computer_use:main:main_plan:global"
    )
    gate = ApprovalGate()
    request = gate.create_request(session, operation)
    gate.approve(session, request.approval_id)

    invocation = executor.run(
        session=session,
        role="main",
        stage="main_plan",
        prompt="inspect a web application",
        command_text=None,
        sandbox="read-only",
    )

    assert invocation.ok
    assert invocation.output == "computer-finished"
    assert len(bridge_calls) == 1
    assert bridge_calls[0]["action"] == {"type": "screenshot"}
    assert client.responses.calls[0]["tools"] == [{"type": "computer"}]
    second_input = client.responses.calls[1]["input"][0]
    assert second_input["type"] == "computer_call_output"
    assert second_input["output"]["type"] == "computer_screenshot"


def test_openai_text_provider_is_not_used_for_workspace_write(
    tmp_path,
    monkeypatch,
):
    configure_openai(monkeypatch)
    client = FakeClient(
        [FakeResponse(response_id="resp-3", output_text="should-not-run")]
    )
    executor = make_executor(tmp_path, client)
    session = make_session(tmp_path)

    invocation = executor.run(
        session=session,
        role="sub",
        stage="sub_execute",
        prompt="write code",
        command_text=None,
        sandbox="workspace-write",
    )

    assert not invocation.ok
    assert invocation.skipped
    assert "workspace-write" in invocation.stderr
    assert not client.responses.calls
