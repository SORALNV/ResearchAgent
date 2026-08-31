from __future__ import annotations

from dataclasses import dataclass

from harness.platform.models import Domain, Project, WorkSession
from harness.platform.registry import PlatformRegistry
from harness.runtime.base import (
    RuntimeCapability,
    RuntimeContext,
    RuntimeRequest,
    RuntimeResult,
)
from harness.runtime.openai_responses import OpenAIResponsesRuntime
from harness.runtime.router import AgentRuntimeRouter
from harness.runtime.tools import ToolExecutionContext, build_default_harness_tools


@dataclass
class FakeRuntime:
    name: str
    capabilities: frozenset[RuntimeCapability]
    fail: bool = False
    approval: bool = False
    calls: int = 0

    def available(self):
        return True, "fake"

    def run(self, request):
        self.calls += 1
        return RuntimeResult(
            runtime=self.name,
            model="fake",
            output_text=f"handled by {self.name}",
            requires_approval=self.approval,
            returncode=1 if self.fail else 0,
            error="failed" if self.fail else None,
        )

    def cancel(self, reason="cancel requested"):
        return 0


def test_router_prefers_codex_for_mutating_work_and_openai_for_tools():
    codex = FakeRuntime(
        "codex_cli",
        frozenset(
            {
                RuntimeCapability.CHAT,
                RuntimeCapability.REASONING,
                RuntimeCapability.CODING,
                RuntimeCapability.FILE_EDIT,
                RuntimeCapability.SHELL,
            }
        ),
    )
    openai = FakeRuntime(
        "openai_responses",
        frozenset(
            {
                RuntimeCapability.CHAT,
                RuntimeCapability.REASONING,
                RuntimeCapability.FUNCTION_TOOLS,
                RuntimeCapability.COMPUTER_USE,
            }
        ),
    )
    router = AgentRuntimeRouter([openai, codex])
    coding = router.run(
        RuntimeRequest(
            prompt="implement",
            capabilities=(
                RuntimeCapability.CHAT,
                RuntimeCapability.REASONING,
                RuntimeCapability.CODING,
                RuntimeCapability.FILE_EDIT,
            ),
        )
    )
    assert coding.runtime == "codex_cli"
    tool_request = router.run(
        RuntimeRequest(
            prompt="plan",
            capabilities=(
                RuntimeCapability.CHAT,
                RuntimeCapability.REASONING,
                RuntimeCapability.FUNCTION_TOOLS,
            ),
        )
    )
    assert tool_request.runtime == "openai_responses"


def test_mutating_request_does_not_fallback_after_codex_failure():
    codex = FakeRuntime(
        "codex_cli",
        frozenset(
            {
                RuntimeCapability.CODING,
                RuntimeCapability.FILE_EDIT,
                RuntimeCapability.CHAT,
                RuntimeCapability.REASONING,
            }
        ),
        fail=True,
    )
    fallback = FakeRuntime(
        "openai_responses",
        frozenset(
            {
                RuntimeCapability.CODING,
                RuntimeCapability.FILE_EDIT,
                RuntimeCapability.CHAT,
                RuntimeCapability.REASONING,
            }
        ),
    )
    result = AgentRuntimeRouter([codex, fallback]).run(
        RuntimeRequest(
            prompt="change files",
            capabilities=(
                RuntimeCapability.CODING,
                RuntimeCapability.FILE_EDIT,
            ),
        )
    )
    assert not result.ok
    assert codex.calls == 1
    assert fallback.calls == 0


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def create(self, payload):
        self.payloads.append(dict(payload))
        return self.responses.pop(0)


def test_openai_runtime_executes_only_typed_harness_tools(tmp_path):
    registry = PlatformRegistry(tmp_path / "platform.sqlite3")
    project = registry.create_project(
        Project.new(domain=Domain.RESEARCH, title="tool-loop")
    )
    session = registry.create_work_session(
        WorkSession.new(
            project_id=project.project_id,
            title="session",
            objective="schedule a smoke test",
        )
    )
    transport = FakeTransport(
        [
            {
                "id": "resp-1",
                "model": "fake-model",
                "output": [
                    {
                        "type": "function_call",
                        "name": "propose_job",
                        "call_id": "call-1",
                        "arguments": "{\"work_session_id\":\"%s\",\"domain\":\"research\",\"task_type\":\"smoke_test\",\"entrypoint\":\"python run.py\",\"backend_preferences\":[\"fake\"],\"resources\":{\"accelerator\":\"cpu\",\"min_vram_gb\":0,\"preferred_gpu_count\":0,\"cpu_cores\":1,\"ram_gb\":2,\"max_runtime_minutes\":5,\"network_required\":false,\"capabilities\":[]},\"inputs\":{},\"outputs\":[\"result.json\"],\"metadata\":{}}"
                        % session.session_id,
                    }
                ],
            },
            {
                "id": "resp-2",
                "model": "fake-model",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Jobを作成しました。",
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 4},
            },
        ]
    )
    tools = build_default_harness_tools()
    runtime = OpenAIResponsesRuntime(
        model="fake-model",
        transport=transport,
        tools=tools,
        tool_context_factory=lambda request: ToolExecutionContext(
            registry=registry,
            project_id=project.project_id,
            work_session_id=session.session_id,
            actor="test",
        ),
    )
    result = runtime.run(
        RuntimeRequest(
            prompt="smoke testを登録",
            capabilities=(
                RuntimeCapability.CHAT,
                RuntimeCapability.REASONING,
                RuntimeCapability.FUNCTION_TOOLS,
            ),
            context=RuntimeContext(
                project_id=project.project_id,
                work_session_id=session.session_id,
            ),
        )
    )
    assert result.ok
    assert result.output_text == "Jobを作成しました。"
    jobs = registry.list_jobs(work_session_id=session.session_id)
    assert len(jobs) == 1
    assert jobs[0].spec.task_type == "smoke_test"
    assert transport.payloads[1]["previous_response_id"] == "resp-1"
    assert transport.payloads[1]["input"][0]["type"] == "function_call_output"


def test_computer_use_is_fail_closed_before_explicit_approval():
    transport = FakeTransport([])
    runtime = OpenAIResponsesRuntime(
        model="fake-model",
        transport=transport,
        computer_tool={"type": "computer"},
    )
    result = runtime.run(
        RuntimeRequest(
            prompt="open a browser",
            capabilities=(RuntimeCapability.COMPUTER_USE,),
            computer_use_allowed=False,
        )
    )
    assert result.requires_approval
    assert result.pending_actions[0]["type"] == "computer_use_session"
    assert transport.payloads == []
