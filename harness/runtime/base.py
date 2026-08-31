from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class RuntimeCapability(StrEnum):
    CHAT = "chat"
    REASONING = "reasoning"
    CODING = "coding"
    FILE_EDIT = "file_edit"
    SHELL = "shell"
    FUNCTION_TOOLS = "function_tools"
    COMPUTER_USE = "computer_use"
    VISION = "vision"


@dataclass(frozen=True)
class RuntimeContext:
    project_id: str | None = None
    work_session_id: str | None = None
    job_id: str | None = None
    research_session_id: str | None = None
    role: str = "assistant"
    stage: str = "conversation"
    working_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeRequest:
    prompt: str
    system_prompt: str = ""
    capabilities: tuple[RuntimeCapability, ...] = (
        RuntimeCapability.CHAT,
        RuntimeCapability.REASONING,
    )
    preferred_runtime: str | None = None
    model: str | None = None
    context: RuntimeContext = field(default_factory=RuntimeContext)
    tools_enabled: bool = True
    computer_use_allowed: bool = False
    max_tool_rounds: int = 8
    max_output_chars: int = 20000
    response_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def requires(self, capability: RuntimeCapability | str) -> bool:
        normalized = RuntimeCapability(str(capability))
        return normalized in self.capabilities


@dataclass(frozen=True)
class RuntimeResult:
    runtime: str
    model: str | None
    output_text: str
    response_id: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    pending_actions: tuple[dict[str, Any], ...] = ()
    requires_approval: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    duration_seconds: float = 0.0
    returncode: int = 0
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None and not self.requires_approval

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "model": self.model,
            "output_text": self.output_text,
            "response_id": self.response_id,
            "tool_calls": list(self.tool_calls),
            "tool_results": list(self.tool_results),
            "pending_actions": list(self.pending_actions),
            "requires_approval": self.requires_approval,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "duration_seconds": self.duration_seconds,
            "returncode": self.returncode,
            "error": self.error,
            "raw": self.raw,
        }


@runtime_checkable
class AgentRuntime(Protocol):
    name: str
    capabilities: frozenset[RuntimeCapability]

    def available(self) -> tuple[bool, str]:
        """Return whether the runtime can currently accept work."""

    def run(self, request: RuntimeRequest) -> RuntimeResult:
        """Run one request synchronously.

        Long-running runtimes are called from the Core worker rather than the
        Discord event loop. Durable jobs should use ComputeBackend instead.
        """

    def cancel(self, reason: str = "cancel requested") -> int:
        """Cancel active work and return the number of signalled processes."""
