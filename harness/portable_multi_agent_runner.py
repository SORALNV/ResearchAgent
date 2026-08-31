from __future__ import annotations

from harness.multi_agent_runner import MultiAgentRunner as HardenedMultiAgentRunner
from harness.multi_agent_types import AgentCommandExecutor, AgentInvocation
from harness.provider_executor import build_provider_executor_class


ProviderAwareAgentCommandExecutor = build_provider_executor_class(
    AgentCommandExecutor,
    AgentInvocation,
)


class MultiAgentRunner(HardenedMultiAgentRunner):
    """Hardened runner with runtime-provider routing.

    The inherited pipeline keeps checkpointing, review gates, artifact
    promotion, convergence, and cancellation. Only the executor is replaced,
    so switching between Codex CLI, OpenAI Responses, and optional Computer Use
    does not fork the orchestration logic.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.executor = ProviderAwareAgentCommandExecutor(
            config,
            self._lock,
            self.cancellation,
        )
