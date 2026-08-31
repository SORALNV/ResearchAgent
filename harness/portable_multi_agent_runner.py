from __future__ import annotations

from harness.multi_agent_runner import MultiAgentRunner as HardenedMultiAgentRunner
from harness.provider_runtime import ProviderAwareAgentCommandExecutor


class MultiAgentRunner(HardenedMultiAgentRunner):
    """Hardened runner with runtime-provider routing.

    The inherited pipeline keeps checkpointing, review gates, artifact
    promotion, convergence, and cancellation. Only the executor is replaced,
    so switching between Codex CLI, OpenAI Responses, and optional Computer Use
    does not fork orchestration logic. Workspace-write stages remain restricted
    to local CLI providers.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.executor = ProviderAwareAgentCommandExecutor(
            config,
            self._lock,
            self.cancellation,
        )
