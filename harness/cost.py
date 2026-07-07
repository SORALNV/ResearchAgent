from __future__ import annotations

from dataclasses import dataclass

from harness.approval import ProposedOperation
from harness.config import HarnessConfig
from harness.state import ResearchSession


@dataclass(frozen=True)
class CostCheck:
    ok: bool
    reason: str | None = None
    used: int = 0
    limit: int = 0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


class CostManager:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def record_literature_search(self, session: ResearchSession, query: str, result_text: str) -> CostCheck:
        session.cost.api_calls += 1
        session.cost.literature_searches += 1
        session.cost.estimated_tokens += estimate_tokens(query) + estimate_tokens(result_text)
        return self.check(session)

    def check(self, session: ResearchSession) -> CostCheck:
        if self.config.max_agent_calls > 0 and session.cost.agent_calls >= self.config.max_agent_calls:
            return CostCheck(
                ok=False,
                reason="MAX_AGENT_CALLS reached",
                used=session.cost.agent_calls,
                limit=self.config.max_agent_calls,
            )
        if self.config.max_api_calls > 0 and session.cost.api_calls >= self.config.max_api_calls:
            return CostCheck(
                ok=False,
                reason="MAX_API_CALLS reached",
                used=session.cost.api_calls,
                limit=self.config.max_api_calls,
            )
        if self.config.max_total_tokens > 0 and session.cost.estimated_tokens >= self.config.max_total_tokens:
            return CostCheck(
                ok=False,
                reason="MAX_TOTAL_TOKENS reached",
                used=session.cost.estimated_tokens,
                limit=self.config.max_total_tokens,
            )
        return CostCheck(ok=True)

    def make_limit_operation(self, check: CostCheck) -> ProposedOperation:
        return ProposedOperation(
            operation=f"cost_limit: {check.reason}",
            reason="コスト上限に到達したため自動続行を停止する。",
            impact="MVPでは実外部操作を行わず、APPROVAL_BLOCKEDで停止する。",
            dry_run_result=f"{check.reason}: {check.used} / {check.limit}",
        )
