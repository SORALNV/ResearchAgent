from __future__ import annotations

from pathlib import Path

from harness.cost import estimate_tokens
from harness.multi_agent_types import AgentCommandExecutor, AgentInvocation
from harness.provider_executor import (
    RuntimeAttempt,
    ProviderAwareAgentCommandExecutor as ProviderExecutorBase,
    _decorate_invocation,
    _local_retryable,
)


class ProviderAwareAgentCommandExecutor(ProviderExecutorBase):
    """Concrete provider executor with strict local and GUI safety boundaries.

    Plain Responses API calls can reason and review but cannot mutate the local
    task workspace. Only local CLI providers (normally Codex CLI) are eligible
    for ``workspace-write`` until a sandboxed file-tool provider is added.

    Computer Use is also fail-closed here rather than relying on ``doctor``:
    a non-empty stage allowlist and mandatory harness approval are required at
    execution time.
    """

    def __init__(
        self,
        config,
        lock,
        cancellation,
        *,
        openai_client_factory=None,
        bridge_request=None,
    ) -> None:
        super().__init__(
            config,
            lock,
            cancellation,
            local_executor_cls=AgentCommandExecutor,
            invocation_cls=AgentInvocation,
            openai_client_factory=openai_client_factory,
            bridge_request=bridge_request,
        )

    def _run_openai_computer(
        self,
        *,
        session,
        role: str,
        stage: str,
        prompt: str,
        task_id: str | None,
    ) -> RuntimeAttempt:
        if not self.settings.computer_allowed_stages:
            return RuntimeAttempt(
                provider="openai_computer",
                output=(
                    "OpenAI computer provider is blocked: "
                    "OPENAI_COMPUTER_ALLOWED_STAGES is empty."
                ),
                stderr="a non-empty computer stage allowlist is required",
                returncode=126,
                duration_seconds=0.0,
                skipped=True,
                retryable=False,
            )
        if not self.settings.computer_require_approval:
            return RuntimeAttempt(
                provider="openai_computer",
                output=(
                    "OpenAI computer provider is blocked: harness approval "
                    "enforcement must remain enabled."
                ),
                stderr="OPENAI_COMPUTER_REQUIRE_APPROVAL must be true",
                returncode=126,
                duration_seconds=0.0,
                skipped=True,
                retryable=False,
            )
        return super()._run_openai_computer(
            session=session,
            role=role,
            stage=stage,
            prompt=prompt,
            task_id=task_id,
        )

    def run(
        self,
        *,
        session,
        role: str,
        stage: str,
        prompt: str,
        command_text: str | None,
        sandbox: str,
        task_id: str | None = None,
        working_dir: Path | None = None,
    ) -> AgentInvocation:
        if sandbox != "workspace-write":
            return super().run(
                session=session,
                role=role,
                stage=stage,
                prompt=prompt,
                command_text=command_text,
                sandbox=sandbox,
                task_id=task_id,
                working_dir=working_dir,
            )

        chain = self.settings.order_for(role, command_text)
        local_chain = tuple(
            provider for provider in chain if provider in {"cli", "codex_cli"}
        )
        if not local_chain:
            return AgentInvocation(
                role=role,
                stage=stage,
                task_id=task_id,
                command=(),
                output=(
                    "No workspace-capable provider is configured. "
                    "Use Codex CLI for workspace-write stages."
                ),
                stderr=(
                    "openai_responses and openai_computer are not local "
                    "workspace-write providers"
                ),
                returncode=126,
                duration_seconds=0.0,
                skipped=True,
                workspace=str(working_dir) if working_dir else None,
                estimated_input_tokens=estimate_tokens(prompt),
            )

        attempts: list[RuntimeAttempt] = []
        final: AgentInvocation | None = None
        for index, provider in enumerate(local_chain, 1):
            self._record_event(
                session,
                {
                    "event": "provider_started",
                    "provider": provider,
                    "role": role,
                    "stage": stage,
                    "task_id": task_id,
                    "attempt": index,
                },
            )
            selected_command = command_text
            if provider == "codex_cli" and not _is_codex_command(command_text):
                selected_command = "codex"
            invocation = self.local.run(
                session=session,
                role=role,
                stage=stage,
                prompt=prompt,
                command_text=selected_command,
                sandbox=sandbox,
                task_id=task_id,
                working_dir=working_dir,
            )
            attempt = RuntimeAttempt(
                provider=provider,
                output=invocation.output,
                stderr=invocation.stderr,
                returncode=invocation.returncode,
                duration_seconds=invocation.duration_seconds,
                skipped=invocation.skipped,
                timed_out=invocation.timed_out,
                cancelled=invocation.cancelled,
                retryable=_local_retryable(invocation),
                input_tokens=invocation.estimated_input_tokens,
                output_tokens=invocation.estimated_output_tokens,
            )
            attempts.append(attempt)
            final = _decorate_invocation(invocation, provider, attempts)
            self._record_event(
                session,
                {
                    "event": "provider_finished",
                    "role": role,
                    "stage": stage,
                    "task_id": task_id,
                    "attempt": index,
                    **attempt.to_event(),
                },
            )
            if final.ok or final.cancelled or final.timed_out:
                return final
            if not attempt.retryable or final.artifacts:
                return final

        assert final is not None
        return final


def _is_codex_command(command_text: str | None) -> bool:
    if not command_text:
        return False
    try:
        import shlex

        parts = shlex.split(command_text)
    except ValueError:
        return False
    return bool(parts) and Path(parts[0]).name == "codex"
