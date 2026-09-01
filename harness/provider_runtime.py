from __future__ import annotations

import os
import shlex
from dataclasses import replace
from pathlib import Path

from harness.codex_app_server import (
    CodexAppServerAgentExecutor,
    CodexAppServerRuntime,
)
from harness.cost import estimate_tokens
from harness.multi_agent_types import AgentCommandExecutor, AgentInvocation
from harness.provider_executor import (
    RuntimeAttempt,
    ProviderAwareAgentCommandExecutor as ProviderExecutorBase,
    _decorate_invocation,
    _local_retryable,
)


class ProviderAwareAgentCommandExecutor(ProviderExecutorBase):
    """Provider router with Codex served only through `codex app-server`.

    The existing Harness remains above every provider, preserving checkpoints,
    review, cancellation, artifact promotion, and cost accounting. The legacy
    provider selector ``codex_cli`` remains accepted as a compatibility alias,
    but it no longer starts ``codex exec``; it is routed to the shared App
    Server runtime.
    """

    def __init__(
        self,
        config,
        lock,
        cancellation,
        *,
        openai_client_factory=None,
        bridge_request=None,
        codex_app_server: CodexAppServerRuntime | None = None,
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
        self.settings = _settings_with_app_server_orders(self.settings)
        self.codex_app_server = CodexAppServerAgentExecutor(
            config,
            lock,
            cancellation,
            AgentInvocation,
            runtime=codex_app_server,
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
        chain = self.settings.order_for(role, command_text)
        provider_chain = tuple(dict.fromkeys(_provider_label(item) for item in chain))
        if stage.startswith("discord_"):
            provider_chain = (
                "codex_app_server",
                *(
                    item
                    for item in provider_chain
                    if _normalize_provider(item) != "codex_app_server"
                ),
            )
        if _is_codex_command(command_text) and provider_chain == ("cli",):
            provider_chain = ("codex_app_server",)

        if sandbox == "workspace-write":
            provider_chain = tuple(
                provider
                for provider in provider_chain
                if _normalize_provider(provider) in {"cli", "codex_app_server"}
            )
            if not provider_chain:
                return AgentInvocation(
                    role=role,
                    stage=stage,
                    task_id=task_id,
                    command=(),
                    output=(
                        "No workspace-capable provider is configured. "
                        "Use Codex App Server for workspace-write stages."
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
        for index, provider_label in enumerate(provider_chain, 1):
            effective_provider = _normalize_provider(provider_label)
            if effective_provider == "cli" and _is_codex_command(command_text):
                effective_provider = "codex_app_server"
                provider_label = "codex_app_server"

            self._record_event(
                session,
                {
                    "event": "provider_started",
                    "provider": provider_label,
                    "transport": (
                        "codex_app_server"
                        if effective_provider == "codex_app_server"
                        else effective_provider
                    ),
                    "role": role,
                    "stage": stage,
                    "task_id": task_id,
                    "attempt": index,
                },
            )

            if effective_provider == "codex_app_server":
                invocation = self.codex_app_server.run(
                    session=session,
                    role=role,
                    stage=stage,
                    prompt=prompt,
                    command_text=None,
                    sandbox=sandbox,
                    task_id=task_id,
                    working_dir=working_dir,
                )
                attempt = RuntimeAttempt(
                    provider=provider_label,
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
                final = _decorate_invocation(invocation, provider_label, attempts)
            elif effective_provider == "cli":
                invocation = self.local.run(
                    session=session,
                    role=role,
                    stage=stage,
                    prompt=prompt,
                    command_text=command_text,
                    sandbox=sandbox,
                    task_id=task_id,
                    working_dir=working_dir,
                )
                attempt = RuntimeAttempt(
                    provider=provider_label,
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
                final = _decorate_invocation(invocation, provider_label, attempts)
            elif effective_provider == "openai_responses":
                attempt = self._run_openai_responses(
                    session=session,
                    role=role,
                    stage=stage,
                    prompt=prompt,
                    task_id=task_id,
                )
                attempts.append(attempt)
                final = self._remote_invocation(
                    attempt,
                    role=role,
                    stage=stage,
                    task_id=task_id,
                    working_dir=working_dir,
                    prompt=prompt,
                    attempts=attempts,
                )
            elif effective_provider == "openai_computer":
                attempt = self._run_openai_computer(
                    session=session,
                    role=role,
                    stage=stage,
                    prompt=prompt,
                    task_id=task_id,
                )
                attempts.append(attempt)
                final = self._remote_invocation(
                    attempt,
                    role=role,
                    stage=stage,
                    task_id=task_id,
                    working_dir=working_dir,
                    prompt=prompt,
                    attempts=attempts,
                )
            else:
                attempt = RuntimeAttempt(
                    provider=provider_label,
                    output=f"Unsupported agent provider: {provider_label}",
                    stderr="provider is not registered",
                    returncode=127,
                    duration_seconds=0.0,
                    skipped=True,
                    retryable=True,
                )
                attempts.append(attempt)
                final = self._remote_invocation(
                    attempt,
                    role=role,
                    stage=stage,
                    task_id=task_id,
                    working_dir=working_dir,
                    prompt=prompt,
                    attempts=attempts,
                )

            self._record_event(
                session,
                {
                    "event": "provider_finished",
                    "role": role,
                    "stage": stage,
                    "task_id": task_id,
                    "attempt": index,
                    "transport": (
                        "codex_app_server"
                        if effective_provider == "codex_app_server"
                        else effective_provider
                    ),
                    **attempt.to_event(),
                },
            )
            if final.ok or final.cancelled or final.timed_out:
                return final
            if not attempt.retryable or final.artifacts:
                return final

        if final is not None:
            return final
        return AgentInvocation(
            role=role,
            stage=stage,
            task_id=task_id,
            command=(),
            output="No agent provider is configured.",
            stderr="empty provider chain",
            returncode=127,
            duration_seconds=0.0,
            skipped=True,
            workspace=str(working_dir) if working_dir else None,
            estimated_input_tokens=estimate_tokens(prompt),
        )


def _settings_with_app_server_orders(settings):
    global_order = _runtime_order(
        os.getenv("AGENT_RUNTIME_ORDER"),
        settings.global_order,
    )
    role_orders = dict(settings.role_orders)
    for role in ("main", "sub", "review", "fresh", "claude", "planning"):
        raw = os.getenv(f"{role.upper()}_AGENT_RUNTIME_ORDER")
        if raw is not None and raw.strip():
            role_orders[role] = _runtime_order(raw, ())
        elif role in role_orders:
            role_orders[role] = tuple(
                _normalize_provider(item) for item in role_orders[role]
            )
    return replace(
        settings,
        global_order=global_order,
        role_orders=role_orders,
    )


def _runtime_order(
    raw: str | None,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        values = fallback
    else:
        values = tuple(item.strip() for item in raw.split(",") if item.strip())
    result: list[str] = []
    for value in values:
        label = _provider_label(value)
        effective = _normalize_provider(label)
        if effective not in {
            "codex_app_server",
            "cli",
            "openai_responses",
            "openai_computer",
        }:
            continue
        if label not in result:
            result.append(label)
    return tuple(result)


def _provider_label(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"codex", "codex_cli"}:
        return "codex_cli"
    if normalized in {"codex_appserver", "app_server", "appserver"}:
        return "codex_app_server"
    if normalized in {"openai", "responses"}:
        return "openai_responses"
    if normalized in {"computer", "computer_use"}:
        return "openai_computer"
    if normalized == "generic_cli":
        return "cli"
    return normalized


def _normalize_provider(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "codex": "codex_app_server",
        "codex_cli": "codex_app_server",
        "codex_appserver": "codex_app_server",
        "app_server": "codex_app_server",
        "appserver": "codex_app_server",
        "openai": "openai_responses",
        "responses": "openai_responses",
        "computer": "openai_computer",
        "computer_use": "openai_computer",
        "generic_cli": "cli",
    }
    return aliases.get(normalized, normalized)


def _is_codex_command(command_text: str | None) -> bool:
    if not command_text:
        return False
    try:
        parts = shlex.split(command_text)
    except ValueError:
        return False
    return bool(parts) and Path(parts[0]).name.lower() in {"codex", "codex.exe"}
