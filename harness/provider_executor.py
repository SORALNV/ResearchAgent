from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from harness.cost import estimate_tokens


_SUPPORTED_PROVIDERS = {
    "cli",
    "codex_cli",
    "openai_responses",
    "openai_computer",
}
_PROVIDER_ALIASES = {
    "codex": "codex_cli",
    "codex-cli": "codex_cli",
    "openai": "openai_responses",
    "responses": "openai_responses",
    "computer": "openai_computer",
    "computer_use": "openai_computer",
    "computer-use": "openai_computer",
    "generic_cli": "cli",
    "generic-cli": "cli",
}


@dataclass(frozen=True)
class RuntimeAttempt:
    provider: str
    output: str
    stderr: str
    returncode: int
    duration_seconds: float
    skipped: bool = False
    timed_out: bool = False
    cancelled: bool = False
    retryable: bool = False
    response_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ok(self) -> bool:
        return not (self.skipped or self.timed_out or self.cancelled) and self.returncode == 0

    def to_event(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "returncode": self.returncode,
            "duration_seconds": round(self.duration_seconds, 3),
            "skipped": self.skipped,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "retryable": self.retryable,
            "response_id": self.response_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stderr_tail": self.stderr[-500:],
        }


@dataclass(frozen=True)
class RuntimeSettings:
    global_order: tuple[str, ...]
    role_orders: dict[str, tuple[str, ...]]
    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str | None
    openai_reasoning_effort: str | None
    openai_request_timeout_seconds: float
    openai_max_retries: int
    computer_enabled: bool
    computer_model: str | None
    computer_bridge_url: str | None
    computer_bridge_token: str | None
    computer_max_steps: int
    computer_allowed_stages: tuple[str, ...]
    computer_require_approval: bool
    runtime_event_log_enabled: bool

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        global_order = _provider_order(os.getenv("AGENT_RUNTIME_ORDER", ""))
        role_orders: dict[str, tuple[str, ...]] = {}
        for role in ("main", "sub", "review", "fresh", "claude", "planning"):
            value = _provider_order(os.getenv(f"{role.upper()}_AGENT_RUNTIME_ORDER", ""))
            if value:
                role_orders[role] = value
        return cls(
            global_order=global_order,
            role_orders=role_orders,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            openai_model=os.getenv("OPENAI_MODEL") or None,
            openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT") or None,
            openai_request_timeout_seconds=max(
                1.0, _float_env("OPENAI_REQUEST_TIMEOUT_SECONDS", 120.0)
            ),
            openai_max_retries=max(0, _int_env("OPENAI_MAX_RETRIES", 1)),
            computer_enabled=_bool_env("OPENAI_COMPUTER_ENABLED", False),
            computer_model=os.getenv("OPENAI_COMPUTER_MODEL") or None,
            computer_bridge_url=os.getenv("OPENAI_COMPUTER_BRIDGE_URL") or None,
            computer_bridge_token=os.getenv("OPENAI_COMPUTER_BRIDGE_TOKEN") or None,
            computer_max_steps=max(1, _int_env("OPENAI_COMPUTER_MAX_STEPS", 20)),
            computer_allowed_stages=_csv_env("OPENAI_COMPUTER_ALLOWED_STAGES"),
            computer_require_approval=_bool_env(
                "OPENAI_COMPUTER_REQUIRE_APPROVAL", True
            ),
            runtime_event_log_enabled=_bool_env("RUNTIME_EVENT_LOG_ENABLED", True),
        )

    def order_for(self, role: str, command_text: str | None) -> tuple[str, ...]:
        configured = self.role_orders.get(role) or self.global_order
        if configured:
            return configured
        if _is_codex_command(command_text):
            return ("codex_cli",)
        return ("cli",)


class ProviderAwareAgentCommandExecutor:
    """Wrap the local executor with Codex/OpenAI provider routing.

    The local executor remains the source of truth for subprocess sandboxing,
    cancellation, artifact manifests, and cost accounting. Remote providers are
    used only when explicitly selected by runtime order.
    """

    def __init__(
        self,
        config,
        lock,
        cancellation,
        *,
        local_executor_cls,
        invocation_cls,
        openai_client_factory: Callable[[RuntimeSettings], object] | None = None,
        bridge_request: Callable[[RuntimeSettings, dict[str, object]], dict[str, object]] | None = None,
    ) -> None:
        self.config = config
        self.lock = lock
        self.cancellation = cancellation
        self.invocation_cls = invocation_cls
        self.local = local_executor_cls(config, lock, cancellation)
        self.settings = RuntimeSettings.from_environment()
        self.openai_client_factory = openai_client_factory or _default_openai_client
        self.bridge_request = bridge_request or _default_bridge_request

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
    ):
        chain = self.settings.order_for(role, command_text)
        attempts: list[RuntimeAttempt] = []
        final_invocation = None

        for index, provider in enumerate(chain):
            self._record_event(
                session,
                {
                    "event": "provider_started",
                    "provider": provider,
                    "role": role,
                    "stage": stage,
                    "task_id": task_id,
                    "attempt": index + 1,
                },
            )

            if provider in {"cli", "codex_cli"}:
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
                final_invocation = _decorate_invocation(
                    invocation,
                    provider,
                    attempts,
                )
            elif provider == "openai_responses":
                attempt = self._run_openai_responses(
                    session=session,
                    role=role,
                    stage=stage,
                    prompt=prompt,
                    task_id=task_id,
                )
                attempts.append(attempt)
                final_invocation = self._remote_invocation(
                    attempt,
                    role=role,
                    stage=stage,
                    task_id=task_id,
                    working_dir=working_dir,
                    prompt=prompt,
                    attempts=attempts,
                )
            elif provider == "openai_computer":
                attempt = self._run_openai_computer(
                    session=session,
                    role=role,
                    stage=stage,
                    prompt=prompt,
                    task_id=task_id,
                )
                attempts.append(attempt)
                final_invocation = self._remote_invocation(
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
                    provider=provider,
                    output=f"Unsupported agent provider: {provider}",
                    stderr="provider is not registered",
                    returncode=127,
                    duration_seconds=0.0,
                    skipped=True,
                    retryable=True,
                )
                attempts.append(attempt)
                final_invocation = self._remote_invocation(
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
                    "attempt": index + 1,
                    **attempt.to_event(),
                },
            )

            if final_invocation.ok:
                return final_invocation
            if final_invocation.cancelled or final_invocation.timed_out:
                return final_invocation
            if not attempt.retryable:
                return final_invocation
            if sandbox == "workspace-write" and final_invocation.artifacts:
                return final_invocation

        if final_invocation is not None:
            return final_invocation
        return self.invocation_cls(
            role=role,
            stage=stage,
            task_id=task_id,
            command=(),
            output="No agent runtime provider is configured.",
            stderr="empty provider chain",
            returncode=127,
            duration_seconds=0.0,
            skipped=True,
            workspace=str(working_dir) if working_dir else None,
            estimated_input_tokens=estimate_tokens(prompt),
        )

    def _run_openai_responses(
        self,
        *,
        session,
        role: str,
        stage: str,
        prompt: str,
        task_id: str | None,
    ) -> RuntimeAttempt:
        unavailable = self._openai_unavailable("openai_responses")
        if unavailable:
            return unavailable
        if self.cancellation.cancelled:
            return RuntimeAttempt(
                provider="openai_responses",
                output=f"Agent cancelled before OpenAI request: {self.cancellation.reason}",
                stderr="",
                returncode=130,
                duration_seconds=0.0,
                cancelled=True,
            )
        if not self._reserve_agent_call(session):
            return RuntimeAttempt(
                provider="openai_responses",
                output="OpenAI provider skipped: MAX_AGENT_CALLS reached.",
                stderr="",
                returncode=125,
                duration_seconds=0.0,
                skipped=True,
                retryable=False,
            )

        started = time.monotonic()
        try:
            client = self.openai_client_factory(self.settings)
            arguments: dict[str, object] = {
                "model": self.settings.openai_model,
                "input": prompt,
                "metadata": _metadata(session, role, stage, task_id),
            }
            if self.settings.openai_reasoning_effort:
                arguments["reasoning"] = {
                    "effort": self.settings.openai_reasoning_effort
                }
            response = client.responses.create(**arguments)
            output = str(getattr(response, "output_text", "") or "").strip()
            if not output:
                output = _response_debug_text(response)
            input_tokens, output_tokens = _response_usage(response)
            self._record_tokens(
                session,
                input_tokens or estimate_tokens(prompt),
                output_tokens or estimate_tokens(output),
            )
            if self.cancellation.cancelled:
                return RuntimeAttempt(
                    provider="openai_responses",
                    output=output,
                    stderr=f"cancelled after response: {self.cancellation.reason}",
                    returncode=130,
                    duration_seconds=time.monotonic() - started,
                    cancelled=True,
                    response_id=str(getattr(response, "id", "") or "") or None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            return RuntimeAttempt(
                provider="openai_responses",
                output=output,
                stderr="",
                returncode=0,
                duration_seconds=time.monotonic() - started,
                response_id=str(getattr(response, "id", "") or "") or None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except Exception as exc:  # SDK exceptions vary by installed version.
            return RuntimeAttempt(
                provider="openai_responses",
                output=f"OpenAI Responses provider failed: {type(exc).__name__}",
                stderr=str(exc),
                returncode=1,
                duration_seconds=time.monotonic() - started,
                retryable=_openai_retryable(exc),
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
        started = time.monotonic()
        if not self.settings.computer_enabled:
            return RuntimeAttempt(
                provider="openai_computer",
                output="OpenAI computer provider is disabled.",
                stderr="set OPENAI_COMPUTER_ENABLED=true",
                returncode=127,
                duration_seconds=0.0,
                skipped=True,
                retryable=True,
            )
        unavailable = self._openai_unavailable("openai_computer")
        if unavailable:
            return unavailable
        if not self.settings.computer_model:
            return RuntimeAttempt(
                provider="openai_computer",
                output="OpenAI computer model is not configured.",
                stderr="OPENAI_COMPUTER_MODEL is required",
                returncode=127,
                duration_seconds=0.0,
                skipped=True,
                retryable=True,
            )
        if not self.settings.computer_bridge_url:
            return RuntimeAttempt(
                provider="openai_computer",
                output="OpenAI computer bridge is not configured.",
                stderr="OPENAI_COMPUTER_BRIDGE_URL is required",
                returncode=127,
                duration_seconds=0.0,
                skipped=True,
                retryable=True,
            )
        if (
            self.settings.computer_allowed_stages
            and stage not in self.settings.computer_allowed_stages
        ):
            return RuntimeAttempt(
                provider="openai_computer",
                output=f"Computer use is not allowed for stage: {stage}",
                stderr="stage is outside OPENAI_COMPUTER_ALLOWED_STAGES",
                returncode=126,
                duration_seconds=0.0,
                skipped=True,
                retryable=False,
            )

        blanket_operation = _computer_operation(role, stage, task_id)
        if (
            self.settings.computer_require_approval
            and not _operation_is_approved(session, blanket_operation)
        ):
            return RuntimeAttempt(
                provider="openai_computer",
                output=_approval_required_line(
                    blanket_operation,
                    "Computer Useを開始する前に人間承認が必要です。",
                    "外部GUIを操作し、画面上の状態を変更する可能性があります。",
                ),
                stderr="computer use approval required",
                returncode=126,
                duration_seconds=time.monotonic() - started,
                retryable=False,
            )
        if self.cancellation.cancelled:
            return RuntimeAttempt(
                provider="openai_computer",
                output=f"Computer use cancelled before request: {self.cancellation.reason}",
                stderr="",
                returncode=130,
                duration_seconds=0.0,
                cancelled=True,
            )
        if not self._reserve_agent_call(session):
            return RuntimeAttempt(
                provider="openai_computer",
                output="OpenAI computer provider skipped: MAX_AGENT_CALLS reached.",
                stderr="",
                returncode=125,
                duration_seconds=0.0,
                skipped=True,
            )

        total_input = 0
        total_output = 0
        try:
            client = self.openai_client_factory(self.settings)
            response = client.responses.create(
                model=self.settings.computer_model,
                tools=[{"type": "computer"}],
                input=prompt,
                metadata=_metadata(session, role, stage, task_id),
            )
            for step in range(1, self.settings.computer_max_steps + 1):
                input_tokens, output_tokens = _response_usage(response)
                total_input += input_tokens
                total_output += output_tokens
                calls = _computer_calls(response)
                if not calls:
                    output = str(getattr(response, "output_text", "") or "").strip()
                    if not output:
                        output = _response_debug_text(response)
                    self._record_tokens(
                        session,
                        total_input or estimate_tokens(prompt),
                        total_output or estimate_tokens(output),
                    )
                    return RuntimeAttempt(
                        provider="openai_computer",
                        output=output,
                        stderr="",
                        returncode=0,
                        duration_seconds=time.monotonic() - started,
                        response_id=str(getattr(response, "id", "") or "") or None,
                        input_tokens=total_input,
                        output_tokens=total_output,
                    )

                call = calls[0]
                safety_checks = _model_dump_list(
                    getattr(call, "pending_safety_checks", None) or []
                )
                safety_operation = _computer_safety_operation(
                    role,
                    stage,
                    task_id,
                    safety_checks,
                )
                if safety_checks and not _operation_is_approved(
                    session, safety_operation
                ):
                    return RuntimeAttempt(
                        provider="openai_computer",
                        output=_approval_required_line(
                            safety_operation,
                            "OpenAI computer toolが安全確認を要求しました。",
                            json.dumps(safety_checks, ensure_ascii=False),
                        ),
                        stderr="pending computer safety checks",
                        returncode=126,
                        duration_seconds=time.monotonic() - started,
                        retryable=False,
                        response_id=str(getattr(response, "id", "") or "") or None,
                        input_tokens=total_input,
                        output_tokens=total_output,
                    )

                if self.cancellation.cancelled:
                    return RuntimeAttempt(
                        provider="openai_computer",
                        output=f"Computer use cancelled: {self.cancellation.reason}",
                        stderr="",
                        returncode=130,
                        duration_seconds=time.monotonic() - started,
                        cancelled=True,
                        response_id=str(getattr(response, "id", "") or "") or None,
                        input_tokens=total_input,
                        output_tokens=total_output,
                    )

                bridge_payload = {
                    "session_id": session.session_id,
                    "role": role,
                    "stage": stage,
                    "task_id": task_id,
                    "step": step,
                    "response_id": str(getattr(response, "id", "") or ""),
                    "call_id": str(getattr(call, "call_id", "") or ""),
                    "action": _model_dump(getattr(call, "action", None)),
                    "actions": _model_dump(getattr(call, "actions", None)),
                }
                bridge_result = self.bridge_request(self.settings, bridge_payload)
                screenshot = str(
                    bridge_result.get("screenshot_data_url")
                    or bridge_result.get("image_url")
                    or ""
                )
                if not screenshot.startswith("data:image/"):
                    raise ValueError(
                        "computer bridge must return screenshot_data_url as a data URL"
                    )
                acknowledged = [
                    {
                        "id": str(item.get("id") or ""),
                        "code": item.get("code"),
                        "message": item.get("message"),
                    }
                    for item in safety_checks
                ]
                next_input: dict[str, object] = {
                    "type": "computer_call_output",
                    "call_id": str(getattr(call, "call_id", "") or ""),
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": screenshot,
                    },
                }
                if acknowledged:
                    next_input["acknowledged_safety_checks"] = acknowledged
                response = client.responses.create(
                    model=self.settings.computer_model,
                    tools=[{"type": "computer"}],
                    previous_response_id=str(getattr(response, "id", "")),
                    input=[next_input],
                    metadata=_metadata(session, role, stage, task_id),
                )

            return RuntimeAttempt(
                provider="openai_computer",
                output=(
                    "OpenAI computer provider reached OPENAI_COMPUTER_MAX_STEPS "
                    f"({self.settings.computer_max_steps})."
                ),
                stderr="computer action loop did not converge",
                returncode=1,
                duration_seconds=time.monotonic() - started,
                retryable=False,
                response_id=str(getattr(response, "id", "") or "") or None,
                input_tokens=total_input,
                output_tokens=total_output,
            )
        except Exception as exc:
            return RuntimeAttempt(
                provider="openai_computer",
                output=f"OpenAI computer provider failed: {type(exc).__name__}",
                stderr=str(exc),
                returncode=1,
                duration_seconds=time.monotonic() - started,
                retryable=_openai_retryable(exc),
                input_tokens=total_input,
                output_tokens=total_output,
            )

    def _remote_invocation(
        self,
        attempt: RuntimeAttempt,
        *,
        role: str,
        stage: str,
        task_id: str | None,
        working_dir: Path | None,
        prompt: str,
        attempts: Iterable[RuntimeAttempt],
    ):
        trace = _provider_trace(attempts)
        stderr = "\n".join(item for item in (trace, attempt.stderr) if item)
        command = [f"provider:{attempt.provider}"]
        if attempt.response_id:
            command.append(f"response_id:{attempt.response_id}")
        return self.invocation_cls(
            role=role,
            stage=stage,
            task_id=task_id,
            command=tuple(command),
            output=attempt.output,
            stderr=stderr,
            returncode=attempt.returncode,
            duration_seconds=attempt.duration_seconds,
            skipped=attempt.skipped,
            timed_out=attempt.timed_out,
            cancelled=attempt.cancelled,
            workspace=str(working_dir) if working_dir else None,
            estimated_input_tokens=(
                attempt.input_tokens or estimate_tokens(prompt)
            ),
            estimated_output_tokens=(
                attempt.output_tokens
                or estimate_tokens(attempt.output)
                + estimate_tokens(attempt.stderr)
            ),
        )

    def _openai_unavailable(self, provider: str) -> RuntimeAttempt | None:
        if not self.settings.openai_api_key:
            return RuntimeAttempt(
                provider=provider,
                output="OpenAI provider is unavailable: OPENAI_API_KEY is not set.",
                stderr="missing OPENAI_API_KEY",
                returncode=127,
                duration_seconds=0.0,
                skipped=True,
                retryable=True,
            )
        if provider == "openai_responses" and not self.settings.openai_model:
            return RuntimeAttempt(
                provider=provider,
                output="OpenAI provider is unavailable: OPENAI_MODEL is not set.",
                stderr="missing OPENAI_MODEL",
                returncode=127,
                duration_seconds=0.0,
                skipped=True,
                retryable=True,
            )
        return None

    def _reserve_agent_call(self, session) -> bool:
        with self.lock:
            if (
                self.config.max_agent_calls > 0
                and session.cost.agent_calls >= self.config.max_agent_calls
            ):
                return False
            session.cost.agent_calls += 1
            return True

    def _record_tokens(self, session, input_tokens: int, output_tokens: int) -> None:
        with self.lock:
            session.cost.estimated_tokens += max(0, input_tokens) + max(
                0, output_tokens
            )

    def _record_event(self, session, event: dict[str, object]) -> None:
        if not self.settings.runtime_event_log_enabled:
            return
        root = Path(session.research_dir or self.config.project_root)
        path = root / "artifacts" / "runtime_events.jsonl"
        payload = {
            "timestamp": time.time(),
            "session_id": session.session_id,
            **event,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(payload, ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
        except OSError:
            # Runtime event logging must never break the research pipeline.
            return


def build_provider_executor_class(local_executor_cls, invocation_cls):
    """Return a concrete executor without importing multi_agent_types here."""

    class ConcreteProviderExecutor(ProviderAwareAgentCommandExecutor):
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
                local_executor_cls=local_executor_cls,
                invocation_cls=invocation_cls,
                openai_client_factory=openai_client_factory,
                bridge_request=bridge_request,
            )

    ConcreteProviderExecutor.__name__ = "ProviderAwareAgentCommandExecutor"
    return ConcreteProviderExecutor


def _decorate_invocation(invocation, provider: str, attempts: Iterable[RuntimeAttempt]):
    trace = _provider_trace(attempts)
    stderr = "\n".join(item for item in (trace, invocation.stderr) if item)
    return replace(
        invocation,
        command=(f"provider:{provider}", *invocation.command),
        stderr=stderr,
    )


def _provider_trace(attempts: Iterable[RuntimeAttempt]) -> str:
    rows = []
    for index, attempt in enumerate(attempts, 1):
        rows.append(
            "provider_attempt "
            f"{index}: provider={attempt.provider} returncode={attempt.returncode} "
            f"skipped={attempt.skipped} retryable={attempt.retryable}"
        )
    return "\n".join(rows)


def _provider_order(value: str) -> tuple[str, ...]:
    result: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        item = _PROVIDER_ALIASES.get(item, item)
        if item not in _SUPPORTED_PROVIDERS:
            continue
        if item not in result:
            result.append(item)
    return tuple(result)


def _is_codex_command(command_text: str | None) -> bool:
    if not command_text:
        return False
    try:
        import shlex

        parts = shlex.split(command_text)
    except ValueError:
        return False
    return bool(parts) and Path(parts[0]).name == "codex"


def _local_retryable(invocation) -> bool:
    if invocation.cancelled or invocation.timed_out:
        return False
    if invocation.artifacts:
        return False
    return invocation.skipped or invocation.returncode in {126, 127}


def _default_openai_client(settings: RuntimeSettings):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the OpenAI provider with `pip install -e .[openai]`."
        ) from exc
    arguments: dict[str, object] = {
        "api_key": settings.openai_api_key,
        "timeout": settings.openai_request_timeout_seconds,
        "max_retries": settings.openai_max_retries,
    }
    if settings.openai_base_url:
        arguments["base_url"] = settings.openai_base_url
    return OpenAI(**arguments)


def _default_bridge_request(
    settings: RuntimeSettings,
    payload: dict[str, object],
) -> dict[str, object]:
    assert settings.computer_bridge_url is not None
    url = settings.computer_bridge_url.rstrip("/") + "/v1/actions"
    headers = {"Content-Type": "application/json"}
    if settings.computer_bridge_token:
        headers["Authorization"] = f"Bearer {settings.computer_bridge_token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=settings.openai_request_timeout_seconds,
        ) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"computer bridge HTTP {exc.code}: {detail}") from exc
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("computer bridge response must be a JSON object")
    return value


def _response_usage(response: object) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _response_debug_text(response: object) -> str:
    for name in ("model_dump_json", "json"):
        method = getattr(response, name, None)
        if callable(method):
            try:
                return str(method())
            except Exception:
                continue
    return str(response)


def _computer_calls(response: object) -> list[object]:
    output = getattr(response, "output", None)
    if not isinstance(output, (list, tuple)):
        return []
    return [item for item in output if getattr(item, "type", None) == "computer_call"]


def _model_dump(value: object) -> object:
    if value is None:
        return None
    method = getattr(value, "model_dump", None)
    if callable(method):
        return method(exclude_none=True)
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    data = getattr(value, "__dict__", None)
    return dict(data) if isinstance(data, dict) else str(value)


def _model_dump_list(values: Iterable[object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for value in values:
        dumped = _model_dump(value)
        if isinstance(dumped, dict):
            result.append(dumped)
    return result


def _metadata(session, role: str, stage: str, task_id: str | None) -> dict[str, str]:
    return {
        "research_session_id": str(session.session_id)[:64],
        "agent_role": role[:64],
        "agent_stage": stage[:64],
        "agent_task_id": (task_id or "global")[:64],
    }


def _computer_operation(role: str, stage: str, task_id: str | None) -> str:
    return f"openai_computer_use:{role}:{stage}:{task_id or 'global'}"


def _computer_safety_operation(
    role: str,
    stage: str,
    task_id: str | None,
    checks: list[dict[str, object]],
) -> str:
    identifiers = [
        str(item.get("code") or item.get("id") or "unknown") for item in checks
    ]
    suffix = ",".join(identifiers) or "unknown"
    return (
        f"openai_computer_safety:{role}:{stage}:{task_id or 'global'}:{suffix}"
    )


def _operation_is_approved(session, operation: str) -> bool:
    requests = getattr(session, "approval_requests", {}) or {}
    for request in requests.values():
        if (
            getattr(request, "operation", None) == operation
            and getattr(request, "status", None) == "approved"
        ):
            return True
    return False


def _approval_required_line(operation: str, reason: str, impact: str) -> str:
    safe_reason = reason.replace(";", ",")
    safe_impact = impact.replace(";", ",")
    return (
        "APPROVAL_REQUIRED: "
        f"operation={operation}; reason={safe_reason}; impact={safe_impact}; "
        "dry_run_result=未実行"
    )


def _openai_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    return isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip()
            for item in os.getenv(name, "").split(",")
            if item.strip()
        )
    )
