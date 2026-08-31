from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from harness.runtime.base import (
    AgentRuntime,
    RuntimeCapability,
    RuntimeRequest,
    RuntimeResult,
)
from harness.runtime.computer import ComputerUseDriver
from harness.runtime.tools import HarnessToolRegistry, ToolExecutionContext


class ResponsesTransport(Protocol):
    def create(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Create one OpenAI Response."""


@dataclass
class UrllibResponsesTransport:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    organization: str | None = None
    project: str | None = None
    timeout_seconds: int = 180

    def create(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=max(1, self.timeout_seconds),
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI Responses API HTTP {exc.code}: {detail[-4000:]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI Responses API connection failed: {exc}") from exc
        value = json.loads(body)
        if not isinstance(value, dict):
            raise RuntimeError("OpenAI Responses API returned a non-object response")
        return value

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project
        return headers


class OpenAIResponsesRuntime(AgentRuntime):
    """OpenAI Responses runtime with typed ResearchAgent harness tools.

    Function calls are executed only through HarnessToolRegistry. Computer-use
    is disabled by default, requires an explicit request flag, and is returned as
    an approval proposal when a driver or safety acknowledgement is missing.
    """

    name = "openai_responses"
    capabilities = frozenset(
        {
            RuntimeCapability.CHAT,
            RuntimeCapability.REASONING,
            RuntimeCapability.FUNCTION_TOOLS,
            RuntimeCapability.VISION,
            RuntimeCapability.COMPUTER_USE,
        }
    )

    def __init__(
        self,
        *,
        model: str,
        transport: ResponsesTransport,
        tools: HarnessToolRegistry | None = None,
        tool_context_factory: Callable[[RuntimeRequest], ToolExecutionContext]
        | None = None,
        computer_tool: Mapping[str, Any] | None = None,
        computer_driver_factory: Callable[[RuntimeRequest], ComputerUseDriver]
        | None = None,
    ) -> None:
        self.model = model
        self.transport = transport
        self.tools = tools
        self.tool_context_factory = tool_context_factory
        self.computer_tool = dict(computer_tool or {})
        self.computer_driver_factory = computer_driver_factory
        self._active_driver: ComputerUseDriver | None = None

    def available(self) -> tuple[bool, str]:
        if not self.model.strip():
            return False, "OPENAI_MODEL is not configured"
        return True, self.model

    def run(self, request: RuntimeRequest) -> RuntimeResult:
        started = time.monotonic()
        if request.requires(RuntimeCapability.COMPUTER_USE) and not request.computer_use_allowed:
            return RuntimeResult(
                runtime=self.name,
                model=request.model or self.model,
                output_text=(
                    "Computer use requires explicit approval. No browser action was executed."
                ),
                pending_actions=(
                    {
                        "type": "computer_use_session",
                        "purpose": request.prompt,
                        "context": request.context.to_dict(),
                    },
                ),
                requires_approval=True,
                duration_seconds=time.monotonic() - started,
            )

        payload = self._initial_payload(request)
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        pending_actions: list[dict[str, Any]] = []
        response: dict[str, Any] = {}
        driver: ComputerUseDriver | None = None
        try:
            for _ in range(max(1, request.max_tool_rounds + 1)):
                response = self.transport.create(payload)
                output_items = _output_items(response)
                function_calls = [
                    item for item in output_items if item.get("type") == "function_call"
                ]
                computer_calls = [
                    item
                    for item in output_items
                    if item.get("type")
                    in {"computer_call", "computer_use_call"}
                ]

                next_input: list[dict[str, Any]] = []
                if function_calls:
                    if not request.tools_enabled or self.tools is None:
                        return self._result(
                            request=request,
                            response=response,
                            started=started,
                            tool_calls=tool_calls,
                            tool_results=tool_results,
                            pending_actions=[
                                {
                                    "type": "unavailable_function_tools",
                                    "calls": function_calls,
                                }
                            ],
                            requires_approval=True,
                            error="Model requested unavailable harness tools",
                        )
                    context = self._tool_context(request)
                    for call in function_calls:
                        call_record = {
                            "type": "function_call",
                            "name": str(call.get("name") or ""),
                            "arguments": call.get("arguments") or "{}",
                            "call_id": str(call.get("call_id") or call.get("id") or ""),
                        }
                        tool_calls.append(call_record)
                        result = self.tools.execute(
                            call_record["name"],
                            call_record["arguments"],
                            context,
                        )
                        result_record = {
                            "call_id": call_record["call_id"],
                            "name": call_record["name"],
                            "result": result,
                        }
                        tool_results.append(result_record)
                        if result.get("requires_approval"):
                            pending_actions.append(
                                {
                                    "type": "tool_approval",
                                    **dict(result.get("proposal") or {}),
                                    "call_id": call_record["call_id"],
                                }
                            )
                            return self._result(
                                request=request,
                                response=response,
                                started=started,
                                tool_calls=tool_calls,
                                tool_results=tool_results,
                                pending_actions=pending_actions,
                                requires_approval=True,
                            )
                        next_input.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_record["call_id"],
                                "output": json.dumps(
                                    result,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            }
                        )

                if computer_calls:
                    if not request.computer_use_allowed:
                        return self._result(
                            request=request,
                            response=response,
                            started=started,
                            tool_calls=tool_calls,
                            tool_results=tool_results,
                            pending_actions=computer_calls,
                            requires_approval=True,
                        )
                    if driver is None:
                        if self.computer_driver_factory is None:
                            return self._result(
                                request=request,
                                response=response,
                                started=started,
                                tool_calls=tool_calls,
                                tool_results=tool_results,
                                pending_actions=computer_calls,
                                requires_approval=True,
                                error="Computer-use driver is not configured",
                            )
                        driver = self.computer_driver_factory(request)
                        self._active_driver = driver
                    for call in computer_calls:
                        safety_checks = list(call.get("pending_safety_checks") or [])
                        acknowledged = set(
                            str(item)
                            for item in request.metadata.get(
                                "acknowledged_safety_check_ids", []
                            )
                        )
                        unresolved = [
                            item
                            for item in safety_checks
                            if str(item.get("id") or item) not in acknowledged
                        ]
                        if unresolved:
                            return self._result(
                                request=request,
                                response=response,
                                started=started,
                                tool_calls=tool_calls,
                                tool_results=tool_results,
                                pending_actions=(
                                    [
                                        {
                                            "type": "computer_safety_checks",
                                            "call_id": call.get("call_id") or call.get("id"),
                                            "checks": unresolved,
                                            "current_url": driver.current_url(),
                                        }
                                    ]
                                ),
                                requires_approval=True,
                            )
                        action = call.get("action")
                        if not isinstance(action, dict):
                            return self._result(
                                request=request,
                                response=response,
                                started=started,
                                tool_calls=tool_calls,
                                tool_results=tool_results,
                                pending_actions=[call],
                                requires_approval=True,
                                error="Computer call has no structured action",
                            )
                        driver.apply(action)
                        screenshot = driver.screenshot_base64()
                        next_input.append(
                            {
                                "type": "computer_call_output",
                                "call_id": str(
                                    call.get("call_id") or call.get("id") or ""
                                ),
                                "output": {
                                    "type": "computer_screenshot",
                                    "image_url": f"data:image/png;base64,{screenshot}",
                                },
                                "acknowledged_safety_checks": safety_checks,
                            }
                        )

                if not next_input:
                    return self._result(
                        request=request,
                        response=response,
                        started=started,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                        pending_actions=pending_actions,
                    )

                payload = {
                    "model": request.model or self.model,
                    "previous_response_id": str(response.get("id") or ""),
                    "input": next_input,
                }
                if self._tools_for(request):
                    payload["tools"] = self._tools_for(request)
            return self._result(
                request=request,
                response=response,
                started=started,
                tool_calls=tool_calls,
                tool_results=tool_results,
                pending_actions=pending_actions,
                error="Maximum tool rounds reached",
                returncode=2,
            )
        except Exception as exc:
            return RuntimeResult(
                runtime=self.name,
                model=request.model or self.model,
                output_text="",
                tool_calls=tuple(tool_calls),
                tool_results=tuple(tool_results),
                pending_actions=tuple(pending_actions),
                duration_seconds=time.monotonic() - started,
                returncode=1,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if driver is not None:
                driver.close()
            self._active_driver = None

    def cancel(self, reason: str = "cancel requested") -> int:
        driver = self._active_driver
        if driver is None:
            return 0
        try:
            driver.close()
        finally:
            self._active_driver = None
        return 1

    def _initial_payload(self, request: RuntimeRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "input": request.prompt,
        }
        if request.system_prompt.strip():
            payload["instructions"] = request.system_prompt.strip()
        tools = self._tools_for(request)
        if tools:
            payload["tools"] = tools
        if request.response_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "research_agent_response",
                    "schema": request.response_schema,
                    "strict": True,
                }
            }
        metadata = {
            key: str(value)[:512]
            for key, value in {
                "project_id": request.context.project_id,
                "work_session_id": request.context.work_session_id,
                "job_id": request.context.job_id,
                "role": request.context.role,
                "stage": request.context.stage,
            }.items()
            if value is not None
        }
        if metadata:
            payload["metadata"] = metadata
        return payload

    def _tools_for(self, request: RuntimeRequest) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if request.tools_enabled and self.tools is not None:
            result.extend(self.tools.definitions())
        if (
            request.requires(RuntimeCapability.COMPUTER_USE)
            and request.computer_use_allowed
            and self.computer_tool
        ):
            result.append(dict(self.computer_tool))
        return result

    def _tool_context(self, request: RuntimeRequest) -> ToolExecutionContext:
        if self.tool_context_factory is None:
            raise RuntimeError("tool_context_factory is not configured")
        return self.tool_context_factory(request)

    def _result(
        self,
        *,
        request: RuntimeRequest,
        response: dict[str, Any],
        started: float,
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        pending_actions: list[dict[str, Any]],
        requires_approval: bool = False,
        error: str | None = None,
        returncode: int = 0,
    ) -> RuntimeResult:
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        input_details = (
            usage.get("input_tokens_details")
            if isinstance(usage.get("input_tokens_details"), dict)
            else {}
        )
        return RuntimeResult(
            runtime=self.name,
            model=str(response.get("model") or request.model or self.model),
            output_text=_extract_output_text(response)[: request.max_output_chars],
            response_id=(str(response["id"]) if response.get("id") else None),
            tool_calls=tuple(tool_calls),
            tool_results=tuple(tool_results),
            pending_actions=tuple(pending_actions),
            requires_approval=requires_approval,
            input_tokens=_optional_int(usage.get("input_tokens")),
            output_tokens=_optional_int(usage.get("output_tokens")),
            cached_tokens=_optional_int(input_details.get("cached_tokens")),
            duration_seconds=time.monotonic() - started,
            returncode=returncode,
            error=error,
            raw=response,
        )


def _output_items(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = response.get("output")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _extract_output_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    parts: list[str] = []
    for item in _output_items(response):
        if item.get("type") == "message":
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") in {"output_text", "text"}:
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(text, Mapping) and isinstance(text.get("value"), str):
                        parts.append(str(text["value"]))
        elif item.get("type") in {"output_text", "text"}:
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
