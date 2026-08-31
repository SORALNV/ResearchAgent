from __future__ import annotations

from typing import Any, Mapping


class CoreApiError(RuntimeError):
    pass


class CoreApiClient:
    """Small async client used by the Discord Edge container."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install with `pip install -e '.[api]'`") from exc
        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health", authenticated=False)

    async def capabilities(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/capabilities")

    async def create_project(
        self,
        *,
        domain: str,
        title: str,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/projects",
            {
                "domain": domain,
                "title": title,
                "description": description,
                "metadata": dict(metadata or {}),
            },
        )

    async def create_work_session(
        self,
        *,
        project_id: str,
        title: str,
        objective: str,
        parent_session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/work-sessions",
            {
                "project_id": project_id,
                "title": title,
                "objective": objective,
                "parent_session_id": parent_session_id,
                "metadata": dict(metadata or {}),
            },
        )

    async def attach_discord_route(
        self,
        session_id: str,
        *,
        guild_id: str | int,
        parent_channel_id: str | int,
        thread_id: str | int,
        live_message_id: str | int | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/work-sessions/{session_id}/discord-route",
            {
                "guild_id": str(guild_id),
                "parent_channel_id": str(parent_channel_id),
                "thread_id": str(thread_id),
                "live_message_id": (
                    str(live_message_id) if live_message_id is not None else None
                ),
            },
        )

    async def message(
        self,
        session_id: str,
        *,
        text: str,
        actor: str,
        correlation_id: str,
        mode: str = "auto",
        steering_kind: str | None = None,
        computer_use_allowed: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/work-sessions/{session_id}/messages",
            {
                "text": text,
                "actor": actor,
                "correlation_id": correlation_id,
                "mode": mode,
                "steering_kind": steering_kind,
                "computer_use_allowed": computer_use_allowed,
                "metadata": dict(metadata or {}),
            },
        )

    async def status(self, session_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/work-sessions/{session_id}")

    async def events(
        self,
        session_id: str,
        *,
        after_sequence: int,
        limit: int = 500,
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/v1/work-sessions/{session_id}/events",
            params={"after_sequence": after_sequence, "limit": limit},
        )

    async def cancel_session(self, session_id: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/work-sessions/{session_id}/cancel",
            {},
        )

    async def approve_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/v1/jobs/{job_id}/approve", {})

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/v1/jobs/{job_id}/cancel", {})

    async def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        *,
        params: Mapping[str, Any] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        headers = None if authenticated else {"Authorization": ""}
        response = await self.client.request(
            method,
            path,
            json=dict(body) if body is not None else None,
            params=dict(params or {}),
            headers=headers,
        )
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise CoreApiError(
                f"Core API {method} {path} returned {response.status_code}: {detail}"
            )
        value = response.json()
        if not isinstance(value, dict):
            raise CoreApiError(f"Core API {method} {path} returned non-object JSON")
        return value
