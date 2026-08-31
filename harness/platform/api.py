from __future__ import annotations

import argparse
import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from typing import Any, Mapping

from harness.platform.config import PlatformConfig
from harness.platform.models import Domain, JobSpec, ResourceRequest, SteeringEvent, SteeringKind
from harness.platform.service import PlatformService


def create_app(service: PlatformService | None = None):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError("Install with `pip install -e '.[api]'` to run Core API") from exc

    core = service or PlatformService.from_env()
    config = core.config

    @asynccontextmanager
    async def lifespan(_app):
        core.start()
        try:
            yield
        finally:
            core.stop()

    app = FastAPI(
        title="ResearchAgent Core",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.service = core

    def authorize(authorization: str | None = Header(default=None)) -> None:
        token = _bearer(authorization)
        if not config.core_token or not hmac.compare_digest(token, config.core_token):
            raise HTTPException(status_code=401, detail="invalid Core token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        snapshot = core.health()
        return {
            "ok": snapshot["ok"],
            "scheduler": snapshot["scheduler"],
            "runtime_names": sorted(snapshot["runtimes"]),
            "compute_backend_names": sorted(snapshot["compute_backends"]),
        }

    @app.get("/v1/capabilities", dependencies=[Depends(authorize)])
    def capabilities() -> dict[str, Any]:
        return core.health()

    @app.post("/v1/projects", dependencies=[Depends(authorize)])
    def create_project(body: dict[str, Any]) -> dict[str, Any]:
        try:
            project = core.create_project(
                domain=Domain(str(body.get("domain") or Domain.RESEARCH.value)),
                title=_required(body, "title"),
                description=str(body.get("description") or ""),
                metadata=_object(body.get("metadata")),
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return project.to_dict()

    @app.get("/v1/projects", dependencies=[Depends(authorize)])
    def list_projects(limit: int = 100) -> dict[str, Any]:
        return {
            "projects": [
                item.to_dict()
                for item in core.registry.list_projects(limit=max(1, min(limit, 1000)))
            ]
        }

    @app.get("/v1/projects/{project_id}", dependencies=[Depends(authorize)])
    def get_project(project_id: str) -> dict[str, Any]:
        project = core.registry.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return project.to_dict()

    @app.post("/v1/work-sessions", dependencies=[Depends(authorize)])
    def create_session(body: dict[str, Any]) -> dict[str, Any]:
        try:
            session = core.create_work_session(
                project_id=_required(body, "project_id"),
                title=_required(body, "title"),
                objective=_required(body, "objective"),
                parent_session_id=(
                    str(body["parent_session_id"])
                    if body.get("parent_session_id")
                    else None
                ),
                metadata=_object(body.get("metadata")),
            )
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return session.to_dict()

    @app.get("/v1/work-sessions", dependencies=[Depends(authorize)])
    def list_sessions(project_id: str | None = None, limit: int = 100) -> dict[str, Any]:
        return {
            "work_sessions": [
                item.to_dict()
                for item in core.registry.list_work_sessions(
                    project_id=project_id,
                    limit=max(1, min(limit, 1000)),
                )
            ]
        }

    @app.get("/v1/work-sessions/{session_id}", dependencies=[Depends(authorize)])
    def session_status(session_id: str) -> dict[str, Any]:
        try:
            return core.session_status(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/v1/work-sessions/{session_id}/discord-route",
        dependencies=[Depends(authorize)],
    )
    def attach_route(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            session = core.attach_discord_route(
                session_id,
                guild_id=_required(body, "guild_id"),
                parent_channel_id=_required(body, "parent_channel_id"),
                thread_id=_required(body, "thread_id"),
                live_message_id=(
                    str(body["live_message_id"])
                    if body.get("live_message_id")
                    else None
                ),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return session.to_dict()

    @app.post(
        "/v1/work-sessions/{session_id}/messages",
        dependencies=[Depends(authorize)],
    )
    def message(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return core.handle_message(
                session_id=session_id,
                text=_required(body, "text"),
                actor=str(body.get("actor") or "api"),
                correlation_id=_required(body, "correlation_id"),
                mode=str(body.get("mode") or "auto"),
                steering_kind=(
                    SteeringKind(str(body["steering_kind"]))
                    if body.get("steering_kind")
                    else None
                ),
                computer_use_allowed=bool(body.get("computer_use_allowed", False)),
                metadata=_object(body.get("metadata")),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/v1/work-sessions/{session_id}/events",
        dependencies=[Depends(authorize)],
    )
    def events(
        session_id: str,
        after_sequence: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        if core.registry.get_work_session(session_id) is None:
            raise HTTPException(status_code=404, detail="work session not found")
        items = core.registry.list_events(
            session_id,
            after_sequence=max(0, after_sequence),
            limit=max(1, min(limit, 5000)),
        )
        return {
            "events": [item.to_dict() for item in items],
            "last_sequence": items[-1].sequence if items else after_sequence,
        }

    @app.post(
        "/v1/work-sessions/{session_id}/steering",
        dependencies=[Depends(authorize)],
    )
    def steering(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        if core.registry.get_work_session(session_id) is None:
            raise HTTPException(status_code=404, detail="work session not found")
        try:
            event = SteeringEvent.new(
                work_session_id=session_id,
                kind=SteeringKind(_required(body, "kind")),
                instruction=_required(body, "instruction"),
                apply_after=str(body.get("apply_after") or "next_checkpoint"),
                job_id=(str(body["job_id"]) if body.get("job_id") else None),
                metadata=_object(body.get("metadata")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return core.registry.add_steering(event).to_dict()

    @app.post(
        "/v1/work-sessions/{session_id}/cancel",
        dependencies=[Depends(authorize)],
    )
    def cancel_session(session_id: str) -> dict[str, Any]:
        try:
            return core.cancel_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/jobs", dependencies=[Depends(authorize)])
    def create_job(body: dict[str, Any]) -> dict[str, Any]:
        raw = body.get("spec") if isinstance(body.get("spec"), Mapping) else body
        try:
            spec = JobSpec.new(
                work_session_id=_required(raw, "work_session_id"),
                domain=Domain(str(raw.get("domain") or Domain.RESEARCH.value)),
                task_type=_required(raw, "task_type"),
                entrypoint=str(raw.get("entrypoint") or ""),
                parent_job_id=(
                    str(raw["parent_job_id"]) if raw.get("parent_job_id") else None
                ),
                backend_preferences=[
                    str(item) for item in raw.get("backend_preferences", [])
                ],
                resources=ResourceRequest.from_dict(
                    raw.get("resources")
                    if isinstance(raw.get("resources"), Mapping)
                    else None
                ),
                inputs=_object(raw.get("inputs")),
                outputs=[str(item) for item in raw.get("outputs", [])],
                metadata=_object(raw.get("metadata")),
            )
            record = core.registry.create_job(spec)
            if bool(body.get("enqueue", True)):
                record = core.scheduler.enqueue(spec.job_id)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return record.to_dict()

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def get_job(job_id: str) -> dict[str, Any]:
        job = core.registry.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.to_dict()

    @app.post("/v1/jobs/{job_id}/enqueue", dependencies=[Depends(authorize)])
    def enqueue_job(job_id: str) -> dict[str, Any]:
        try:
            return core.scheduler.enqueue(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/approve", dependencies=[Depends(authorize)])
    def approve_job(job_id: str) -> dict[str, Any]:
        try:
            return core.approve_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return core.scheduler.cancel(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.websocket("/v1/ws/work-sessions/{session_id}")
    async def session_events(websocket: WebSocket, session_id: str) -> None:
        token = websocket.query_params.get("token", "")
        if not config.core_token or not hmac.compare_digest(token, config.core_token):
            await websocket.close(code=4401)
            return
        if core.registry.get_work_session(session_id) is None:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        sequence = int(websocket.query_params.get("after_sequence", "0") or 0)
        try:
            while True:
                items = await asyncio.to_thread(
                    core.registry.list_events,
                    session_id,
                    after_sequence=sequence,
                    limit=500,
                )
                for item in items:
                    await websocket.send_text(
                        json.dumps(item.to_dict(), ensure_ascii=False)
                    )
                    sequence = item.sequence
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent portable Core API")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--workdir", default=".")
    args = parser.parse_args()
    config = PlatformConfig.from_env(args.workdir)
    errors = config.validate_core()
    if errors:
        raise SystemExit("; ".join(errors))
    app = create_app(PlatformService.from_env(args.workdir))
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install with `pip install -e '.[api]'`") from exc
    uvicorn.run(
        app,
        host=args.host or config.core_host,
        port=args.port or config.core_port,
        log_level="info",
    )


def _bearer(value: str | None) -> str:
    if not value:
        return ""
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _required(body: Mapping[str, Any], key: str) -> str:
    value = str(body.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
