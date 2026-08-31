from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from harness.control_plane import (
    ConflictError,
    ControlPlaneStore,
    Domain,
    Event,
    EventLane,
    NotFoundError,
    Project,
    WorkSession,
)
from harness.discord_channel_map import (
    ChannelDomainMap,
    ChannelResolution,
    ChannelRoutingError,
    DiscordLocation,
    UnmappedDiscordChannelError,
    normalize_domain,
)
from harness.human_decision_policy import (
    ControlledAction,
    HumanDecisionKind,
    HumanDecisionVerdict,
    HumanGateResult,
    HumanResponsibilityPolicy,
    decision_event_type,
    normalize_subject_ref,
)


__all__ = [
    "ChannelDomainMap",
    "ChannelResolution",
    "ChannelRoutingError",
    "ControlledAction",
    "DiscordChannelDispatcher",
    "DiscordDispatchResult",
    "DiscordIngressResult",
    "DiscordLocation",
    "DiscordThreadRoute",
    "DiscordThreadRouter",
    "HumanDecisionKind",
    "HumanDecisionVerdict",
    "HumanGateResult",
    "HumanResponsibilityPolicy",
    "MissingDomainHandlerError",
    "UnmappedDiscordChannelError",
]


class MissingDomainHandlerError(ChannelRoutingError):
    """Raised when a mapped channel has no selected-domain handler."""


@dataclass(frozen=True)
class DiscordThreadRoute:
    resolution: ChannelResolution
    project: Project
    work_session: WorkSession

    @property
    def domain(self) -> Domain:
        return self.resolution.domain


@dataclass(frozen=True)
class DiscordIngressResult:
    route: DiscordThreadRoute
    event: Event


@dataclass(frozen=True)
class DiscordDispatchResult:
    ingress: DiscordIngressResult
    handler_result: Any

    @property
    def domain(self) -> Domain:
        return self.ingress.route.domain

    @property
    def correlation_id(self) -> str:
        return self.ingress.event.event_id


class DiscordChannelDispatcher:
    """Dispatch by configured channel ID, never by message-text inference."""

    def __init__(
        self,
        router: "DiscordThreadRouter",
        handlers: Mapping[Domain | str, Callable[[DiscordIngressResult], Any]],
    ) -> None:
        normalized: dict[Domain, Callable[[DiscordIngressResult], Any]] = {}
        for domain, handler in handlers.items():
            parsed = normalize_domain(domain)
            if parsed not in {Domain.RESEARCH, Domain.KAGGLE}:
                raise ValueError("Discord handlers must target research or kaggle")
            if not callable(handler):
                raise TypeError(f"handler for {parsed.value} must be callable")
            normalized[parsed] = handler
        self.router = router
        self.handlers = normalized

    def dispatch_message(
        self,
        location: DiscordLocation,
        *,
        message_id: str,
        actor_id: str,
        text: str,
        title: str,
        project_id: str | None = None,
    ) -> DiscordDispatchResult:
        ingress = self.router.ingest_message(
            location,
            message_id=message_id,
            actor_id=actor_id,
            text=text,
            title=title,
            project_id=project_id,
        )
        handler = self.handlers.get(ingress.route.domain)
        if handler is None:
            raise MissingDomainHandlerError(
                f"no {ingress.route.domain.value} handler is configured for "
                f"Discord channel {location.channel_id}"
            )
        return DiscordDispatchResult(ingress, handler(ingress))


class DiscordThreadRouter:
    """Bind Discord conversations to domain-scoped Projects and WorkSessions."""

    def __init__(
        self,
        store: ControlPlaneStore,
        channel_domains: ChannelDomainMap,
    ) -> None:
        self.store = store
        self.channel_domains = channel_domains

    @classmethod
    def from_environment(
        cls,
        root: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "DiscordThreadRouter":
        source = dict(os.environ if environ is None else environ)
        if root is not None:
            target = Path(root).expanduser()
        else:
            configured = Path(
                str(source.get("CONTROL_PLANE_DIR") or "control_plane")
            ).expanduser()
            if configured.is_absolute():
                target = configured
            else:
                project_root = Path(
                    str(source.get("PROJECT_ROOT") or ".")
                ).expanduser()
                target = project_root / configured
        return cls(
            ControlPlaneStore(target),
            ChannelDomainMap.from_environment(source),
        )

    def resolve_work_session(
        self,
        location: DiscordLocation,
        *,
        title: str,
        project_id: str | None = None,
    ) -> DiscordThreadRoute:
        resolution = self.channel_domains.resolve(
            location.channel_id,
            parent_channel_id=location.parent_channel_id,
        )
        existing = self.store.find_work_session_by_external_ref(
            location.identity_ref(),
            origin="discord",
        )
        if existing is not None:
            project = self.store.get_project(existing.project_id)
            if project.domain != resolution.domain:
                raise ConflictError(
                    "Discord conversation is already bound to "
                    f"{project.domain.value}, not {resolution.domain.value}"
                )
            if project_id is not None and existing.project_id != project_id:
                raise ConflictError(
                    "Discord conversation is already bound to another project"
                )
            return DiscordThreadRoute(resolution, project, existing)

        project = self._ensure_project(
            location,
            resolution,
            project_id=project_id,
        )
        try:
            session = self.store.create_work_session(
                project.project_id,
                title=title.strip() or f"Discord {resolution.domain.value} session",
                origin="discord",
                external_ref=location.external_ref(),
                metadata={
                    "domain": resolution.domain.value,
                    "route_channel_id": resolution.route_channel_id,
                    "human_responsibility_policy": HumanResponsibilityPolicy.metadata(
                        resolution.domain
                    ),
                },
            )
        except ConflictError:
            # Concurrent Discord deliveries can both miss the initial lookup.
            # The store serializes creation and enforces external-ref uniqueness;
            # the losing caller reloads the canonical binding.
            session = self.store.find_work_session_by_external_ref(
                location.identity_ref(),
                origin="discord",
            )
            if session is None:
                raise
            project = self.store.get_project(session.project_id)
            if project.domain != resolution.domain:
                raise ConflictError(
                    "Discord conversation was concurrently bound to another domain"
                )
            if project_id is not None and session.project_id != project_id:
                raise ConflictError(
                    "Discord conversation was concurrently bound to another project"
                )
        self.store.append_event(
            event_type="discord.route.bound",
            lane=EventLane.CONTROL,
            project_id=project.project_id,
            work_session_id=session.work_session_id,
            actor="system",
            payload={
                "domain": resolution.domain.value,
                "guild_id": location.guild_id,
                "channel_id": location.channel_id,
                "parent_channel_id": location.parent_channel_id,
                "conversation_id": location.conversation_id,
                "route_channel_id": resolution.route_channel_id,
                "inherited_from_parent": resolution.inherited_from_parent,
            },
            idempotency_key=(
                f"discord:{location.guild_id}:{location.conversation_id}:route"
            ),
        )
        return DiscordThreadRoute(resolution, project, session)

    def ingest_message(
        self,
        location: DiscordLocation,
        *,
        message_id: str,
        actor_id: str,
        text: str,
        title: str,
        project_id: str | None = None,
    ) -> DiscordIngressResult:
        actor_id = str(actor_id).strip()
        message_id = str(message_id).strip()
        if not actor_id:
            raise ValueError("actor_id must be non-empty")
        if not message_id.isdigit():
            raise ValueError("message_id must be a numeric Discord snowflake")
        if not text.strip():
            raise ValueError("Discord message text must be non-empty")
        route = self.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        event = self.store.append_event(
            event_type="discord.message.received",
            lane=EventLane.CONTROL,
            project_id=route.project.project_id,
            work_session_id=route.work_session.work_session_id,
            actor=f"discord:{actor_id}",
            payload={
                "domain": route.domain.value,
                "message_id": message_id,
                "guild_id": location.guild_id,
                "channel_id": location.channel_id,
                "parent_channel_id": location.parent_channel_id,
                "conversation_id": location.conversation_id,
                "text": text.strip(),
            },
            idempotency_key=f"discord:{location.guild_id}:{message_id}:message",
        )
        return DiscordIngressResult(route, event)

    def record_human_decision(
        self,
        route: DiscordThreadRoute,
        *,
        kind: HumanDecisionKind | str,
        verdict: HumanDecisionVerdict | str,
        subject_ref: str,
        text: str,
        actor_id: str,
        message_id: str,
        actor_is_human: bool,
    ) -> Event:
        self._validate_route(route)
        normalized_kind = HumanDecisionKind(kind)
        normalized_verdict = HumanDecisionVerdict(verdict)
        if normalized_kind not in HumanResponsibilityPolicy.required_decisions(
            route.domain
        ):
            raise ValueError(
                f"{normalized_kind.value} is not a human decision in "
                f"{route.domain.value} mode"
            )
        if not actor_is_human:
            raise PermissionError("an Agent or Bot cannot satisfy a human decision gate")
        actor_id = str(actor_id).strip()
        message_id = str(message_id).strip()
        if not actor_id:
            raise ValueError("actor_id must be non-empty")
        if not message_id.isdigit():
            raise ValueError("message_id must be a numeric Discord snowflake")
        normalized_subject = normalize_subject_ref(normalized_kind, subject_ref)
        return self.store.append_event(
            event_type=decision_event_type(normalized_kind),
            lane=EventLane.CONTROL,
            project_id=route.project.project_id,
            work_session_id=route.work_session.work_session_id,
            actor=f"discord-human:{actor_id}",
            payload={
                "actor_type": "human",
                "domain": route.domain.value,
                "kind": normalized_kind.value,
                "verdict": normalized_verdict.value,
                "subject_ref": normalized_subject,
                "text": text.strip(),
                "message_id": message_id,
            },
            idempotency_key=(
                f"discord:{route.work_session.work_session_id}:{message_id}:"
                f"decision:{normalized_kind.value}"
            ),
        )

    def check_human_gate(
        self,
        route: DiscordThreadRoute,
        *,
        action: ControlledAction | str,
        subject_ref: str,
    ) -> HumanGateResult:
        self._validate_route(route)
        normalized_action = ControlledAction(action)
        required = HumanResponsibilityPolicy.decision_for_action(
            route.domain,
            normalized_action,
        )
        normalized_subject = normalize_subject_ref(required, subject_ref)
        matching: list[tuple[Event, HumanDecisionVerdict]] = []
        cursor = 0
        while True:
            page = self.store.list_events(
                work_session_id=route.work_session.work_session_id,
                after_sequence=cursor,
                lanes=[EventLane.CONTROL],
                limit=500,
            )
            if not page:
                break
            for event in page:
                cursor = max(cursor, event.sequence)
                if event.event_type != decision_event_type(required):
                    continue
                payload = event.payload
                if payload.get("actor_type") != "human":
                    continue
                if str(payload.get("domain") or "") != route.domain.value:
                    continue
                if str(payload.get("subject_ref") or "") != normalized_subject:
                    continue
                try:
                    verdict = HumanDecisionVerdict(str(payload.get("verdict") or ""))
                except ValueError:
                    continue
                matching.append((event, verdict))
            if len(page) < 500:
                break
        if not matching:
            return HumanGateResult(
                False,
                normalized_action,
                required,
                normalized_subject,
                reason=(
                    f"human decision {required.value} is required for "
                    f"{normalized_action.value}"
                ),
            )
        event, verdict = matching[-1]
        allowed = verdict == HumanDecisionVerdict.ACCEPT
        return HumanGateResult(
            allowed,
            normalized_action,
            required,
            normalized_subject,
            verdict=verdict,
            event_id=event.event_id,
            reason=(
                "human decision accepted"
                if allowed
                else f"latest human decision is {verdict.value}"
            ),
        )

    def _validate_route(self, route: DiscordThreadRoute) -> None:
        project = self.store.get_project(route.project.project_id)
        session = self.store.get_work_session(route.work_session.work_session_id)
        if session.project_id != project.project_id:
            raise ConflictError("Discord route project/session scope is invalid")
        if route.work_session.project_id != project.project_id:
            raise ConflictError("Discord route object references another project")
        if project.domain != route.resolution.domain:
            raise ConflictError("Discord route domain does not match its project")

    def _ensure_project(
        self,
        location: DiscordLocation,
        resolution: ChannelResolution,
        *,
        project_id: str | None,
    ) -> Project:
        target_id = project_id or _default_project_id(location, resolution)
        try:
            project = self.store.get_project(target_id)
        except NotFoundError:
            try:
                return self.store.create_project(
                    name=(
                        f"Discord {resolution.domain.value} channel "
                        f"{resolution.route_channel_id}"
                    ),
                    domain=resolution.domain,
                    project_id=target_id,
                    root_ref=(
                        f"discord://{location.guild_id}/"
                        f"{resolution.route_channel_id}"
                    ),
                    metadata={
                        "guild_id": location.guild_id,
                        "route_channel_id": resolution.route_channel_id,
                        "domain": resolution.domain.value,
                        "human_responsibility_policy": (
                            HumanResponsibilityPolicy.metadata(resolution.domain)
                        ),
                    },
                )
            except ConflictError:
                project = self.store.get_project(target_id)
        if project.domain != resolution.domain:
            raise ConflictError(
                f"project {project.project_id} is {project.domain.value}, "
                f"not {resolution.domain.value}"
            )
        return project


def _default_project_id(
    location: DiscordLocation,
    resolution: ChannelResolution,
) -> str:
    digest = hashlib.sha256(
        (
            f"discord:{location.guild_id}:{resolution.route_channel_id}:"
            f"{resolution.domain.value}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"PRJ-DISCORD-{resolution.domain.value.upper()}-{digest}"
