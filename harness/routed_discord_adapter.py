from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from harness.control_plane import Domain, EventLane
from harness.discord_channel_map import (
    DiscordLocation,
    UnmappedDiscordChannelError,
)
from harness.discord_thread_router import (
    DiscordChannelDispatcher,
    DiscordThreadRoute,
    DiscordThreadRouter,
)
from harness.domain_consultation import DomainConsultationResponse
from harness.human_decision_policy import (
    ControlledAction,
    HumanDecisionKind,
    HumanDecisionVerdict,
    HumanGateResult,
    HumanResponsibilityPolicy,
)


@dataclass(frozen=True)
class RoutedDiscordReply:
    domain: Domain
    work_session_id: str
    message: str
    correlation_id: str
    cached: bool = False


@dataclass(frozen=True)
class RoutedDecisionReply:
    domain: Domain
    work_session_id: str
    kind: HumanDecisionKind
    verdict: HumanDecisionVerdict
    subject_ref: str
    event_id: str


class RoutedDiscordService:
    """Transport-neutral service used by the real Discord Edge and tests."""

    def __init__(
        self,
        router: DiscordThreadRouter,
        dispatcher: DiscordChannelDispatcher,
    ) -> None:
        if dispatcher.router is not router:
            raise ValueError("dispatcher and service must use the same router")
        self.router = router
        self.dispatcher = dispatcher

    def handle_message(
        self,
        location: DiscordLocation,
        *,
        message_id: str,
        actor_id: str,
        text: str,
        title: str,
        project_id: str | None = None,
    ) -> RoutedDiscordReply:
        result = self.dispatcher.dispatch_message(
            location,
            message_id=message_id,
            actor_id=actor_id,
            text=text,
            title=title,
            project_id=project_id,
        )
        handler_result = result.handler_result
        message = _handler_message(handler_result)
        return RoutedDiscordReply(
            domain=result.domain,
            work_session_id=result.ingress.route.work_session.work_session_id,
            message=message,
            correlation_id=result.correlation_id,
            cached=bool(getattr(handler_result, "cached", False)),
        )

    def record_decision(
        self,
        location: DiscordLocation,
        *,
        title: str,
        kind: HumanDecisionKind | str,
        verdict: HumanDecisionVerdict | str,
        subject_ref: str,
        note: str,
        actor_id: str,
        message_id: str,
        actor_is_human: bool,
        project_id: str | None = None,
    ) -> RoutedDecisionReply:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        event = self.router.record_human_decision(
            route,
            kind=kind,
            verdict=verdict,
            subject_ref=subject_ref,
            text=note,
            actor_id=actor_id,
            message_id=message_id,
            actor_is_human=actor_is_human,
        )
        return RoutedDecisionReply(
            domain=route.domain,
            work_session_id=route.work_session.work_session_id,
            kind=HumanDecisionKind(str(event.payload["kind"])),
            verdict=HumanDecisionVerdict(str(event.payload["verdict"])),
            subject_ref=str(event.payload["subject_ref"]),
            event_id=event.event_id,
        )

    def check_gate(
        self,
        location: DiscordLocation,
        *,
        title: str,
        action: ControlledAction | str,
        subject_ref: str,
        project_id: str | None = None,
    ) -> HumanGateResult:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        return self.router.check_human_gate(
            route,
            action=action,
            subject_ref=subject_ref,
        )

    def status(
        self,
        location: DiscordLocation,
        *,
        title: str,
        project_id: str | None = None,
    ) -> str:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        snapshot = self.router.store.snapshot(
            route.work_session.work_session_id,
            event_limit=50,
        )
        jobs = snapshot.get("jobs") or []
        steering = snapshot.get("pending_steering") or []
        decisions = _latest_decisions(route, snapshot.get("events") or [])
        human_tasks = HumanResponsibilityPolicy.human_tasks(route.domain)
        agent_tasks = HumanResponsibilityPolicy.agent_tasks(route.domain)
        return "\n".join(
            [
                f"mode: {route.domain.value}",
                f"work_session: {route.work_session.work_session_id}",
                f"status: {route.work_session.status.value}",
                f"jobs: {len(jobs)} / pending steering: {len(steering)}",
                "人間が確定する判断:",
                *(f"- {item}" for item in human_tasks),
                "直近の人間判断:",
                *(
                    f"- {kind.value}: {decisions.get(kind, '未記録')}"
                    for kind in HumanResponsibilityPolicy.required_decisions(
                        route.domain
                    )
                ),
                "AIの担当:",
                *(f"- {item}" for item in agent_tasks),
            ]
        )


class DomainRoutedDiscordBotAdapter:
    """discord.py Edge that selects Research/Kaggle strictly by channel ID."""

    def __init__(
        self,
        *,
        token: str,
        service: RoutedDiscordService,
        create_threads: bool = True,
        log_channel_id: str | None = None,
    ) -> None:
        self.token = token
        self.service = service
        self.create_threads = bool(create_threads)
        self.log_channel_id = int(log_channel_id) if log_channel_id else None

    def run(self) -> None:
        try:
            import discord
            from discord import app_commands
        except ImportError as exc:
            raise RuntimeError(
                "Install with `pip install -e .[discord]` to run the routed bot."
            ) from exc

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        locks: dict[str, asyncio.Lock] = {}

        async def send_chunks(channel, message: str) -> None:
            for chunk in _chunks(message, 1900):
                await channel.send(chunk)

        async def reply(interaction, message: str) -> None:
            chunks = _chunks(message, 1900)
            if interaction.response.is_done():
                await interaction.followup.send(chunks[0])
            else:
                await interaction.response.send_message(chunks[0])
            for chunk in chunks[1:]:
                await interaction.followup.send(chunk)

        async def log(message: str) -> None:
            if self.log_channel_id is None:
                return
            try:
                channel = client.get_channel(self.log_channel_id)
                if channel is None:
                    channel = await client.fetch_channel(self.log_channel_id)
                await send_chunks(channel, message)
            except Exception:
                return

        def location_from_channel(guild, channel) -> DiscordLocation:
            if guild is None:
                raise ValueError("Discord guild context is required")
            parent_id = getattr(channel, "parent_id", None)
            is_thread = isinstance(channel, discord.Thread)
            return DiscordLocation(
                guild_id=str(guild.id),
                channel_id=str(channel.id),
                parent_channel_id=(
                    str(parent_id) if parent_id is not None else None
                ),
                thread_id=str(channel.id) if is_thread else None,
            )

        async def message_target(message):
            location = location_from_channel(message.guild, message.channel)
            if location.thread_id is not None or not self.create_threads:
                return location, message.channel
            resolution = self.service.router.channel_domains.resolve(
                location.channel_id,
                parent_channel_id=location.parent_channel_id,
            )
            existing_thread = getattr(message, "thread", None)
            if existing_thread is not None:
                thread = existing_thread
            else:
                creator = getattr(message, "create_thread", None)
                if not callable(creator):
                    raise RuntimeError(
                        "mapped parent channel cannot create a Discord thread"
                    )
                thread = await creator(
                    name=_thread_title(
                        resolution.domain,
                        message.content,
                        str(message.id),
                    ),
                    auto_archive_duration=1440,
                )
            return (
                DiscordLocation(
                    guild_id=str(message.guild.id),
                    channel_id=str(thread.id),
                    parent_channel_id=str(message.channel.id),
                    thread_id=str(thread.id),
                ),
                thread,
            )

        def interaction_location(interaction, *, require_thread: bool) -> DiscordLocation:
            location = location_from_channel(
                interaction.guild,
                interaction.channel,
            )
            if require_thread and self.create_threads and location.thread_id is None:
                raise ValueError(
                    "この操作は対象WorkSessionのDiscord Thread内で実行してください。"
                )
            return location

        def require_human(interaction) -> None:
            if bool(getattr(interaction.user, "bot", False)):
                raise PermissionError(
                    "human-authenticated Discord user is required"
                )

        @client.event
        async def on_message(message) -> None:
            if (
                message.author.bot
                or not str(message.content or "").strip()
                or str(message.content).startswith("/")
                or message.guild is None
            ):
                return
            try:
                location, target = await message_target(message)
            except UnmappedDiscordChannelError:
                return
            except Exception as exc:
                await send_chunks(
                    message.channel,
                    f"Discord routing failed: {type(exc).__name__}: {exc}",
                )
                await log(
                    f"routing error channel={getattr(message.channel, 'id', '?')}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return

            lock = locks.setdefault(location.conversation_id, asyncio.Lock())
            async with lock:
                try:
                    async with target.typing():
                        result = await asyncio.to_thread(
                            self.service.handle_message,
                            location,
                            message_id=str(message.id),
                            actor_id=str(message.author.id),
                            text=str(message.content),
                            title=_thread_title(
                                self.service.router.channel_domains.resolve(
                                    location.channel_id,
                                    parent_channel_id=location.parent_channel_id,
                                ).domain,
                                str(message.content),
                                str(message.id),
                            ),
                        )
                    await send_chunks(target, result.message)
                except Exception as exc:
                    await send_chunks(
                        target,
                        f"Agent routing error: {type(exc).__name__}: {exc}",
                    )
                    await log(
                        f"handler error conversation={location.conversation_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        agent = app_commands.Group(
            name="agent",
            description="Research/Kaggle共通のWorkSession操作",
        )

        @agent.command(name="mode", description="このチャンネルのDomainを表示します。")
        async def mode(interaction) -> None:
            try:
                location = interaction_location(
                    interaction,
                    require_thread=False,
                )
                resolution = self.service.router.channel_domains.resolve(
                    location.channel_id,
                    parent_channel_id=location.parent_channel_id,
                )
                await reply(
                    interaction,
                    f"mode: {resolution.domain.value}\n"
                    f"route_channel_id: {resolution.route_channel_id}",
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"mode resolution failed: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="status",
            description="WorkSession、Job、人間判断ゲートの状態を表示します。",
        )
        async def status(interaction) -> None:
            await interaction.response.defer(thinking=True)
            try:
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                message = await asyncio.to_thread(
                    self.service.status,
                    location,
                    title=f"Discord WorkSession {location.conversation_id}",
                )
                await reply(interaction, message)
            except Exception as exc:
                await reply(
                    interaction,
                    f"status failed: {type(exc).__name__}: {exc}",
                )

        async def record_decision(
            interaction,
            *,
            kind: HumanDecisionKind,
            subject_ref: str,
            verdict: str,
            note: str,
        ) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                result = await asyncio.to_thread(
                    self.service.record_decision,
                    location,
                    title=f"Discord WorkSession {location.conversation_id}",
                    kind=kind,
                    verdict=_parse_verdict(verdict),
                    subject_ref=subject_ref,
                    note=note,
                    actor_id=str(interaction.user.id),
                    message_id=str(interaction.id),
                    actor_is_human=True,
                )
                await reply(
                    interaction,
                    "\n".join(
                        [
                            f"mode: {result.domain.value}",
                            f"human decision: {result.kind.value}",
                            f"verdict: {result.verdict.value}",
                            f"subject_ref: {result.subject_ref}",
                            f"event_id: {result.event_id}",
                        ]
                    ),
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"decision rejected: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="hypothesis",
            description="試す仮説に対する人間判断を記録します。",
        )
        async def hypothesis(
            interaction,
            subject_ref: str,
            verdict: str,
            note: str,
        ) -> None:
            await record_decision(
                interaction,
                kind=HumanDecisionKind.HYPOTHESIS,
                subject_ref=subject_ref,
                verdict=verdict,
                note=note,
            )

        @agent.command(
            name="interpret",
            description="実験結果に対する人間の解釈を記録します。",
        )
        async def interpret(
            interaction,
            result_ref: str,
            verdict: str,
            interpretation: str,
        ) -> None:
            await record_decision(
                interaction,
                kind=HumanDecisionKind.RESULT_INTERPRETATION,
                subject_ref=result_ref,
                verdict=verdict,
                note=interpretation,
            )

        @agent.command(
            name="submit",
            description="Kaggle submissionの最終判断をSHA-256へ紐づけます。",
        )
        async def submit(
            interaction,
            sha256: str,
            verdict: str,
            note: str,
        ) -> None:
            await record_decision(
                interaction,
                kind=HumanDecisionKind.KAGGLE_SUBMISSION,
                subject_ref=sha256,
                verdict=verdict,
                note=note,
            )

        @agent.command(
            name="paper",
            description="研究結果を論文化するかの人間判断を記録します。",
        )
        async def paper(
            interaction,
            result_ref: str,
            verdict: str,
            note: str,
        ) -> None:
            await record_decision(
                interaction,
                kind=HumanDecisionKind.RESEARCH_PAPER,
                subject_ref=result_ref,
                verdict=verdict,
                note=note,
            )

        @agent.command(
            name="gate",
            description="対象操作に必要な人間判断が揃っているか確認します。",
        )
        async def gate(
            interaction,
            action: str,
            subject_ref: str,
        ) -> None:
            await interaction.response.defer(thinking=True)
            try:
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                result = await asyncio.to_thread(
                    self.service.check_gate,
                    location,
                    title=f"Discord WorkSession {location.conversation_id}",
                    action=ControlledAction(action.strip().lower()),
                    subject_ref=subject_ref,
                )
                await reply(
                    interaction,
                    "\n".join(
                        [
                            f"allowed: {result.allowed}",
                            f"action: {result.action.value}",
                            (
                                "required_decision: "
                                f"{result.required_decision.value}"
                            ),
                            f"subject_ref: {result.subject_ref}",
                            f"verdict: {result.verdict.value if result.verdict else 'none'}",
                            f"reason: {result.reason}",
                        ]
                    ),
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"gate check failed: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="compute_backends",
            description="利用可能なCompute Backendと能力を表示します。",
        )
        async def compute_backends(interaction) -> None:
            await interaction.response.defer(thinking=True)
            try:
                compute = getattr(self.service, "compute", None)
                broker = getattr(compute, "broker", None)
                if broker is None:
                    raise RuntimeError("autonomous compute is not enabled")
                snapshot = await asyncio.to_thread(broker.snapshot)
                lines = ["Compute Backends:"]
                for name, state in snapshot.items():
                    capabilities = state.get("capabilities") or {}
                    lines.append(
                        "- "
                        + str(name)
                        + ": available="
                        + str(bool(state.get("available")))
                        + "; approval_required="
                        + str(bool(state.get("approval_required")))
                        + "; accelerators="
                        + ",".join(capabilities.get("accelerators") or [])
                        + "; gpu_count="
                        + str(capabilities.get("gpu_count") or 0)
                        + "; detail="
                        + str(state.get("detail") or "")
                    )
                await reply(interaction, "\n".join(lines))
            except Exception as exc:
                await reply(
                    interaction,
                    f"backend status failed: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="approve_compute",
            description="課金等で承認待ちのCompute Jobを許可します。",
        )
        async def approve_compute(interaction, job_id: str) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                method = getattr(self.service, "approve_compute", None)
                if not callable(method):
                    raise RuntimeError("compute approval is not enabled")
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                job = await asyncio.to_thread(
                    method,
                    location,
                    title=f"Discord WorkSession {location.conversation_id}",
                    job_id=job_id,
                    actor_id=str(interaction.user.id),
                )
                await reply(
                    interaction,
                    "\n".join(
                        [
                            f"compute approved: {job.job_id}",
                            f"status: {job.status.value}",
                            f"backend: {job.backend_id or '-'}",
                        ]
                    ),
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"compute approval failed: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="cancel_job",
            description="このWorkSessionのCompute Jobを停止します。",
        )
        async def cancel_job(interaction, job_id: str) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                method = getattr(self.service, "cancel_compute", None)
                if not callable(method):
                    raise RuntimeError("compute cancellation is not enabled")
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                job = await asyncio.to_thread(
                    method,
                    location,
                    title=f"Discord WorkSession {location.conversation_id}",
                    job_id=job_id,
                    actor_id=str(interaction.user.id),
                )
                await reply(
                    interaction,
                    "\n".join(
                        [
                            f"compute cancellation recorded: {job.job_id}",
                            f"status: {job.status.value}",
                            f"backend: {job.backend_id or '-'}",
                        ]
                    ),
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"compute cancellation failed: {type(exc).__name__}: {exc}",
                )

        tree.add_command(agent)

        @client.event
        async def on_ready() -> None:
            await tree.sync()
            await client.change_presence(
                status=discord.Status.online,
                activity=discord.Game(name="Research / Kaggle"),
            )
            print(f"Logged in as {client.user}")

        client.run(self.token)


def _handler_message(result: Any) -> str:
    if isinstance(result, DomainConsultationResponse):
        return result.message
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        message = result.get("message")
        if isinstance(message, str):
            return message
    message = getattr(result, "message", None)
    if isinstance(message, str):
        return message
    return str(result)


def _parse_verdict(value: str) -> HumanDecisionVerdict:
    normalized = str(value).strip().lower()
    aliases = {
        "approve": "accept",
        "approved": "accept",
        "yes": "accept",
        "deny": "reject",
        "denied": "reject",
        "rejected": "reject",
        "later": "defer",
        "pending": "defer",
    }
    return HumanDecisionVerdict(aliases.get(normalized, normalized))


def _latest_decisions(
    route: DiscordThreadRoute,
    events: list[Any],
) -> dict[HumanDecisionKind, str]:
    result: dict[HumanDecisionKind, str] = {}
    prefix = "human.decision."
    for raw in events:
        event_type = (
            str(raw.get("event_type") or "")
            if isinstance(raw, Mapping)
            else str(getattr(raw, "event_type", "") or "")
        )
        if not event_type.startswith(prefix):
            continue
        payload = (
            raw.get("payload") or {}
            if isinstance(raw, Mapping)
            else getattr(raw, "payload", {}) or {}
        )
        if str(payload.get("domain") or "") != route.domain.value:
            continue
        try:
            kind = HumanDecisionKind(
                str(payload.get("kind") or event_type[len(prefix):])
            )
            verdict = HumanDecisionVerdict(str(payload.get("verdict") or ""))
        except ValueError:
            continue
        subject = str(payload.get("subject_ref") or "?")
        result[kind] = f"{verdict.value} ({subject})"
    return result


def _thread_title(domain: Domain, content: str, fallback: str) -> str:
    normalized = " ".join(str(content).split())
    prefix = "Kaggle" if domain == Domain.KAGGLE else "Research"
    body = normalized[:80].strip() or fallback
    return f"{prefix}: {body}"[:100]


def _chunks(message: str, max_length: int) -> list[str]:
    text = str(message)
    if len(text) <= max_length:
        return [text]
    return [
        text[index : index + max_length]
        for index in range(0, len(text), max_length)
    ]
