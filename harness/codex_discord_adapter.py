from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping

from harness.codex_app_server import CodexAppServerBusy
from harness.control_plane import Domain
from harness.discord_channel_map import DiscordLocation, UnmappedDiscordChannelError
from harness.human_decision_policy import (
    ControlledAction,
    HumanDecisionKind,
    HumanDecisionVerdict,
)


class CodexAppServerDiscordBotAdapter:
    """Discord front-end for routed Research/Kaggle plus Codex App Server.

    A normal message starts a new Codex turn when the WorkSession is idle. While
    that Discord WorkSession has an active Codex turn, subsequent messages are
    sent through the official ``turn/steer`` method instead of waiting behind the
    original request. Slash commands expose interrupt and approval responses.
    """

    def __init__(
        self,
        *,
        token: str,
        service: Any,
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
                "Install with `pip install -e .[discord]` to run the Discord bot."
            ) from exc

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        locks: dict[str, asyncio.Lock] = {}
        session_targets: dict[str, Any] = {}
        event_loop: asyncio.AbstractEventLoop | None = None

        async def send_chunks(channel: Any, message: str) -> None:
            for chunk in _chunks(message, 1900):
                await channel.send(chunk)

        async def reply(interaction: Any, message: str) -> None:
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

        def location_from_channel(guild: Any, channel: Any) -> DiscordLocation:
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

        async def message_target(message: Any) -> tuple[DiscordLocation, Any]:
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
                        str(message.content),
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

        def interaction_location(
            interaction: Any,
            *,
            require_thread: bool,
        ) -> DiscordLocation:
            location = location_from_channel(
                interaction.guild,
                interaction.channel,
            )
            if require_thread and self.create_threads and location.thread_id is None:
                raise ValueError(
                    "この操作は対象WorkSessionのDiscord Thread内で実行してください。"
                )
            return location

        def require_human(interaction: Any) -> None:
            if bool(getattr(interaction.user, "bot", False)):
                raise PermissionError(
                    "human-authenticated Discord user is required"
                )

        async def bind_target(
            location: DiscordLocation,
            target: Any,
            title: str,
        ) -> Any:
            route = await asyncio.to_thread(
                self.service.router.resolve_work_session,
                location,
                title=title,
            )
            session_targets[route.work_session.work_session_id] = target
            return route

        async def target_for_session(session_id: str) -> Any | None:
            target = session_targets.get(session_id)
            if target is not None:
                return target
            try:
                session = await asyncio.to_thread(
                    self.service.router.store.get_work_session,
                    session_id,
                )
                external = session.external_ref or {}
                target_id = (
                    external.get("conversation_id")
                    or external.get("thread_id")
                    or external.get("channel_id")
                )
                if not target_id:
                    return None
                target = client.get_channel(int(str(target_id)))
                if target is None:
                    target = await client.fetch_channel(int(str(target_id)))
                session_targets[session_id] = target
                return target
            except Exception:
                return None

        async def deliver_codex_event(session_id: str, message: str) -> None:
            target = await target_for_session(session_id)
            if target is None:
                return
            try:
                await send_chunks(target, message)
            except Exception as exc:
                await log(
                    f"Codex event delivery failed session={session_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        def codex_event_sink(session_id: str, message: str) -> None:
            loop = event_loop
            if loop is None or loop.is_closed():
                return
            asyncio.run_coroutine_threadsafe(
                deliver_codex_event(session_id, message),
                loop,
            )

        @client.event
        async def on_message(message: Any) -> None:
            if (
                message.author.bot
                or not str(message.content or "").strip()
                or str(message.content).startswith("/")
                or message.guild is None
            ):
                return
            try:
                location, target = await message_target(message)
                resolution = self.service.router.channel_domains.resolve(
                    location.channel_id,
                    parent_channel_id=location.parent_channel_id,
                )
                title = _thread_title(
                    resolution.domain,
                    str(message.content),
                    str(message.id),
                )
                await bind_target(location, target, title)
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

            try:
                steered = await asyncio.to_thread(
                    self.service.try_steer_codex,
                    location,
                    message_id=str(message.id),
                    actor_id=str(message.author.id),
                    text=str(message.content),
                    title=title,
                )
                if steered is not None:
                    await send_chunks(
                        target,
                        "Codex steer accepted: "
                        f"turn=`{steered.turn_id}`"
                        + (" (cached)" if steered.cached else ""),
                    )
                    return
            except Exception as exc:
                await send_chunks(
                    target,
                    f"Codex steer failed: {type(exc).__name__}: {exc}",
                )
                await log(
                    f"steer error conversation={location.conversation_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return

            lock = locks.setdefault(location.conversation_id, asyncio.Lock())
            async with lock:
                # A prior message may have started a turn while this delivery was
                # waiting for the WorkSession lock. Re-check before starting a new
                # turn so the message becomes official turn/steer input instead.
                try:
                    steered = await asyncio.to_thread(
                        self.service.try_steer_codex,
                        location,
                        message_id=str(message.id),
                        actor_id=str(message.author.id),
                        text=str(message.content),
                        title=title,
                    )
                    if steered is not None:
                        await send_chunks(
                            target,
                            f"Codex steer accepted: turn=`{steered.turn_id}`",
                        )
                        return
                    async with target.typing():
                        result = await asyncio.to_thread(
                            self.service.handle_message,
                            location,
                            message_id=str(message.id),
                            actor_id=str(message.author.id),
                            text=str(message.content),
                            title=title,
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
            description="Research/Kaggle WorkSession and Codex controls",
        )

        @agent.command(name="mode", description="このチャンネルのDomainを表示します。")
        async def mode(interaction: Any) -> None:
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
            description="WorkSession、Job、Codex、人間判断の状態を表示します。",
        )
        async def status(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                title = f"Discord WorkSession {location.conversation_id}"
                await bind_target(location, interaction.channel, title)
                message = await asyncio.to_thread(
                    self.service.status,
                    location,
                    title=title,
                )
                await reply(interaction, message)
            except Exception as exc:
                await reply(
                    interaction,
                    f"status failed: {type(exc).__name__}: {exc}",
                )

        async def record_decision(
            interaction: Any,
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
                title = f"Discord WorkSession {location.conversation_id}"
                await bind_target(location, interaction.channel, title)
                result = await asyncio.to_thread(
                    self.service.record_decision,
                    location,
                    title=title,
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
            interaction: Any,
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
            interaction: Any,
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
            description="Kaggle提出判断を提出CSVのSHA-256へ紐づけます。",
        )
        async def submit(
            interaction: Any,
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
            interaction: Any,
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
            interaction: Any,
            action: str,
            subject_ref: str,
        ) -> None:
            await interaction.response.defer(thinking=True)
            try:
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                title = f"Discord WorkSession {location.conversation_id}"
                result = await asyncio.to_thread(
                    self.service.check_gate,
                    location,
                    title=title,
                    action=ControlledAction(action.strip().lower()),
                    subject_ref=subject_ref,
                )
                await reply(
                    interaction,
                    "\n".join(
                        [
                            f"allowed: {result.allowed}",
                            f"action: {result.action.value}",
                            f"required_decision: {result.required_decision.value}",
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
            name="codex_status",
            description="Codex thread、active turn、approvalを表示します。",
        )
        async def codex_status(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                title = f"Discord WorkSession {location.conversation_id}"
                await bind_target(location, interaction.channel, title)
                state = await asyncio.to_thread(
                    self.service.codex_status,
                    location,
                    title=title,
                )
                await reply(interaction, _format_codex_status(state))
            except Exception as exc:
                await reply(
                    interaction,
                    f"Codex status failed: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="steer",
            description="実行中のCodex turnへ途中指示を送ります。",
        )
        async def steer(interaction: Any, instruction: str) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                title = f"Discord WorkSession {location.conversation_id}"
                await bind_target(location, interaction.channel, title)
                result = await asyncio.to_thread(
                    self.service.steer_codex,
                    location,
                    message_id=str(interaction.id),
                    actor_id=str(interaction.user.id),
                    text=instruction,
                    title=title,
                )
                await reply(
                    interaction,
                    "Codex steer accepted: "
                    f"thread=`{result.thread_id}` turn=`{result.turn_id}`",
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"Codex steer failed: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="interrupt",
            description="実行中のCodex turnを停止します。",
        )
        async def interrupt(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                title = f"Discord WorkSession {location.conversation_id}"
                await bind_target(location, interaction.channel, title)
                result = await asyncio.to_thread(
                    self.service.interrupt_codex,
                    location,
                    title=title,
                    actor_id=str(interaction.user.id),
                    request_id=str(interaction.id),
                )
                await reply(
                    interaction,
                    "Codex interrupt sent: "
                    f"thread=`{result.thread_id}` turn=`{result.turn_id}`"
                    + (" (cached)" if result.cached else ""),
                )
            except CodexAppServerBusy as exc:
                await reply(interaction, f"Codex interrupt not sent: {exc}")
            except Exception as exc:
                await reply(
                    interaction,
                    f"Codex interrupt failed: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="codex_approvals",
            description="このWorkSessionのCodex承認待ち一覧を表示します。",
        )
        async def codex_approvals(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                title = f"Discord WorkSession {location.conversation_id}"
                await bind_target(location, interaction.channel, title)
                approvals = await asyncio.to_thread(
                    self.service.pending_codex_approvals,
                    location,
                    title=title,
                )
                lines = ["Codex pending approvals:"]
                if not approvals:
                    lines.append("- なし")
                for item in approvals:
                    command = str(item.params.get("command") or "").strip()
                    reason = str(item.params.get("reason") or "").strip()
                    lines.append(
                        f"- `{item.approval_ref}`: {item.kind}; "
                        f"turn={item.turn_id}; command={command[:500] or '-'}; "
                        f"reason={reason[:500] or '-'}"
                    )
                await reply(interaction, "\n".join(lines))
            except Exception as exc:
                await reply(
                    interaction,
                    f"Codex approvals failed: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="codex_approval",
            description="Codexのcommand/file approvalへ回答します。",
        )
        async def codex_approval(
            interaction: Any,
            approval_ref: str,
            decision: str,
        ) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                location = interaction_location(
                    interaction,
                    require_thread=True,
                )
                title = f"Discord WorkSession {location.conversation_id}"
                await bind_target(location, interaction.channel, title)
                result = await asyncio.to_thread(
                    self.service.resolve_codex_approval,
                    location,
                    title=title,
                    approval_ref=approval_ref,
                    decision=decision,
                    actor_id=str(interaction.user.id),
                    request_id=str(interaction.id),
                )
                await reply(
                    interaction,
                    "Codex approval resolved: "
                    f"`{result.approval_ref}` decision=`{result.decision}` "
                    f"turn=`{result.turn_id}`",
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"Codex approval failed: {type(exc).__name__}: {exc}",
                )

        @agent.command(
            name="compute_backends",
            description="利用可能なCompute Backendと能力を表示します。",
        )
        async def compute_backends(interaction: Any) -> None:
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
        async def approve_compute(interaction: Any, job_id: str) -> None:
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
        async def cancel_job(interaction: Any, job_id: str) -> None:
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
            nonlocal event_loop
            event_loop = asyncio.get_running_loop()
            setter = getattr(self.service, "set_codex_event_sink", None)
            if callable(setter):
                setter(codex_event_sink)
            await tree.sync()
            await client.change_presence(
                status=discord.Status.online,
                activity=discord.Game(name="Research / Kaggle / Codex"),
            )
            print(f"Logged in as {client.user}")

        client.run(self.token)


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


def _format_codex_status(state: Mapping[str, Any]) -> str:
    lines = [
        "Codex App Server:",
        f"- running: {bool(state.get('running'))}",
    ]
    active = state.get("active_turn")
    if isinstance(active, Mapping):
        lines.append(
            "- active_turn: thread=`"
            + str(active.get("thread_id") or "?")
            + "` turn=`"
            + str(active.get("turn_id") or "?")
            + "` status=`"
            + str(active.get("status") or "?")
            + "`"
        )
    else:
        lines.append("- active_turn: なし")
    threads = state.get("threads") or []
    lines.append(f"- bound_threads: {len(threads)}")
    for item in threads[:10]:
        if isinstance(item, Mapping):
            lines.append(
                "  - `"
                + str(item.get("thread_id") or "?")
                + "` role="
                + str(item.get("role") or "?")
            )
    approvals = state.get("pending_approvals") or []
    lines.append(f"- pending_approvals: {len(approvals)}")
    for item in approvals[:10]:
        if isinstance(item, Mapping):
            lines.append(
                "  - `"
                + str(item.get("approval_ref") or "?")
                + "` "
                + str(item.get("kind") or item.get("method") or "approval")
            )
    return "\n".join(lines)


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
