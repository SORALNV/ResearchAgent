from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping

from harness.codex_app_server import CodexAppServerBusy
from harness.discord_channel_map import DiscordLocation, UnmappedDiscordChannelError
from harness.discord_markdown import compact_discord_markdown
from harness.natural_channel_service_v2 import (
    _explicit_paper_intent,
    _explicit_run_intent,
    _explicit_submit_intent,
)


class NaturalChannelDiscordBotAdapter:
    """Discord Edge where one channel is one persistent Research/Kaggle chat."""

    def __init__(
        self,
        *,
        token: str,
        service: Any,
        create_threads: bool = False,
        log_channel_id: str | None = None,
    ) -> None:
        self.token = token
        self.service = service
        # Kept for configuration compatibility. Channel-native mode never creates
        # a new Discord thread implicitly; the user creates/archives channels.
        self.create_threads = False
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
            compact = compact_discord_markdown(message, max_chars=24000)
            for chunk in _chunks(compact, 1900):
                await channel.send(chunk)

        async def reply(interaction: Any, message: str) -> None:
            chunks = _chunks(compact_discord_markdown(message, max_chars=24000), 1900)
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
                parent_channel_id=(str(parent_id) if parent_id is not None else None),
                thread_id=str(channel.id) if is_thread else None,
            )

        def interaction_location(interaction: Any) -> DiscordLocation:
            return location_from_channel(interaction.guild, interaction.channel)

        def require_human(interaction: Any) -> None:
            if bool(getattr(interaction.user, "bot", False)):
                raise PermissionError("human-authenticated Discord user is required")

        async def remember_target(location: DiscordLocation, target: Any) -> None:
            channel = self.service.registry.get(location)
            if channel and channel.work_session_id:
                session_targets[channel.work_session_id] = target

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
            location = location_from_channel(message.guild, message.channel)
            channel = self.service.registry.get(location)
            if channel is None:
                await send_chunks(
                    message.channel,
                    "**このチャンネルは未設定です。** `/agent setup`で `research` または `kaggle` と対象を登録してください。",
                )
                return
            if channel.status.value != "active":
                await send_chunks(
                    message.channel,
                    "**この案件は終了済みです。** 新しい案件は新しいDiscordチャンネルでセットしてください。",
                )
                return
            await remember_target(location, message.channel)
            title = channel.subject

            action_message = _requires_fresh_turn(str(message.content))
            try:
                steered = None
                if not action_message:
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
                        message.channel,
                        f"**途中指示を反映しました。** turn `{steered.turn_id}`",
                    )
                    return
            except CodexAppServerBusy:
                pass
            except Exception as exc:
                await log(
                    f"steer error conversation={location.conversation_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

            lock = locks.setdefault(location.conversation_id, asyncio.Lock())
            async with lock:
                try:
                    # Recheck after lock acquisition so a closely arriving message
                    # becomes official turn/steer input rather than a queued turn.
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
                            message.channel,
                            f"**途中指示を反映しました。** turn `{steered.turn_id}`",
                        )
                        return
                    async with message.channel.typing():
                        result = await asyncio.to_thread(
                            self.service.handle_message,
                            location,
                            message_id=str(message.id),
                            actor_id=str(message.author.id),
                            text=str(message.content),
                            title=title,
                        )
                    await send_chunks(message.channel, result.message)
                except UnmappedDiscordChannelError as exc:
                    await send_chunks(message.channel, str(exc))
                except Exception as exc:
                    await send_chunks(
                        message.channel,
                        f"**処理に失敗しました。** `{type(exc).__name__}: {exc}`",
                    )
                    await log(
                        f"handler error conversation={location.conversation_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        agent = app_commands.Group(
            name="agent",
            description="チャンネル単位のResearch/Kaggle Agent操作",
        )

        @agent.command(
            name="setup",
            description="このチャンネルを一つのResearchまたはKaggle案件に割り当てます。",
        )
        async def setup(
            interaction: Any,
            mode: str,
            subject: str,
            target: str = "",
        ) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                location = interaction_location(interaction)
                result = await asyncio.to_thread(
                    self.service.setup_channel,
                    location,
                    mode=mode,
                    subject=subject,
                    target_ref=target,
                    actor_id=str(interaction.user.id),
                )
                session_targets[result.route.work_session.work_session_id] = interaction.channel
                await reply(interaction, result.message())
            except Exception as exc:
                await reply(
                    interaction,
                    f"**セットできませんでした。** `{type(exc).__name__}: {exc}`",
                )

        @agent.command(
            name="channel",
            description="このチャンネルに割り当てた案件を表示します。",
        )
        async def channel(interaction: Any) -> None:
            try:
                await reply(
                    interaction,
                    self.service.channel_info(interaction_location(interaction)),
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"**確認に失敗しました。** `{type(exc).__name__}: {exc}`",
                )

        @agent.command(
            name="status",
            description="案件、Codex turn、実験Jobの状態をまとめて表示します。",
        )
        async def status(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                location = interaction_location(interaction)
                await remember_target(location, interaction.channel)
                message = await asyncio.to_thread(
                    self.service.status,
                    location,
                    title=f"Discord channel {location.conversation_id}",
                )
                await reply(interaction, message)
            except Exception as exc:
                await reply(
                    interaction,
                    f"**状態取得に失敗しました。** `{type(exc).__name__}: {exc}`",
                )

        @agent.command(
            name="finish",
            description="この案件を終了し、内部WorkSessionをアーカイブします。",
        )
        async def finish(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                location = interaction_location(interaction)
                archived = await asyncio.to_thread(
                    self.service.finish_channel,
                    location,
                    actor_id=str(interaction.user.id),
                )
                await reply(
                    interaction,
                    f"**終了しました:** **{archived.subject}** を内部でアーカイブしました。通常チャンネルはDiscord側で手動アーカイブし、Discord Threadならこの後自動アーカイブします。",
                )
                if isinstance(interaction.channel, discord.Thread):
                    try:
                        await interaction.channel.edit(archived=True)
                    except Exception as exc:
                        await log(
                            f"Discord thread archive failed channel={interaction.channel.id}: "
                            f"{type(exc).__name__}: {exc}"
                        )
            except Exception as exc:
                await reply(
                    interaction,
                    f"**終了できませんでした。** `{type(exc).__name__}: {exc}`",
                )

        @agent.command(
            name="codex_status",
            description="この案件のCodex thread、turn、approvalを表示します。",
        )
        async def codex_status(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                location = interaction_location(interaction)
                channel_config = self.service.registry.get(location)
                if channel_config is None:
                    raise KeyError("channel is not configured")
                await remember_target(location, interaction.channel)
                state = await asyncio.to_thread(
                    self.service.codex_status,
                    location,
                    title=channel_config.subject,
                )
                await reply(interaction, _format_codex_status(state))
            except Exception as exc:
                await reply(
                    interaction,
                    f"**Codex状態取得に失敗しました。** `{type(exc).__name__}: {exc}`",
                )

        @agent.command(
            name="steer",
            description="実行中のCodex turnへ途中指示を送ります。",
        )
        async def steer(interaction: Any, instruction: str) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                location = interaction_location(interaction)
                channel_config = self.service.registry.get(location)
                if channel_config is None:
                    raise KeyError("channel is not configured")
                await remember_target(location, interaction.channel)
                result = await asyncio.to_thread(
                    self.service.steer_codex,
                    location,
                    message_id=str(interaction.id),
                    actor_id=str(interaction.user.id),
                    text=instruction,
                    title=channel_config.subject,
                )
                await reply(
                    interaction,
                    f"**途中指示を反映しました。** thread `{result.thread_id}` · turn `{result.turn_id}`",
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"**途中指示に失敗しました。** `{type(exc).__name__}: {exc}`",
                )

        @agent.command(
            name="interrupt",
            description="実行中のCodex turnを停止します。",
        )
        async def interrupt(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                location = interaction_location(interaction)
                channel_config = self.service.registry.get(location)
                if channel_config is None:
                    raise KeyError("channel is not configured")
                result = await asyncio.to_thread(
                    self.service.interrupt_codex,
                    location,
                    title=channel_config.subject,
                    actor_id=str(interaction.user.id),
                    request_id=str(interaction.id),
                )
                await reply(
                    interaction,
                    f"**停止要求を送りました。** thread `{result.thread_id}` · turn `{result.turn_id}`",
                )
            except CodexAppServerBusy as exc:
                await reply(interaction, f"**実行中のturnはありません。** {exc}")
            except Exception as exc:
                await reply(
                    interaction,
                    f"**停止に失敗しました。** `{type(exc).__name__}: {exc}`",
                )

        @agent.command(
            name="codex_approvals",
            description="この案件のCodex承認待ちを表示します。",
        )
        async def codex_approvals(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                location = interaction_location(interaction)
                channel_config = self.service.registry.get(location)
                if channel_config is None:
                    raise KeyError("channel is not configured")
                approvals = await asyncio.to_thread(
                    self.service.pending_codex_approvals,
                    location,
                    title=channel_config.subject,
                )
                if not approvals:
                    await reply(interaction, "**Codex承認待ち:** なし")
                    return
                body = " · ".join(
                    f"`{item.approval_ref}` {item.kind}"
                    for item in approvals[:10]
                )
                await reply(interaction, f"**Codex承認待ち:** {body}")
            except Exception as exc:
                await reply(
                    interaction,
                    f"**承認待ち取得に失敗しました。** `{type(exc).__name__}: {exc}`",
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
                location = interaction_location(interaction)
                channel_config = self.service.registry.get(location)
                if channel_config is None:
                    raise KeyError("channel is not configured")
                result = await asyncio.to_thread(
                    self.service.resolve_codex_approval,
                    location,
                    title=channel_config.subject,
                    approval_ref=approval_ref,
                    decision=decision,
                    actor_id=str(interaction.user.id),
                    request_id=str(interaction.id),
                )
                await reply(
                    interaction,
                    f"**Codex承認を処理しました。** `{result.approval_ref}` → `{result.decision}`",
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"**Codex承認に失敗しました。** `{type(exc).__name__}: {exc}`",
                )

        @agent.command(
            name="compute_backends",
            description="利用可能なCompute Backendを表示します。",
        )
        async def compute_backends(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                broker = getattr(getattr(self.service, "compute", None), "broker", None)
                if broker is None:
                    raise RuntimeError("autonomous compute is not enabled")
                snapshot = await asyncio.to_thread(broker.snapshot)
                body = " · ".join(
                    f"`{name}` available={bool(state.get('available'))} gpu={int((state.get('capabilities') or {}).get('gpu_count') or 0)}"
                    for name, state in snapshot.items()
                )
                await reply(interaction, f"**Compute Backends:** {body or 'なし'}")
            except Exception as exc:
                await reply(
                    interaction,
                    f"**Backend取得に失敗しました。** `{type(exc).__name__}: {exc}`",
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
                location = interaction_location(interaction)
                channel_config = self.service.registry.get(location)
                if channel_config is None:
                    raise KeyError("channel is not configured")
                job = await asyncio.to_thread(
                    method,
                    location,
                    title=channel_config.subject,
                    job_id=job_id,
                    actor_id=str(interaction.user.id),
                )
                await reply(
                    interaction,
                    f"**Compute承認:** Job `{job.job_id}` · `{job.status.value}` · backend `{job.backend_id or '-'}`",
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"**Compute承認に失敗しました。** `{type(exc).__name__}: {exc}`",
                )

        @agent.command(
            name="cancel_job",
            description="この案件のCompute Jobを停止します。",
        )
        async def cancel_job(interaction: Any, job_id: str) -> None:
            await interaction.response.defer(thinking=True)
            try:
                require_human(interaction)
                method = getattr(self.service, "cancel_compute", None)
                if not callable(method):
                    raise RuntimeError("compute cancellation is not enabled")
                location = interaction_location(interaction)
                channel_config = self.service.registry.get(location)
                if channel_config is None:
                    raise KeyError("channel is not configured")
                job = await asyncio.to_thread(
                    method,
                    location,
                    title=channel_config.subject,
                    job_id=job_id,
                    actor_id=str(interaction.user.id),
                )
                await reply(
                    interaction,
                    f"**停止記録:** Job `{job.job_id}` · `{job.status.value}` · backend `{job.backend_id or '-'}`",
                )
            except Exception as exc:
                await reply(
                    interaction,
                    f"**Job停止に失敗しました。** `{type(exc).__name__}: {exc}`",
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
                activity=discord.Game(name="Research / Kaggle channels"),
            )
            print(f"Logged in as {client.user}")

        client.run(self.token)


def _requires_fresh_turn(text: str) -> bool:
    return (
        _explicit_run_intent(text)
        or _explicit_submit_intent(text)
        or _explicit_paper_intent(text)
    )


def _format_codex_status(state: Mapping[str, Any]) -> str:
    active = state.get("active_turn")
    active_text = (
        f"thread `{active.get('thread_id')}` · turn `{active.get('turn_id')}` · `{active.get('status')}`"
        if isinstance(active, Mapping)
        else "なし"
    )
    threads = state.get("threads") or []
    approvals = state.get("pending_approvals") or []
    return (
        f"**Codex App Server:** running `{bool(state.get('running'))}` · active {active_text} · "
        f"threads `{len(threads)}` · approvals `{len(approvals)}`"
    )


def _chunks(message: str, max_length: int) -> list[str]:
    text = str(message)
    if len(text) <= max_length:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        split = remaining.rfind("\n", 0, max_length)
        if split < max_length // 2:
            split = max_length
        chunks.append(remaining[:split].rstrip())
        remaining = remaining[split:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


# Existing tests and external imports may still use the historical class name.
CodexAppServerDiscordBotAdapter = NaturalChannelDiscordBotAdapter
