from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from harness.platform.config import PlatformConfig
from harness.platform.core_client import CoreApiClient, CoreApiError


_STATUS_TAGS = {
    "planning": "Planning",
    "waiting_input": "Waiting Input",
    "queued": "Queued",
    "running": "Running",
    "review": "Review",
    "waiting_approval": "Waiting Approval",
    "completed": "Completed",
    "failed": "Failed",
    "paused": "Paused",
    "cancelled": "Paused",
}


@dataclass
class ThreadRoute:
    work_session_id: str
    project_id: str
    thread_id: int
    live_message_id: int | None = None
    after_sequence: int = 0
    event_task: asyncio.Task[None] | None = None
    update_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class DiscordEdgeBot:
    """Low-power Discord edge that can run on Jetson or beside Core.

    The Edge owns no OpenAI/Kaggle/GPU credentials. It creates one Discord
    thread/forum post per WorkSession, forwards user messages to Core, edits one
    live status card, and posts only milestones/approvals/errors as new messages.
    """

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config
        self.allowed_users = frozenset(config.discord_allowed_user_ids)
        self.routes_by_thread: dict[int, ThreadRoute] = {}
        self.routes_by_session: dict[str, ThreadRoute] = {}
        self._stopping = asyncio.Event()

    def run(self) -> None:
        errors = self.config.validate_edge()
        if errors:
            raise SystemExit("; ".join(errors))
        try:
            import discord
            from discord import app_commands
        except ImportError as exc:
            raise RuntimeError("Install with `pip install -e '.[discord,api]'`") from exc

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        core = CoreApiClient(
            base_url=self.config.core_url,
            token=self.config.core_token,
            timeout_seconds=180,
        )
        edge = self

        async def ensure_allowed_interaction(interaction: discord.Interaction) -> bool:
            if str(interaction.user.id) in edge.allowed_users:
                return True
            if interaction.response.is_done():
                await interaction.followup.send("このBotを操作する権限がありません。", ephemeral=True)
            else:
                await interaction.response.send_message(
                    "このBotを操作する権限がありません。",
                    ephemeral=True,
                )
            return False

        async def create_remote_session(
            interaction: discord.Interaction,
            *,
            domain: str,
            title: str,
            objective: str,
            description: str = "",
            metadata: Mapping[str, Any] | None = None,
        ) -> None:
            if not await ensure_allowed_interaction(interaction):
                return
            await interaction.response.defer(thinking=True, ephemeral=True)
            try:
                project = await core.create_project(
                    domain=domain,
                    title=title,
                    description=description,
                    metadata=metadata,
                )
                session = await core.create_work_session(
                    project_id=str(project["project_id"]),
                    title=title,
                    objective=objective,
                    metadata={"created_from": "discord-edge", **dict(metadata or {})},
                )
                parent = await edge._work_sessions_channel(client)
                thread, starter = await edge._create_thread(
                    parent,
                    title=f"[{session['session_id']}] {title}",
                    content=edge._session_intro(project, session),
                )
                live = await thread.send(edge._format_live_status({
                    "project": project,
                    "work_session": session,
                    "jobs": [],
                    "active_job_count": 0,
                    "pending_steering": [],
                }))
                with contextlib.suppress(Exception):
                    await live.pin(reason="ResearchAgent live status")
                await core.attach_discord_route(
                    str(session["session_id"]),
                    guild_id=str(interaction.guild_id or 0),
                    parent_channel_id=str(parent.id),
                    thread_id=str(thread.id),
                    live_message_id=str(live.id),
                )
                route = ThreadRoute(
                    work_session_id=str(session["session_id"]),
                    project_id=str(project["project_id"]),
                    thread_id=int(thread.id),
                    live_message_id=int(live.id),
                )
                edge._register_route(route)
                route.event_task = asyncio.create_task(
                    edge._event_loop(client, core, route),
                    name=f"discord-events-{route.work_session_id}",
                )
                await interaction.followup.send(
                    f"作業スレッドを作成しました: {thread.jump_url}",
                    ephemeral=True,
                )
                if starter is not None:
                    with contextlib.suppress(Exception):
                        await starter.add_reaction("✅")
            except Exception as exc:
                await interaction.followup.send(
                    f"WorkSession作成に失敗しました: {type(exc).__name__}: {exc}",
                    ephemeral=True,
                )

        agent = app_commands.Group(name="agent", description="ResearchAgent作業セッション")

        @agent.command(name="new", description="研究または実装の専用スレッドを作成します。")
        @app_commands.describe(
            domain="research または kaggle",
            title="スレッド名",
            objective="相談・実行したい内容",
        )
        async def agent_new(
            interaction: discord.Interaction,
            domain: str,
            title: str,
            objective: str,
        ) -> None:
            normalized = domain.strip().lower()
            if normalized not in {"research", "kaggle"}:
                await interaction.response.send_message(
                    "domainは research または kaggle です。",
                    ephemeral=True,
                )
                return
            await create_remote_session(
                interaction,
                domain=normalized,
                title=title,
                objective=objective,
            )

        @agent.command(name="status", description="現在のスレッドの進捗を表示します。")
        async def agent_status(interaction: discord.Interaction) -> None:
            if not await ensure_allowed_interaction(interaction):
                return
            route = edge.routes_by_thread.get(int(interaction.channel_id or 0))
            if route is None:
                await interaction.response.send_message(
                    "このチャンネルはWorkSessionに紐づいていません。",
                    ephemeral=True,
                )
                return
            await interaction.response.defer(thinking=True, ephemeral=True)
            try:
                status = await core.status(route.work_session_id)
                await interaction.followup.send(
                    edge._format_live_status(status),
                    ephemeral=True,
                )
            except Exception as exc:
                await interaction.followup.send(
                    f"Core status error: {exc}",
                    ephemeral=True,
                )

        @agent.command(name="cancel", description="このWorkSessionの実行を停止します。")
        async def agent_cancel(interaction: discord.Interaction) -> None:
            if not await ensure_allowed_interaction(interaction):
                return
            route = edge.routes_by_thread.get(int(interaction.channel_id or 0))
            if route is None:
                await interaction.response.send_message(
                    "このチャンネルはWorkSessionに紐づいていません。",
                    ephemeral=True,
                )
                return
            await interaction.response.defer(thinking=True)
            try:
                result = await core.cancel_session(route.work_session_id)
                await interaction.followup.send(
                    "停止要求を送りました。\n"
                    + json.dumps(result, ensure_ascii=False, indent=2)[:1600]
                )
            except Exception as exc:
                await interaction.followup.send(f"停止に失敗しました: {exc}")

        kg = app_commands.Group(name="kg", description="Kaggle WorkSession")

        @kg.command(name="new", description="Kaggleコンペ専用スレッドを作成します。")
        @app_commands.describe(
            competition_url="Kaggle competition URLまたはslug",
            title="任意の短い名称",
            objective="今回進めたい範囲",
        )
        async def kg_new(
            interaction: discord.Interaction,
            competition_url: str,
            title: str,
            objective: str,
        ) -> None:
            await create_remote_session(
                interaction,
                domain="kaggle",
                title=title,
                objective=objective,
                description=f"Kaggle competition: {competition_url}",
                metadata={"competition_url": competition_url},
            )

        tree.add_command(agent)
        tree.add_command(kg)

        @client.event
        async def on_ready() -> None:
            await tree.sync()
            await edge._restore_routes(client, core)
            print(f"ResearchAgent Edge logged in as {client.user}")

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot or not message.content.strip():
                return
            if str(message.author.id) not in edge.allowed_users:
                return
            route = edge.routes_by_thread.get(int(message.channel.id))
            if route is None:
                return
            content = message.content.strip()
            mode = "auto"
            computer_use_allowed = False
            if content.lower().startswith(("code:", "実装:", "コード:")):
                mode = "code"
                content = content.split(":", 1)[1].strip()
            elif content.lower().startswith(("computer:", "browser:", "ブラウザ:")):
                mode = "computer"
                content = content.split(":", 1)[1].strip()
            async with message.channel.typing():
                try:
                    result = await core.message(
                        route.work_session_id,
                        text=content,
                        actor=str(message.author.id),
                        correlation_id=str(message.id),
                        mode=mode,
                        computer_use_allowed=computer_use_allowed,
                        metadata={
                            "discord_message_id": str(message.id),
                            "discord_thread_id": str(message.channel.id),
                        },
                    )
                except CoreApiError as exc:
                    await message.reply(f"Core API error: {exc}", mention_author=False)
                    return
                except Exception as exc:
                    await message.reply(
                        f"ResearchAgent error: {type(exc).__name__}: {exc}",
                        mention_author=False,
                    )
                    return
            text = str(result.get("message") or "応答なし")
            for chunk in _chunks(text, 1900):
                await message.channel.send(chunk)
            runtime = result.get("runtime")
            if isinstance(runtime, Mapping) and runtime.get("requires_approval"):
                pending = runtime.get("pending_actions") or []
                view = edge._computer_approval_view(
                    discord,
                    core,
                    route,
                    original_text=content,
                    actor=str(message.author.id),
                    pending=pending,
                )
                await message.channel.send(
                    "Computer-useまたは外部操作が承認待ちです。",
                    view=view,
                )

        @client.event
        async def on_disconnect() -> None:
            print("ResearchAgent Edge disconnected; Core jobs continue independently")

        original_close = client.close

        async def close_with_resources() -> None:
            edge._stopping.set()
            for route in list(edge.routes_by_session.values()):
                if route.event_task:
                    route.event_task.cancel()
            await core.close()
            await original_close()

        client.close = close_with_resources  # type: ignore[method-assign]
        client.run(self.config.discord_bot_token)

    async def _work_sessions_channel(self, client):
        channel_id = int(self.config.discord_work_sessions_channel_id or 0)
        channel = client.get_channel(channel_id)
        if channel is None:
            channel = await client.fetch_channel(channel_id)
        return channel

    async def _create_thread(self, parent, *, title: str, content: str):
        name = title[:100]
        if parent.__class__.__name__ == "ForumChannel":
            created = await parent.create_thread(name=name, content=content)
            thread = getattr(created, "thread", created)
            starter = getattr(created, "message", None)
            return thread, starter
        starter = await parent.send(content)
        thread = await starter.create_thread(name=name)
        return thread, starter

    async def _restore_routes(self, client, core: CoreApiClient) -> None:
        try:
            response = await core._request("GET", "/v1/work-sessions", params={"limit": 1000})
        except Exception:
            return
        for raw in response.get("work_sessions", []):
            if not isinstance(raw, Mapping) or not raw.get("discord_thread_id"):
                continue
            route = ThreadRoute(
                work_session_id=str(raw["session_id"]),
                project_id=str(raw["project_id"]),
                thread_id=int(raw["discord_thread_id"]),
                live_message_id=(
                    int(raw["discord_live_message_id"])
                    if raw.get("discord_live_message_id")
                    else None
                ),
            )
            self._register_route(route)
            if route.event_task is None or route.event_task.done():
                route.event_task = asyncio.create_task(
                    self._event_loop(client, core, route),
                    name=f"discord-events-{route.work_session_id}",
                )

    def _register_route(self, route: ThreadRoute) -> None:
        previous = self.routes_by_session.get(route.work_session_id)
        if previous and previous.thread_id != route.thread_id:
            self.routes_by_thread.pop(previous.thread_id, None)
        self.routes_by_thread[route.thread_id] = route
        self.routes_by_session[route.work_session_id] = route

    async def _event_loop(self, client, core: CoreApiClient, route: ThreadRoute) -> None:
        while not self._stopping.is_set():
            try:
                response = await core.events(
                    route.work_session_id,
                    after_sequence=route.after_sequence,
                    limit=500,
                )
                events = response.get("events", [])
                if events:
                    route.after_sequence = int(response.get("last_sequence") or route.after_sequence)
                    await self._deliver_events(client, core, route, events)
                await self._refresh_live_status(client, core, route)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._send_ops(client, f"{route.work_session_id}: event poll failed: {exc}")
            await asyncio.sleep(self.config.discord_event_poll_seconds)

    async def _deliver_events(self, client, core, route: ThreadRoute, events: list[Any]) -> None:
        thread = client.get_channel(route.thread_id)
        if thread is None:
            with contextlib.suppress(Exception):
                thread = await client.fetch_channel(route.thread_id)
        if thread is None:
            return
        for raw in events:
            if not isinstance(raw, Mapping):
                continue
            kind = str(raw.get("kind") or "")
            if kind in {"user_message", "assistant_message", "progress", "log", "heartbeat", "status"}:
                continue
            message = str(raw.get("message") or "")
            payload = raw.get("payload") if isinstance(raw.get("payload"), Mapping) else {}
            if kind == "approval" and payload.get("job_id"):
                view = self._job_approval_view(
                    __import__("discord"),
                    core,
                    route,
                    str(payload["job_id"]),
                )
                await thread.send(f"⏳ {message}", view=view)
            elif kind == "error":
                await thread.send(f"❌ {message[:1800]}")
            elif kind == "artifact":
                await thread.send(f"📦 {message[:1800]}")
            else:
                await thread.send(f"• {message[:1800]}")

    async def _refresh_live_status(self, client, core: CoreApiClient, route: ThreadRoute) -> None:
        async with route.update_lock:
            try:
                status = await core.status(route.work_session_id)
            except Exception:
                return
            thread = client.get_channel(route.thread_id)
            if thread is None:
                with contextlib.suppress(Exception):
                    thread = await client.fetch_channel(route.thread_id)
            if thread is None:
                return
            live = None
            if route.live_message_id:
                with contextlib.suppress(Exception):
                    live = await thread.fetch_message(route.live_message_id)
            if live is None:
                live = await thread.send(self._format_live_status(status))
                route.live_message_id = int(live.id)
                session = status.get("work_session") or {}
                await core.attach_discord_route(
                    route.work_session_id,
                    guild_id=str(getattr(thread.guild, "id", 0)),
                    parent_channel_id=str(getattr(thread, "parent_id", 0)),
                    thread_id=str(thread.id),
                    live_message_id=str(live.id),
                )
            else:
                with contextlib.suppress(Exception):
                    await live.edit(content=self._format_live_status(status))
            await self._apply_forum_tag(thread, status)

    async def _apply_forum_tag(self, thread, status: Mapping[str, Any]) -> None:
        parent = getattr(thread, "parent", None)
        tags = getattr(parent, "available_tags", None)
        if not tags:
            return
        session = status.get("work_session")
        if not isinstance(session, Mapping):
            return
        wanted = _STATUS_TAGS.get(str(session.get("status") or ""))
        if not wanted:
            return
        tag = next((item for item in tags if str(item.name).lower() == wanted.lower()), None)
        if tag is None:
            return
        current = list(getattr(thread, "applied_tags", []) or [])
        if any(getattr(item, "id", None) == getattr(tag, "id", None) for item in current):
            return
        with contextlib.suppress(Exception):
            await thread.edit(applied_tags=[tag], reason="ResearchAgent status update")

    def _job_approval_view(self, discord, core, route: ThreadRoute, job_id: str):
        edge = self

        class JobApprovalView(discord.ui.View):
            def __init__(self) -> None:
                super().__init__(timeout=86400)

            @discord.ui.button(label="承認して実行", style=discord.ButtonStyle.success)
            async def approve(self, interaction, _button) -> None:
                if str(interaction.user.id) not in edge.allowed_users:
                    await interaction.response.send_message("権限がありません。", ephemeral=True)
                    return
                await interaction.response.defer(thinking=True)
                try:
                    result = await core.approve_job(job_id)
                    await interaction.followup.send(
                        f"{job_id}を承認しました。状態: {result.get('status')}"
                    )
                    self.stop()
                except Exception as exc:
                    await interaction.followup.send(f"承認に失敗しました: {exc}")

            @discord.ui.button(label="却下・中止", style=discord.ButtonStyle.danger)
            async def reject(self, interaction, _button) -> None:
                if str(interaction.user.id) not in edge.allowed_users:
                    await interaction.response.send_message("権限がありません。", ephemeral=True)
                    return
                await interaction.response.defer(thinking=True)
                try:
                    await core.cancel_job(job_id)
                    await interaction.followup.send(f"{job_id}を中止しました。")
                    self.stop()
                except Exception as exc:
                    await interaction.followup.send(f"中止に失敗しました: {exc}")

        return JobApprovalView()

    def _computer_approval_view(
        self,
        discord,
        core,
        route: ThreadRoute,
        *,
        original_text: str,
        actor: str,
        pending: Any,
    ):
        edge = self

        class ComputerApprovalView(discord.ui.View):
            def __init__(self) -> None:
                super().__init__(timeout=3600)

            @discord.ui.button(label="Computer-useを承認", style=discord.ButtonStyle.danger)
            async def approve(self, interaction, _button) -> None:
                if str(interaction.user.id) not in edge.allowed_users:
                    await interaction.response.send_message("権限がありません。", ephemeral=True)
                    return
                await interaction.response.defer(thinking=True)
                try:
                    result = await core.message(
                        route.work_session_id,
                        text=original_text,
                        actor=actor,
                        correlation_id=f"computer-approved-{interaction.id}",
                        mode="computer",
                        computer_use_allowed=True,
                        metadata={"approval_source": str(interaction.id), "pending": pending},
                    )
                    await interaction.followup.send(str(result.get("message") or "完了"))
                    self.stop()
                except Exception as exc:
                    await interaction.followup.send(f"Computer-useに失敗しました: {exc}")

            @discord.ui.button(label="却下", style=discord.ButtonStyle.secondary)
            async def reject(self, interaction, _button) -> None:
                if str(interaction.user.id) not in edge.allowed_users:
                    await interaction.response.send_message("権限がありません。", ephemeral=True)
                    return
                await interaction.response.send_message("Computer-useを却下しました。")
                self.stop()

        return ComputerApprovalView()

    async def _send_ops(self, client, message: str) -> None:
        if not self.config.discord_ops_channel_id:
            return
        channel = client.get_channel(int(self.config.discord_ops_channel_id))
        if channel is None:
            with contextlib.suppress(Exception):
                channel = await client.fetch_channel(int(self.config.discord_ops_channel_id))
        if channel is not None:
            with contextlib.suppress(Exception):
                await channel.send(message[:1900])

    @staticmethod
    def _session_intro(project: Mapping[str, Any], session: Mapping[str, Any]) -> str:
        return (
            f"**{session.get('title')}**\n"
            f"Domain: `{project.get('domain')}`\n"
            f"Project: `{project.get('project_id')}`\n"
            f"WorkSession: `{session.get('session_id')}`\n\n"
            f"**目的**\n{session.get('objective')}\n\n"
            "このスレッドでは普通に相談できます。\n"
            "`code:` で始めるとCodexを優先し、`computer:` で始めると承認付きcomputer-use候補になります。\n"
            "補足・制約・別仮説は自然文で送れます。長時間処理はJobとして別Backendへ委譲されます。"
        )

    @staticmethod
    def _format_live_status(status: Mapping[str, Any]) -> str:
        project = status.get("project") if isinstance(status.get("project"), Mapping) else {}
        session = status.get("work_session") if isinstance(status.get("work_session"), Mapping) else {}
        jobs = status.get("jobs") if isinstance(status.get("jobs"), list) else []
        active = [
            item
            for item in jobs
            if isinstance(item, Mapping)
            and str(item.get("status"))
            not in {"completed", "failed", "cancelled"}
        ]
        latest = status.get("latest_event") if isinstance(status.get("latest_event"), Mapping) else {}
        active_lines = []
        for item in active[:5]:
            spec = item.get("spec") if isinstance(item.get("spec"), Mapping) else {}
            active_lines.append(
                f"- `{spec.get('job_id')}` {item.get('status')} / {item.get('current_stage')} "
                f"({round(float(item.get('progress') or 0) * 100)}%) on `{item.get('backend') or 'unassigned'}`"
            )
        if not active_lines:
            active_lines = ["- なし"]
        return "\n".join(
            [
                f"## Live Status — {session.get('session_id', '-')}",
                f"**Domain:** `{project.get('domain', '-')}`",
                f"**状態:** `{session.get('status', '-')}`",
                f"**現在:** `{session.get('current_stage', '-')}`",
                f"**目的:** {session.get('objective', '-')}",
                "",
                "**実行中Job**",
                *active_lines,
                "",
                f"**最新:** {latest.get('message', 'まだイベントはありません')}",
                f"**補足待ち:** {len(status.get('pending_steering') or [])}",
            ]
        )[:1900]


def _chunks(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split = remaining.rfind("\n", 0, limit)
        if split < limit // 2:
            split = limit
        chunks.append(remaining[:split])
        remaining = remaining[split:].lstrip("\n")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent Discord Edge")
    parser.add_argument("--workdir", default=".")
    args = parser.parse_args()
    DiscordEdgeBot(PlatformConfig.from_env(args.workdir)).run()


if __name__ == "__main__":
    main()
