from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from harness.application_runtime import ApplicationRuntime
from harness.command_parser import parse_research_command
from harness.commands import Command, CommandContext
from harness.config import HarnessConfig
from harness.discord_adapter import DiscordChannelAdapter, _chunks, bot_display_mode
from harness.thread_bridge import CreatedThread, WorkSessionThreadBridge
from harness.worker import AsyncCommandWorker, WorkerQueueFullError
from harness.work_session_dialogue import WorkSessionDialogueEngine
from harness.work_sessions import infer_domain, make_session_title


class UnifiedDiscordBotAdapter:
    """One Discord client for ResearchAgent, Kaggle, threaded work, and approvals."""

    def __init__(
        self,
        *,
        orchestrator_factory,
        application: ApplicationRuntime,
        harness_config: HarnessConfig,
        token: str,
        channel_id: str | None = None,
        important_channel_id: str | None = None,
        log_channel_id: str | None = None,
        worker_queue_size: int = 32,
    ) -> None:
        self.orchestrator_factory = orchestrator_factory
        self.application = application
        self.harness_config = harness_config
        self.token = token
        self.channel_id = channel_id
        self.important_channel_id = important_channel_id or channel_id
        self.log_channel_id = log_channel_id
        self.worker_queue_size = max(1, int(worker_queue_size))

    def run(self) -> None:
        try:
            import discord
            from discord import app_commands
        except ImportError as exc:
            raise RuntimeError(
                "Install with `pip install -e .[runtime]` to run the unified bot."
            ) from exc

        intents = discord.Intents.default()
        intents.message_content = True
        client_box: dict[str, Any] = {}
        worker_box: dict[str, AsyncCommandWorker] = {}
        bridge_box: dict[str, WorkSessionThreadBridge] = {}

        class Client(discord.Client):
            async def close(client_self) -> None:
                orchestrator = client_box.get("orchestrator")
                cancel = getattr(orchestrator, "cancel_active", None)
                if callable(cancel):
                    await asyncio.to_thread(cancel, "Discord client shutdown")
                worker = worker_box.get("worker")
                if worker is not None:
                    await worker.close(drain=False)
                self.application.close(cancel_running=False)
                await super().close()

        client = Client(intents=intents)
        tree = app_commands.CommandTree(client)
        output_adapter = ThreadSafeDiscordChannelAdapter(
            client,
            important_channel_id=self.important_channel_id,
            log_channel_id=self.log_channel_id,
        )
        try:
            orchestrator = self.orchestrator_factory(output_adapter)
        except TypeError:
            orchestrator = self.orchestrator_factory()
            orchestrator.discord = output_adapter
        client_box["orchestrator"] = orchestrator

        worker = AsyncCommandWorker(
            orchestrator.handle,
            max_queue_size=self.worker_queue_size,
            name="discord-research-worker",
        )
        worker_box["worker"] = worker
        thread_transport = DiscordPyWorkThreadTransport(
            client,
            forum_channel_id=self.application.config.discord_work_forum_id,
        )
        bridge = WorkSessionThreadBridge(
            self.application.registry,
            self.application.work_session_store,
            self.application.work_sessions,
            thread_transport,
        )
        bridge_box["bridge"] = bridge
        self.application.scheduler.subscribe(bridge.on_job_event)
        dialogue = WorkSessionDialogueEngine(
            self.harness_config,
            self.application.work_sessions,
            self.application.work_session_store,
        )

        async def reply(interaction: discord.Interaction, message: str) -> None:
            chunks = _chunks(message or "処理は完了しました。", 1900)
            if interaction.response.is_done():
                await interaction.followup.send(chunks[0])
            else:
                await interaction.response.send_message(chunks[0])
            for chunk in chunks[1:]:
                await interaction.followup.send(chunk)

        async def sync_presence(override_mode: str | None = None) -> None:
            display_mode = override_mode or bot_display_mode(orchestrator.store.load())
            scheduler = self.application.scheduler.snapshot()
            if scheduler.get("active_job_ids"):
                display_mode = "researching"
            status = {
                "Neutral": discord.Status.idle,
                "plan": discord.Status.online,
                "researching": discord.Status.online,
                "blocked": discord.Status.dnd,
                "finalizing": discord.Status.online,
            }.get(display_mode, discord.Status.online)
            await client.change_presence(
                status=status,
                activity=discord.Game(name=f"mode: {display_mode}"),
            )

        def allowed(user_id: int | str, guild_id: int | str | None) -> bool:
            users = self.application.config.discord_allowed_user_ids
            guilds = self.application.config.discord_allowed_guild_ids
            if users and str(user_id) not in users:
                return False
            if guilds and (guild_id is None or str(guild_id) not in guilds):
                return False
            return True

        async def require_allowed(interaction: discord.Interaction) -> bool:
            guild_id = interaction.guild_id
            if allowed(interaction.user.id, guild_id):
                return True
            if interaction.response.is_done():
                await interaction.followup.send("このBotを操作する権限がありません。", ephemeral=True)
            else:
                await interaction.response.send_message(
                    "このBotを操作する権限がありません。",
                    ephemeral=True,
                )
            return False

        def cancel_active(reason: str) -> int:
            cancel = getattr(orchestrator, "cancel_active", None)
            return int(cancel(reason)) if callable(cancel) else 0

        async def submit_re(interaction: discord.Interaction, command: Command) -> None:
            if not await require_allowed(interaction):
                return
            try:
                result = await worker.submit(command, _context(interaction))
            except WorkerQueueFullError:
                await reply(interaction, "ResearchAgentの処理キューが満杯です。")
                return
            except Exception as exc:
                await reply(interaction, f"ResearchAgent worker error: {exc}")
                return
            await reply(interaction, result.message)
            await sync_presence()

        async def handle_re(interaction: discord.Interaction, command: Command) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            if command.name == "status":
                try:
                    message = await asyncio.to_thread(orchestrator.status)
                except Exception as exc:
                    message = f"ResearchAgent status error: {exc}"
                await reply(interaction, message)
                return
            if command.name in {"pause", "stop"}:
                await asyncio.to_thread(cancel_active, f"Discord /re {command.name}")
            await submit_re(interaction, command)

        re = app_commands.Group(name="re", description="研究セッション操作")

        @re.command(name="new", description="新しい研究セッションを準備します。")
        async def re_new(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("new_session"))

        @re.command(name="plan", description="PLANNING壁打ちを開始します。")
        async def re_plan(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("enter_plan"))

        @re.command(name="start", description="研究を開始します。")
        async def re_start(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("start"))

        @re.command(name="status", description="研究の現在状態を表示します。")
        async def re_status(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("status"))

        @re.command(name="pause", description="実行中Agentを止め一時停止します。")
        async def re_pause(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("pause"))

        @re.command(name="resume", description="checkpointから再開します。")
        async def re_resume(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("resume"))

        @re.command(name="cancel", description="実行中Agentを即時停止します。")
        async def re_cancel(interaction: discord.Interaction) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            count = await asyncio.to_thread(cancel_active, "Discord /re cancel")
            await reply(interaction, f"停止要求を送りました。対象Agent: {count}")

        @re.command(name="search", description="論文検索を実行します。")
        async def re_search(interaction: discord.Interaction, query: str) -> None:
            await handle_re(interaction, Command("search", {"query": query}))

        @re.command(name="papers", description="取得済み文献を表示します。")
        async def re_papers(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("papers"))

        @re.command(name="paper", description="文献の詳細を表示します。")
        async def re_paper(interaction: discord.Interaction, paper_id: str) -> None:
            await handle_re(interaction, Command("paper", {"paper_id": paper_id}))

        @re.command(name="eval", description="研究回答・引用・成果を評価します。")
        async def re_eval(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("eval"))

        @re.command(name="cost", description="Agent/APIコスト状況を表示します。")
        async def re_cost(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("cost"))

        @re.command(name="doctor", description="設定・Provider・sandboxを診断します。")
        async def re_doctor(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("doctor"))

        @re.command(name="runs", description="保存済みrunを表示します。")
        async def re_runs(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("runs"))

        @re.command(name="accept", description="Phase Gateを承認します。")
        async def re_accept(interaction: discord.Interaction, gate_id: str) -> None:
            await handle_re(interaction, Command("accept", {"gate_id": gate_id}))

        @re.command(name="revise", description="Phase Gateを差し戻します。")
        async def re_revise(
            interaction: discord.Interaction,
            gate_id: str,
            reason: str,
        ) -> None:
            await handle_re(
                interaction,
                Command("revise", {"gate_id": gate_id, "reason": reason}),
            )

        @re.command(name="approve", description="承認待ち操作を許可します。")
        async def re_approve(interaction: discord.Interaction, approval_id: str) -> None:
            await handle_re(
                interaction,
                Command("approve", {"approval_id": approval_id}),
            )

        @re.command(name="reject", description="承認待ち操作を却下します。")
        async def re_reject(
            interaction: discord.Interaction,
            approval_id: str,
            reason: str,
        ) -> None:
            await handle_re(
                interaction,
                Command(
                    "reject",
                    {"approval_id": approval_id, "reason": reason},
                ),
            )

        @re.command(name="stop", description="研究を終了しレポートを生成します。")
        async def re_stop(interaction: discord.Interaction) -> None:
            await handle_re(interaction, Command("stop"))

        tree.add_command(re)

        agent = app_commands.Group(name="agent", description="WorkSessionとCompute Job操作")

        @agent.command(name="status", description="WorkSessionまたはControl Plane状態を表示します。")
        async def agent_status(
            interaction: discord.Interaction,
            work_session_id: str | None = None,
        ) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                message = (
                    self.application.work_sessions.status_text(work_session_id)
                    if work_session_id
                    else json_text(self.application.status())
                )
            except Exception as exc:
                message = f"status error: {exc}"
            await reply(interaction, message)

        @agent.command(name="cancel", description="Compute Jobを停止します。")
        async def agent_cancel(interaction: discord.Interaction, job_id: str) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                record = await asyncio.to_thread(
                    self.application.scheduler.cancel,
                    job_id,
                    f"Discord request by {interaction.user.id}",
                )
                message = f"`{record.spec.job_id}` へ停止要求を送りました。"
            except Exception as exc:
                message = f"cancel error: {exc}"
            await reply(interaction, message)

        @agent.command(name="steer", description="次checkpointから適用する補足を登録します。")
        async def agent_steer(
            interaction: discord.Interaction,
            work_session_id: str,
            instruction: str,
            job_id: str | None = None,
        ) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                steering = self.application.work_sessions.steer(
                    work_session_id,
                    instruction,
                    job_id=job_id,
                    actor=str(interaction.user.id),
                )
                message = (
                    f"補足 `{steering.steering_id}` を登録しました。\n"
                    f"適用: `{steering.apply_after}`"
                )
            except Exception as exc:
                message = f"steering error: {exc}"
            await reply(interaction, message)

        tree.add_command(agent)

        kg = app_commands.Group(name="kg", description="Kaggle competition操作")

        @kg.command(name="new", description="Kaggleコンペを登録し専用WorkSessionを作ります。")
        async def kg_new(
            interaction: discord.Interaction,
            competition: str,
            title: str | None = None,
        ) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                root = project_root_for(
                    self.application.config.projects_root,
                    competition,
                )
                competition_record, session = await asyncio.to_thread(
                    self.application.kaggle.create_competition,
                    competition,
                    title=title,
                    project_root=root,
                )
                await asyncio.to_thread(
                    bridge.create_and_bind,
                    session.work_session_id,
                    initial_message=(
                        f"Kaggle `{competition_record.competition_slug}` を登録しました。\n"
                        "Rules Gateから開始します。提出は明示承認まで実行しません。"
                    ),
                )
                message = (
                    f"Competition: `{competition_record.competition_slug}`\n"
                    f"Project: `{competition_record.project_id}`\n"
                    f"WorkSession: `{session.work_session_id}`\n"
                    f"Thread: <#{self.application.registry.get_work_session(session.work_session_id).thread_id}>"
                )
            except Exception as exc:
                message = f"Kaggle project creation error: {exc}"
            await reply(interaction, message)

        @kg.command(name="rules_accept", description="コンペルール確認を記録します。")
        async def kg_rules_accept(
            interaction: discord.Interaction,
            project_id: str,
            notes: str = "",
        ) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                record = self.application.kaggle.confirm_rules(
                    project_id,
                    confirmed_by=str(interaction.user.id),
                    notes=notes,
                )
                message = f"Rules Gateを承認しました。次: `{record.phase}`"
            except Exception as exc:
                message = f"rules confirmation error: {exc}"
            await reply(interaction, message)

        @kg.command(name="status", description="Kaggleプロジェクト状態を表示します。")
        async def kg_status(interaction: discord.Interaction, project_id: str) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                message = json_text(self.application.kaggle.status(project_id))
            except Exception as exc:
                message = f"Kaggle status error: {exc}"
            await reply(interaction, message)

        @kg.command(name="experiment", description="仮説からchild experimentを作ります。")
        async def kg_experiment(
            interaction: discord.Interaction,
            project_id: str,
            hypothesis: str,
            parent_experiment_id: str | None = None,
            cv_spec_id: str | None = None,
        ) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                experiment, session = self.application.kaggle.create_experiment(
                    project_id=project_id,
                    hypothesis=hypothesis,
                    parent_experiment_id=parent_experiment_id,
                    cv_spec_id=cv_spec_id,
                )
                await asyncio.to_thread(
                    bridge.create_and_bind,
                    session.work_session_id,
                    initial_message=(
                        f"実験 `{experiment.experiment_id}`\n"
                        f"仮説: {experiment.hypothesis}\n"
                        "既存実験は変更しません。"
                    ),
                )
                message = (
                    f"Experiment: `{experiment.experiment_id}`\n"
                    f"WorkSession: `{session.work_session_id}`\n"
                    f"Thread: <#{self.application.registry.get_work_session(session.work_session_id).thread_id}>"
                )
            except Exception as exc:
                message = f"experiment creation error: {exc}"
            await reply(interaction, message)

        @kg.command(name="run", description="実験を選択Backendへ投入します。")
        async def kg_run(
            interaction: discord.Interaction,
            experiment_id: str,
            source_dir: str,
            kernel_ref: str | None = None,
            backend: str = "auto",
            smoke_test: bool = False,
        ) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            preferences = (
                ()
                if backend == "auto"
                else (backend,)
            )
            try:
                record = self.application.kaggle.queue_experiment(
                    experiment_id,
                    source_dir=source_dir,
                    kernel_ref=kernel_ref,
                    backend_preferences=preferences or (
                        "kaggle_notebook",
                        "remote_gpu",
                        "gpu_vm",
                        "local_cpu",
                    ),
                    resources={"accelerator": "cpu" if smoke_test else "gpu"},
                    smoke_test=smoke_test,
                )
                message = f"Job `{record.spec.job_id}` をキューへ登録しました。"
            except Exception as exc:
                message = f"experiment queue error: {exc}"
            await reply(interaction, message)

        @kg.command(name="submission_approve", description="hash固定済み提出候補を承認します。")
        async def kg_submission_approve(
            interaction: discord.Interaction,
            candidate_id: str,
        ) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            approval_id = f"DISCORD-{interaction.id}"
            try:
                candidate = self.application.kaggle.approve_submission(
                    candidate_id,
                    approval_id,
                )
                message = (
                    f"提出候補 `{candidate.candidate_id}` を承認しました。\n"
                    f"SHA-256: `{candidate.sha256}`\n"
                    "まだKaggleへは送信していません。"
                )
            except Exception as exc:
                message = f"submission approval error: {exc}"
            await reply(interaction, message)

        @kg.command(name="submit", description="承認済みの同一hashだけをKaggleへ提出します。")
        async def kg_submit(
            interaction: discord.Interaction,
            candidate_id: str,
        ) -> None:
            if not await require_allowed(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                candidate = self.application.kaggle.submit_candidate(candidate_id)
                message = (
                    f"提出しました: `{candidate.candidate_id}`\n"
                    f"Reference: `{candidate.kaggle_submission_ref}`"
                )
            except Exception as exc:
                message = f"Kaggle submission error: {exc}"
            await reply(interaction, message)

        tree.add_command(kg)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot or not message.content.strip() or message.content.startswith("/"):
                return
            if not allowed(message.author.id, message.guild.id if message.guild else None):
                return
            channel_id = str(message.channel.id)
            inbox = self.application.config.discord_inbox_channel_id
            if inbox and channel_id == inbox:
                await create_inbox_session(message, bridge)
                return
            session = self.application.registry.find_work_session_by_thread(channel_id)
            if session is not None:
                async with message.channel.typing():
                    try:
                        decision = await asyncio.to_thread(
                            dialogue.apply,
                            session.work_session_id,
                            message.content,
                            actor=str(message.author.id),
                        )
                    except Exception as exc:
                        await message.channel.send(f"WorkSession error: {exc}")
                        return
                for chunk in _chunks(decision.response, 1900):
                    await message.channel.send(chunk)
                await asyncio.to_thread(bridge.refresh, session.work_session_id)
                return
            # Preserve the existing /re plan normal-message path.
            if self.important_channel_id and channel_id != str(self.important_channel_id):
                return
            research_session = orchestrator.store.load()
            if (
                not research_session
                or research_session.mode.value != "PLANNING"
                or research_session.phase != "plan"
            ):
                return
            async with message.channel.typing():
                try:
                    result = await worker.submit(
                        Command("plan_text", {"text": message.content.strip()}),
                        CommandContext(
                            actor=str(message.author.id),
                            source="discord-message",
                            correlation_id=str(message.id),
                        ),
                    )
                except Exception as exc:
                    await message.channel.send(f"ResearchAgent worker error: {exc}")
                    return
            for chunk in _chunks(result.message, 1900):
                await message.channel.send(chunk)

        async def create_inbox_session(message, bridge: WorkSessionThreadBridge) -> None:
            domain = infer_domain(message.content)
            title = make_session_title(message.content, domain)
            try:
                if domain == "kaggle" and "kaggle.com/competitions/" in message.content.lower():
                    root = project_root_for(
                        self.application.config.projects_root,
                        message.content,
                    )
                    competition, session = await asyncio.to_thread(
                        self.application.kaggle.create_competition,
                        message.content,
                        title=title.removeprefix("[KG] "),
                        project_root=root,
                    )
                    initial = (
                        f"受付: {message.content}\n\n"
                        f"Competition: `{competition.competition_slug}`\n"
                        "Rules Gateから開始します。途中経過はこのスレッドへ流します。"
                    )
                else:
                    root = project_root_for(
                        self.application.config.projects_root,
                        message.content,
                    )
                    project, session = await asyncio.to_thread(
                        self.application.work_sessions.create_session,
                        domain=domain,
                        title=title,
                        project_root=root,
                        project_metadata={"initial_request": message.content},
                    )
                    initial = (
                        f"受付: {message.content}\n\n"
                        f"Project: `{project.project_id}`\n"
                        "相談、実行、追加Q、補足をこのスレッドで続けられます。"
                    )
                bound = await asyncio.to_thread(
                    bridge.create_and_bind,
                    session.work_session_id,
                    initial_message=initial,
                )
                await message.reply(
                    f"作業スレッドを作成しました: <#{bound.thread_id}>"
                )
            except Exception as exc:
                await message.reply(f"WorkSession作成に失敗しました: {exc}")

        @client.event
        async def on_ready() -> None:
            await worker.start()
            await tree.sync()
            await sync_presence()
            print(f"Logged in as {client.user}")

        client.run(self.token)


class DiscordPyWorkThreadTransport:
    """Synchronous facade scheduling Discord operations on the client loop."""

    def __init__(self, client, *, forum_channel_id: str | None) -> None:
        self.client = client
        self.forum_channel_id = int(forum_channel_id) if forum_channel_id else None

    def create_thread(
        self,
        *,
        title: str,
        initial_message: str,
        tags: tuple[str, ...] = (),
    ) -> CreatedThread:
        return self._call(self._create_thread(title, initial_message, tags))

    def send(self, thread_id: str, content: str) -> str | None:
        return self._call(self._send(thread_id, content))

    def upsert_live_status(
        self,
        thread_id: str,
        content: str,
        message_id: str | None = None,
    ) -> str:
        return self._call(self._upsert(thread_id, content, message_id))

    def set_tags(self, thread_id: str, tags: tuple[str, ...]) -> None:
        self._call(self._set_tags(thread_id, tags))

    def _call(self, coroutine):
        loop = self.client.loop
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result(timeout=30)

    async def _channel(self, channel_id: int):
        channel = self.client.get_channel(channel_id)
        return channel or await self.client.fetch_channel(channel_id)

    async def _create_thread(
        self,
        title: str,
        initial_message: str,
        tags: tuple[str, ...],
    ) -> CreatedThread:
        if self.forum_channel_id is None:
            raise RuntimeError("DISCORD_WORK_SESSIONS_FORUM_ID is not configured")
        channel = await self._channel(self.forum_channel_id)
        applied_tags = _resolve_tags(channel, tags)
        if hasattr(channel, "available_tags") and hasattr(channel, "create_thread"):
            kwargs: dict[str, Any] = {
                "name": title[:100],
                "content": initial_message[:2000],
            }
            if applied_tags:
                kwargs["applied_tags"] = applied_tags
            created = await channel.create_thread(**kwargs)
            thread = getattr(created, "thread", created)
            return CreatedThread(
                str(getattr(getattr(thread, "guild", None), "id", "")) or None,
                str(channel.id),
                str(thread.id),
                None,
            )
        starter = await channel.send(initial_message[:2000])
        thread = await starter.create_thread(name=title[:100])
        return CreatedThread(
            str(getattr(getattr(thread, "guild", None), "id", "")) or None,
            str(channel.id),
            str(thread.id),
            None,
        )

    async def _send(self, thread_id: str, content: str) -> str:
        thread = await self._channel(int(thread_id))
        message = await thread.send(content[:2000])
        return str(message.id)

    async def _upsert(
        self,
        thread_id: str,
        content: str,
        message_id: str | None,
    ) -> str:
        thread = await self._channel(int(thread_id))
        if message_id:
            try:
                message = await thread.fetch_message(int(message_id))
                await message.edit(content=content[:2000])
                return str(message.id)
            except Exception:
                pass
        message = await thread.send(content[:2000])
        try:
            await message.pin(reason="ResearchAgent live status")
        except Exception:
            pass
        return str(message.id)

    async def _set_tags(self, thread_id: str, tags: tuple[str, ...]) -> None:
        thread = await self._channel(int(thread_id))
        parent = getattr(thread, "parent", None)
        resolved = _resolve_tags(parent, tags)
        if resolved and hasattr(thread, "edit"):
            try:
                await thread.edit(applied_tags=resolved)
            except Exception:
                return


class ThreadSafeDiscordChannelAdapter(DiscordChannelAdapter):
    def send(self, message: str, channel: str = "important") -> None:
        channel_id = self.log_channel_id if channel == "log" else self.important_channel_id
        if channel_id is None:
            return
        coroutine = self._send(channel_id, message)
        loop = self.client.loop
        loop.call_soon_threadsafe(loop.create_task, coroutine)


def _resolve_tags(channel, names: tuple[str, ...]) -> list[Any]:
    available = getattr(channel, "available_tags", None)
    if not available:
        return []
    wanted = {name.lower() for name in names}
    return [tag for tag in available if str(tag.name).lower() in wanted]


def _context(interaction) -> CommandContext:
    return CommandContext(
        actor=str(interaction.user.id),
        source="discord",
        correlation_id=str(interaction.id),
    )


def json_text(value: Any) -> str:
    import json

    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2)[:1800] + "\n```"


def project_root_for(root: Path, text: str) -> Path:
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:10]
    path = root / digest
    path.mkdir(parents=True, exist_ok=True)
    return path
