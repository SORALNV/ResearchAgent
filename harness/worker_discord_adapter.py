from __future__ import annotations

import asyncio

from harness.commands import Command, CommandContext
from harness.discord_adapter import DiscordChannelAdapter, _chunks, bot_display_mode
from harness.worker import AsyncCommandWorker, WorkerQueueFullError


class WorkerDiscordBotAdapter:
    """Discord adapter with a serialized worker and immediate control/read lane."""

    def __init__(
        self,
        orchestrator_factory,
        token: str,
        channel_id: str | None = None,
        important_channel_id: str | None = None,
        log_channel_id: str | None = None,
        worker_queue_size: int = 32,
    ) -> None:
        self.orchestrator_factory = orchestrator_factory
        self.token = token
        self.channel_id = int(channel_id) if channel_id else None
        self.important_channel_id = important_channel_id or channel_id
        self.log_channel_id = log_channel_id
        self.worker_queue_size = max(1, int(worker_queue_size))

    def run(self) -> None:
        try:
            import discord
            from discord import app_commands
        except ImportError as exc:
            raise RuntimeError(
                "Install with `pip install -e .[discord]` to run the real bot."
            ) from exc

        intents = discord.Intents.default()
        intents.message_content = True
        worker_box: dict[str, AsyncCommandWorker] = {}
        orchestrator_box: dict[str, object] = {}

        class Client(discord.Client):
            async def close(client_self) -> None:
                orchestrator = orchestrator_box.get("orchestrator")
                cancel = getattr(orchestrator, "cancel_active", None)
                if callable(cancel):
                    await asyncio.to_thread(cancel, "Discord client shutdown")
                worker = worker_box.get("worker")
                if worker is not None:
                    await worker.close(drain=False)
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
        orchestrator_box["orchestrator"] = orchestrator

        worker = AsyncCommandWorker(
            orchestrator.handle,
            max_queue_size=self.worker_queue_size,
            name="discord-research-worker",
        )
        worker_box["worker"] = worker

        async def reply(interaction: discord.Interaction, message: str) -> None:
            chunks = _chunks(message, 1900)
            if interaction.response.is_done():
                await interaction.followup.send(chunks[0])
            else:
                await interaction.response.send_message(chunks[0])
            for chunk in chunks[1:]:
                await interaction.followup.send(chunk)

        async def sync_presence(override_mode: str | None = None) -> None:
            display_mode = override_mode or bot_display_mode(orchestrator.store.load())
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

        def cancel_active(reason: str) -> int:
            cancel = getattr(orchestrator, "cancel_active", None)
            return int(cancel(reason)) if callable(cancel) else 0

        async def submit(interaction: discord.Interaction, command: Command) -> None:
            try:
                result = await worker.submit(command, _context(interaction))
            except WorkerQueueFullError:
                await reply(
                    interaction,
                    "ResearchAgentの処理キューが満杯です。現在の処理が完了してから再実行してください。",
                )
                return
            except Exception as exc:
                await reply(interaction, f"ResearchAgent worker error: {exc}")
                await sync_presence()
                return
            await reply(interaction, result.message)
            await sync_presence()

        async def handle(interaction: discord.Interaction, command: Command) -> None:
            await interaction.response.defer(thinking=True)

            # Read-only status bypasses the serialized mutation queue. It remains
            # responsive while a long-running research command owns the worker.
            if command.name == "status":
                try:
                    message = await asyncio.to_thread(orchestrator.status)
                except Exception as exc:
                    message = f"ResearchAgent status error: {exc}"
                await reply(interaction, message)
                await sync_presence()
                return

            if command.name in {"pause", "stop"}:
                await asyncio.to_thread(
                    cancel_active,
                    f"Discord /re {command.name}",
                )
            if command.name == "stop":
                await sync_presence("finalizing")
            await submit(interaction, command)

        re = app_commands.Group(name="re", description="研究モード操作")

        @re.command(name="new", description="前テーマを終了し、新しい研究対話を準備します。")
        async def re_new(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("new_session"))

        @re.command(name="plan", description="planモードで通常メッセージの壁打ちを開始します。")
        async def re_plan(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("enter_plan"))

        @re.command(name="start", description="PLANNINGを承認しRESEARCHを開始します。")
        async def re_start(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("start"))

        @re.command(name="status", description="実行ステージ、Agent数、checkpointを表示します。")
        async def re_status(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("status"))

        @re.command(name="pause", description="実行中Agentを停止しセッションを一時停止します。")
        async def re_pause(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("pause"))

        @re.command(name="resume", description="checkpointから研究セッションを再開します。")
        async def re_resume(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("resume"))

        @re.command(name="cancel", description="実行中Agentを直ちに停止しPAUSEDへ移行します。")
        async def re_cancel(interaction: discord.Interaction) -> None:
            await interaction.response.defer(thinking=True)
            count = await asyncio.to_thread(cancel_active, "Discord /re cancel")
            try:
                result = await worker.submit(Command("pause"), _context(interaction))
                detail = result.message
            except Exception as exc:
                detail = f"pause transition error: {exc}"
            await reply(
                interaction,
                f"実行中Agentへ停止要求を送りました。対象プロセス: {count}\n{detail}",
            )
            await sync_presence()

        @re.command(name="search", description="指定queryで既存研究を検索します。")
        async def re_search(interaction: discord.Interaction, query: str) -> None:
            await handle(interaction, Command("search", {"query": query}))

        @re.command(name="papers", description="取得済み文献を一覧表示します。")
        async def re_papers(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("papers"))

        @re.command(name="paper", description="指定文献IDの詳細を表示します。")
        async def re_paper(interaction: discord.Interaction, paper_id: str) -> None:
            await handle(interaction, Command("paper", {"paper_id": paper_id}))

        @re.command(name="eval", description="研究回答・引用・実行成果を評価します。")
        async def re_eval(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("eval"))

        @re.command(name="cost", description="API・Agent・推定tokenを表示します。")
        async def re_cost(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("cost"))

        @re.command(name="doctor", description="設定、sandbox、接続状態を診断します。")
        async def re_doctor(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("doctor"))

        @re.command(name="runs", description="保存済み研究runを表示します。")
        async def re_runs(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("runs"))

        @re.command(name="accept", description="Phase Gateを承認します。")
        async def re_accept(interaction: discord.Interaction, gate_id: str) -> None:
            await handle(interaction, Command("accept", {"gate_id": gate_id}))

        @re.command(name="revise", description="Phase Gateを差し戻します。")
        async def re_revise(
            interaction: discord.Interaction,
            gate_id: str,
            reason: str,
        ) -> None:
            await handle(
                interaction,
                Command("revise", {"gate_id": gate_id, "reason": reason}),
            )

        @re.command(name="approve", description="危険操作またはfail-closed停止を承認します。")
        async def re_approve(
            interaction: discord.Interaction,
            approval_id: str,
        ) -> None:
            await handle(
                interaction,
                Command("approve", {"approval_id": approval_id}),
            )

        @re.command(name="reject", description="承認待ちを却下します。")
        async def re_reject(
            interaction: discord.Interaction,
            approval_id: str,
            reason: str,
        ) -> None:
            await handle(
                interaction,
                Command(
                    "reject",
                    {"approval_id": approval_id, "reason": reason},
                ),
            )

        @re.command(name="stop", description="Agentを停止し研究セッションを終了します。")
        async def re_stop(interaction: discord.Interaction) -> None:
            await handle(interaction, Command("stop"))

        tree.add_command(re)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot or not message.content.strip() or message.content.startswith("/"):
                return
            if self.important_channel_id and str(message.channel.id) != str(self.important_channel_id):
                return
            session = orchestrator.store.load()
            if not session or session.mode.value != "PLANNING" or session.phase != "plan":
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
                except WorkerQueueFullError:
                    await message.channel.send(
                        "ResearchAgentの処理キューが満杯です。現在の処理が完了してから再送してください。"
                    )
                    return
                except Exception as exc:
                    await message.channel.send(f"ResearchAgent worker error: {exc}")
                    return
            for chunk in _chunks(result.message, 1900):
                await message.channel.send(chunk)
            await sync_presence()

        @client.event
        async def on_ready() -> None:
            await worker.start()
            await tree.sync()
            await sync_presence()
            print(f"Logged in as {client.user}")

        client.run(self.token)


class ThreadSafeDiscordChannelAdapter(DiscordChannelAdapter):
    """Output adapter safe to call from the command worker thread."""

    def send(self, message: str, channel: str = "important") -> None:
        channel_id = self.log_channel_id if channel == "log" else self.important_channel_id
        if channel_id is None:
            return
        coroutine = self._send(channel_id, message)
        try:
            loop = self.client.loop
            if loop.is_running():
                loop.call_soon_threadsafe(loop.create_task, coroutine)
                return
        except (AttributeError, RuntimeError):
            pass
        try:
            asyncio.run(coroutine)
        except RuntimeError:
            coroutine.close()
            raise


def _context(interaction) -> CommandContext:
    return CommandContext(
        actor=str(interaction.user.id),
        source="discord",
        correlation_id=str(interaction.id),
    )
