from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from harness.command_parser import parse_research_command
from harness.commands import Command, CommandContext
from harness.modes import Mode
from harness.state import utc_timestamp


class DiscordAdapter(Protocol):
    def send(self, message: str, channel: str = "important") -> None:
        ...


@dataclass
class FakeDiscordEvent:
    content: str
    user_id: str = "sora"
    channel_id: str = "fake-channel"
    timestamp: str = field(default_factory=utc_timestamp)


@dataclass
class FakeDiscordMessage:
    message: str
    channel: str
    channel_id: str
    timestamp: str = field(default_factory=utc_timestamp)


@dataclass
class FakeDiscordAdapter:
    important_channel_id: str = "important"
    log_channel_id: str = "log"
    messages: list[str] = field(default_factory=list)
    sent_messages: list[FakeDiscordMessage] = field(default_factory=list)
    important_messages: list[str] = field(default_factory=list)
    log_messages: list[str] = field(default_factory=list)
    input_events: list[FakeDiscordEvent] = field(default_factory=list)

    def send(self, message: str, channel: str = "important") -> None:
        channel_id = self.log_channel_id if channel == "log" else self.important_channel_id
        self.sent_messages.append(
            FakeDiscordMessage(message=message, channel=channel, channel_id=channel_id)
        )
        self.messages.append(message)
        if channel == "log":
            self.log_messages.append(message)
        else:
            self.important_messages.append(message)

    def inject(self, orchestrator, content: str, user_id: str = "sora", channel_id: str = "fake-channel"):
        event = FakeDiscordEvent(content=content, user_id=user_id, channel_id=channel_id)
        self.input_events.append(event)
        try:
            command = parse_research_command(content)
        except ValueError as exc:
            self.send(str(exc), channel="important")
            self.send(f"command_parse_error: {exc}", channel="log")
            return None
        before_count = len(self.important_messages)
        result = orchestrator.handle(
            command,
            CommandContext(actor=user_id, source="fake-discord", correlation_id=event.timestamp),
        )
        if len(self.important_messages) == before_count:
            self.send(result.message, channel="important")
        return result

    def inject_message(
        self,
        orchestrator,
        content: str,
        user_id: str = "sora",
        channel_id: str = "fake-channel",
        attachments: list[str] | None = None,
    ):
        event = FakeDiscordEvent(content=content, user_id=user_id, channel_id=channel_id)
        self.input_events.append(event)
        before_count = len(self.important_messages)
        result = orchestrator.handle(
            Command("plan_text", {"text": content, "attachments": attachments or []}),
            CommandContext(actor=user_id, source="fake-discord-message", correlation_id=event.timestamp),
        )
        if len(self.important_messages) == before_count:
            self.send(result.message, channel="important")
        return result


class DiscordChannelAdapter:
    def __init__(
        self,
        client,
        important_channel_id: str | None,
        log_channel_id: str | None,
    ) -> None:
        self.client = client
        self.important_channel_id = int(important_channel_id) if important_channel_id else None
        self.log_channel_id = int(log_channel_id) if log_channel_id else None
        self.important_sent_count = 0
        self._log_buffer: list[str] = []
        self._log_flush_task: asyncio.Task | None = None

    def send(self, message: str, channel: str = "important") -> None:
        channel_id = self.log_channel_id if channel == "log" else self.important_channel_id
        if channel_id is None:
            return
        if channel == "log":
            self._log_buffer.append(message)
            try:
                self._ensure_log_flush_task()
            except (AttributeError, RuntimeError):
                pass
            return
        self.important_sent_count += 1
        try:
            loop = self.client.loop
            loop.create_task(self._send(channel_id, message))
        except RuntimeError:
            asyncio.run(self._send(channel_id, message))

    def _ensure_log_flush_task(self) -> None:
        if self._log_flush_task is not None and not self._log_flush_task.done():
            return
        loop = self.client.loop
        self._log_flush_task = loop.create_task(self._log_flush_loop())

    async def _log_flush_loop(self) -> None:
        while True:
            await asyncio.sleep(2)
            try:
                await self._flush_log_buffer()
            except Exception:
                pass

    async def _flush_log_buffer(self) -> None:
        if not self._log_buffer or self.log_channel_id is None:
            return
        pending = self._log_buffer
        self._log_buffer = []
        chunks, remaining = _drain_log_batch(pending)
        self._log_buffer = remaining + self._log_buffer
        for chunk in chunks:
            try:
                await self._send(self.log_channel_id, chunk)
            except Exception:
                pass

    async def _send(self, channel_id: int, message: str) -> None:
        channel = self.client.get_channel(channel_id) or await self.client.fetch_channel(channel_id)
        for chunk in _chunks(message, 1900):
            await channel.send(chunk)


class DiscordBotAdapter:
    """Thin optional discord.py slash-command adapter for the MVP harness."""

    def __init__(
        self,
        orchestrator_factory,
        token: str,
        channel_id: str | None = None,
        important_channel_id: str | None = None,
        log_channel_id: str | None = None,
    ) -> None:
        self.orchestrator_factory = orchestrator_factory
        self.token = token
        self.channel_id = int(channel_id) if channel_id else None
        self.important_channel_id = important_channel_id or channel_id
        self.log_channel_id = log_channel_id

    def run(self) -> None:
        try:
            import discord
            from discord import app_commands
        except ImportError as exc:
            raise RuntimeError("Install with `pip install -e .[discord]` to run the real bot.") from exc

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        output_adapter = DiscordChannelAdapter(
            client,
            important_channel_id=self.important_channel_id,
            log_channel_id=self.log_channel_id,
        )
        try:
            orchestrator = self.orchestrator_factory(output_adapter)
        except TypeError:
            orchestrator = self.orchestrator_factory()
            orchestrator.discord = output_adapter

        async def _reply(interaction: discord.Interaction, message: str) -> None:
            chunks = _chunks(message, 1900)
            if interaction.response.is_done():
                await interaction.followup.send(chunks[0])
            else:
                await interaction.response.send_message(chunks[0])
            for chunk in chunks[1:]:
                await interaction.followup.send(chunk)

        async def _handle_command(interaction: discord.Interaction, command: Command) -> None:
            await interaction.response.defer(thinking=True)
            if command.name == "stop":
                await _sync_presence("finalizing")
            before_important = output_adapter.important_sent_count
            result = orchestrator.handle(command, _context_from_interaction(interaction))
            if output_adapter.important_sent_count == before_important:
                await _reply(interaction, result.message)
            else:
                await _reply(interaction, "完了しました。")
            await _sync_presence()

        async def _sync_presence(override_mode: str | None = None) -> None:
            display_mode = override_mode or bot_display_mode(orchestrator.store.load())
            status = {
                "Neutral": discord.Status.online,
                "plan": discord.Status.online,
                "researching": discord.Status.online,
                "blocked": discord.Status.dnd,
                "finalizing": discord.Status.online,
            }.get(display_mode, discord.Status.online)
            await client.change_presence(
                status=status,
                activity=discord.Game(name=display_mode),
            )

        re = app_commands.Group(name="re", description="研究モード操作")

        @re.command(name="new", description="前テーマを終了し、新しい研究対話モードを準備します。")
        async def re_new(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("new_session"))

        @re.command(name="plan", description="planモードに切り替え、通常メッセージで壁打ちを開始します。")
        async def re_plan(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("enter_plan"))

        @re.command(name="start", description="PLANNINGを承認し、RESEARCHを開始します。")
        async def re_start(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("start"))

        @re.command(name="status", description="現在の研究セッション状態を表示します。")
        async def re_status(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("status"))

        @re.command(name="stop", description="現在の研究セッションを終了し、レポートを生成します。")
        async def re_stop(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("stop"))

        @re.command(name="approve", description="危険操作やコスト上限の承認待ちAPを許可します。")
        async def re_approve(interaction: discord.Interaction, approval_id: str) -> None:
            await _handle_command(interaction, Command("approve", {"approval_id": approval_id}))

        @re.command(name="reject", description="危険操作やコスト上限の承認待ちAPを却下します。")
        async def re_reject(interaction: discord.Interaction, approval_id: str, reason: str) -> None:
            await _handle_command(interaction, Command("reject", {"approval_id": approval_id, "reason": reason}))

        @re.command(name="accept", description="研究品質確認のPhase Gateを承認します。")
        async def re_accept(interaction: discord.Interaction, gate_id: str) -> None:
            await _handle_command(interaction, Command("accept", {"gate_id": gate_id}))

        @re.command(name="revise", description="Phase Gateを差し戻し、理由を記録します。")
        async def re_revise(interaction: discord.Interaction, gate_id: str, reason: str) -> None:
            await _handle_command(interaction, Command("revise", {"gate_id": gate_id, "reason": reason}))

        @re.command(name="search", description="指定queryで既存研究を検索します。")
        async def re_search(interaction: discord.Interaction, query: str) -> None:
            await _handle_command(interaction, Command("search", {"query": query}))

        @re.command(name="papers", description="取得済み文献の一覧を表示します。")
        async def re_papers(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("papers"))

        @re.command(name="paper", description="論文を執筆・改訂します。P-で始まるIDなら文献詳細を表示します。")
        async def re_paper(interaction: discord.Interaction, instruction: str = "") -> None:
            if instruction.startswith("P-"):
                command = Command("paper", {"paper_id": instruction})
            else:
                command = Command("paper_write", {"instruction": instruction})
            await _handle_command(interaction, command)

        @re.command(name="exps", description="実験一覧を表示")
        async def re_exps(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("exps"))

        @re.command(name="task", description="単発の実験・作業タスクを実行")
        async def re_task(interaction: discord.Interaction, text: str) -> None:
            await _handle_command(interaction, Command("task", {"text": text}))

        @re.command(name="remember", description="エージェントの長期記憶に保存")
        async def re_remember(interaction: discord.Interaction, text: str) -> None:
            await _handle_command(interaction, Command("remember", {"text": text}))

        @re.command(name="cost", description="API呼び出し数や推定トークンなどのコスト状況を表示します。")
        async def re_cost(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("cost"))

        @re.command(name="doctor", description="ResearchAgentの設定と接続状態を診断します。")
        async def re_doctor(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("doctor"))

        @re.command(name="runs", description="保存済み研究runの一覧を表示します。")
        async def re_runs(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("runs"))

        @re.command(name="fork", description="完了済み研究runから続きバージョンを作成します。")
        async def re_fork(interaction: discord.Interaction, version: str = "") -> None:
            await _handle_command(interaction, Command("fork", {"version": version}))

        @re.command(name="eval", description="黄金データセットで簡易評価を実行します。")
        async def re_eval(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("eval"))

        @re.command(name="pause", description="現在の研究セッションを一時停止します。")
        async def re_pause(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("pause"))

        @re.command(name="resume", description="一時停止中の研究セッションを再開します。")
        async def re_resume(interaction: discord.Interaction) -> None:
            await _handle_command(interaction, Command("resume"))

        tree.add_command(re)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot or not message.content.strip() or message.content.startswith("/"):
                return
            if self.important_channel_id and str(message.channel.id) != str(self.important_channel_id):
                return
            session = orchestrator.store.load()
            if not session:
                return
            if session.mode.value == "PLANNING" and session.phase != "plan":
                return
            if session.mode.value not in {"PLANNING", "RESEARCH", "DONE"}:
                return
            attachment_paths = await _save_plan_attachments(message, session)
            before_important = output_adapter.important_sent_count
            async with message.channel.typing():
                result = orchestrator.handle(
                    Command(
                        "plan_text",
                        {
                            "text": message.content.strip() or "添付画像について相談したい",
                            "attachments": attachment_paths,
                        },
                    ),
                    CommandContext(
                        actor=str(message.author.id),
                        source="discord-message",
                        correlation_id=str(message.id),
                    ),
                )
            if output_adapter.important_sent_count == before_important:
                for chunk in _chunks(result.message, 1900):
                    await message.channel.send(chunk)
            await _sync_presence()

        @client.event
        async def on_ready() -> None:
            await tree.sync()
            await _sync_presence()
            print(f"Logged in as {client.user}")

        client.run(self.token)


def _context_from_interaction(interaction) -> CommandContext:
    return CommandContext(
        actor=str(interaction.user.id),
        source="discord",
        correlation_id=str(interaction.id),
    )


def _chunks(message: str, max_length: int) -> list[str]:
    if len(message) <= max_length:
        return [message]
    chunks: list[str] = []
    rest = message
    while len(rest) > max_length:
        cut = rest.rfind("\n", 1, max_length)
        if cut < max_length // 2:
            cut = max_length
        chunks.append(rest[:cut].rstrip("\n"))
        rest = rest[cut:].lstrip("\n")
    if rest:
        chunks.append(rest)
    return chunks


def _drain_log_batch(
    messages: list[str],
    max_length: int = 1900,
    max_chunks: int = 2,
    separator: str = "\n---\n",
) -> tuple[list[str], list[str]]:
    if max_chunks <= 0:
        return [], messages

    chunks: list[str] = []
    current_parts: list[str] = []
    index = 0

    while index < len(messages):
        message = messages[index]

        if len(message) > max_length:
            if current_parts:
                chunks.append(separator.join(current_parts))
                current_parts = []
                if len(chunks) >= max_chunks:
                    return chunks, messages[index:]

            parts = _chunks(message, max_length)
            available = max_chunks - len(chunks)
            chunks.extend(parts[:available])
            if len(parts) > available:
                return chunks, parts[available:] + messages[index + 1 :]

            index += 1
            if len(chunks) >= max_chunks:
                return chunks, messages[index:]
            continue

        candidate_parts = current_parts + [message]
        if len(separator.join(candidate_parts)) <= max_length:
            current_parts = candidate_parts
            index += 1
            continue

        if current_parts:
            chunks.append(separator.join(current_parts))
            current_parts = []
            if len(chunks) >= max_chunks:
                return chunks, messages[index:]
            continue

        chunks.append(message)
        index += 1
        if len(chunks) >= max_chunks:
            return chunks, messages[index:]

    if current_parts and len(chunks) < max_chunks:
        chunks.append(separator.join(current_parts))
    return chunks, []


async def _save_plan_attachments(message, session) -> list[str]:
    saved: list[str] = []
    if not getattr(message, "attachments", None) or not session.research_dir:
        return saved
    attachment_dir = Path(session.research_dir) / "artifacts" / "attachments"
    attachment_dir.mkdir(parents=True, exist_ok=True)
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    for attachment in message.attachments:
        filename = attachment.filename or "attachment"
        suffix = Path(filename).suffix.lower()
        content_type = (getattr(attachment, "content_type", None) or "").lower()
        if suffix not in image_suffixes and not content_type.startswith("image/"):
            continue
        safe_name = "".join(char if char.isalnum() or char in ".-_" else "_" for char in filename)
        path = attachment_dir / f"{message.id}_{safe_name}"
        await attachment.save(path)
        saved.append(str(path))
    return saved


def bot_display_mode(session) -> str:
    if not session or session.mode == Mode.DONE:
        return "Neutral"
    if session.mode == Mode.APPROVAL_BLOCKED:
        return "blocked"
    if session.mode == Mode.RESEARCH:
        return "researching"
    if session.mode == Mode.PLANNING and session.phase == "plan":
        return "plan"
    return "Neutral"
