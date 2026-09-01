from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one source marker in {relative}, found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_before_once(relative: str, marker: str, value: str) -> None:
    replace_once(relative, marker, value + marker)


def patch_discord_adapter() -> None:
    path = "harness/natural_channel_discord.py"
    replace_once(
        path,
        "from harness.discord_channel_map import DiscordLocation, UnmappedDiscordChannelError\nfrom harness.discord_markdown import compact_discord_markdown\n",
        "from harness.discord_channel_map import DiscordLocation, UnmappedDiscordChannelError\n"
        "from harness.discord_execution_ui import (\n"
        "    ExecutionThreadRegistry,\n"
        "    build_help_message,\n"
        "    build_job_list_message,\n"
        "    build_readiness_message,\n"
        "    execution_opening_message,\n"
        "    execution_thread_name,\n"
        "    format_job_progress,\n"
        "    job_is_terminal,\n"
        "    job_progress_key,\n"
        ")\n"
        "from harness.discord_markdown import compact_discord_markdown\n",
    )
    replace_once(
        path,
        "        # Kept for configuration compatibility. Channel-native mode never creates\n"
        "        # a new Discord thread implicitly; the user creates/archives channels.\n"
        "        self.create_threads = False\n",
        "        # One configured channel remains one WorkSession. When enabled, each\n"
        "        # explicit execution gets a child Discord Thread used only as its live log.\n"
        "        self.create_threads = bool(create_threads)\n",
    )
    replace_once(
        path,
        "        locks: dict[str, asyncio.Lock] = {}\n"
        "        session_targets: dict[str, Any] = {}\n"
        "        event_loop: asyncio.AbstractEventLoop | None = None\n",
        "        locks: dict[str, asyncio.Lock] = {}\n"
        "        session_targets: dict[str, Any] = {}\n"
        "        execution_registry = ExecutionThreadRegistry(self.service.registry.root)\n"
        "        job_watchers: dict[str, asyncio.Task[Any]] = {}\n"
        "        event_loop: asyncio.AbstractEventLoop | None = None\n",
    )
    replace_once(
        path,
        "        def location_from_channel(guild: Any, channel: Any) -> DiscordLocation:\n",
        "        def raw_location_from_channel(guild: Any, channel: Any) -> DiscordLocation:\n",
    )
    insert_before_once(
        path,
        "        def interaction_location(interaction: Any) -> DiscordLocation:\n",
        "        def location_from_channel(guild: Any, channel: Any) -> DiscordLocation:\n"
        "            raw = raw_location_from_channel(guild, channel)\n"
        "            if isinstance(channel, discord.Thread):\n"
        "                record = execution_registry.get(str(channel.id))\n"
        "                if record is not None:\n"
        "                    return record.parent_location()\n"
        "            return raw\n\n",
    )
    replace_once(
        path,
        "            target = session_targets.get(session_id)\n"
        "            if target is not None:\n"
        "                return target\n"
        "            try:\n",
        "            target = session_targets.get(session_id)\n"
        "            if target is not None:\n"
        "                return target\n"
        "            latest = execution_registry.latest_for_session(session_id)\n"
        "            if latest is not None and latest.status in {\"active\", \"watching\"}:\n"
        "                try:\n"
        "                    target = client.get_channel(int(latest.thread_id))\n"
        "                    if target is None:\n"
        "                        target = await client.fetch_channel(int(latest.thread_id))\n"
        "                    session_targets[session_id] = target\n"
        "                    return target\n"
        "                except Exception:\n"
        "                    pass\n"
        "            try:\n",
    )
    insert_before_once(
        path,
        "        @client.event\n        async def on_message(message: Any) -> None:\n",
        '''        async def watch_job(
            *,
            thread_id: str,
            target: Any,
            job_id: str,
        ) -> None:
            key = f"{thread_id}:{job_id}"
            previous: tuple[str, ...] | None = None
            try:
                while True:
                    job = await asyncio.to_thread(
                        self.service.router.store.get_job,
                        job_id,
                    )
                    runtime_store = getattr(
                        getattr(self.service, "compute", None),
                        "runtime_store",
                        None,
                    )
                    runtime = (
                        await asyncio.to_thread(runtime_store.load, job_id)
                        if runtime_store is not None
                        else None
                    )
                    current = job_progress_key(job, runtime=runtime)
                    if current != previous:
                        await send_chunks(
                            target,
                            format_job_progress(job, runtime=runtime),
                        )
                        previous = current
                    if job_is_terminal(job):
                        break
                    await asyncio.sleep(5)
                record = execution_registry.get(thread_id)
                if record is not None and record.job_ids:
                    jobs = [
                        await asyncio.to_thread(
                            self.service.router.store.get_job,
                            item,
                        )
                        for item in record.job_ids
                    ]
                    if all(job_is_terminal(item) for item in jobs):
                        final_status = (
                            "failed"
                            if any(
                                job_progress_key(item)[0] in {"failed", "cancelled"}
                                for item in jobs
                            )
                            else "completed"
                        )
                        execution_registry.set_status(thread_id, final_status)
            except Exception as exc:
                await log(
                    f"job watcher failed thread={thread_id} job={job_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                job_watchers.pop(key, None)

        def start_job_watcher(
            *,
            thread_id: str,
            target: Any,
            job_id: str,
        ) -> None:
            key = f"{thread_id}:{job_id}"
            current = job_watchers.get(key)
            if current is not None and not current.done():
                return
            job_watchers[key] = asyncio.create_task(
                watch_job(
                    thread_id=thread_id,
                    target=target,
                    job_id=job_id,
                )
            )

        async def create_execution_target(
            message: Any,
            location: DiscordLocation,
            channel_config: Any,
            action_kind: str,
        ) -> tuple[Any, Any | None]:
            opening = execution_opening_message(
                subject=channel_config.subject,
                action_kind=action_kind,
                request_text=str(message.content),
            )
            if not self.create_threads or isinstance(message.channel, discord.Thread):
                await send_chunks(message.channel, opening)
                return message.channel, execution_registry.get(str(message.channel.id))
            try:
                thread = await message.create_thread(
                    name=execution_thread_name(
                        channel_config.subject,
                        action_kind,
                        str(message.id),
                    ),
                    auto_archive_duration=1440,
                )
                record = execution_registry.bind(
                    thread_id=str(thread.id),
                    location=location,
                    work_session_id=channel_config.work_session_id,
                    source_message_id=str(message.id),
                    action_kind=action_kind,
                    subject=channel_config.subject,
                )
                session_targets[channel_config.work_session_id] = thread
                await send_chunks(thread, opening)
                await send_chunks(
                    message.channel,
                    f"**実行用Thread:** {thread.mention}",
                )
                return thread, record
            except Exception as exc:
                await log(
                    f"execution thread creation failed message={message.id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                await send_chunks(
                    message.channel,
                    "**実行用Threadを作成できなかったため、このチャンネルへ進捗を送ります。**",
                )
                await send_chunks(message.channel, opening)
                return message.channel, None

''',
    )
    replace_once(
        path,
        "            await remember_target(location, message.channel)\n"
        "            title = channel.subject\n\n"
        "            action_message = _requires_fresh_turn(str(message.content))\n",
        "            title = channel.subject\n"
        "            action_kind = _execution_action_kind(str(message.content))\n"
        "            action_message = action_kind is not None\n"
        "            progress_target = message.channel\n"
        "            execution_record = None\n"
        "            before_job_ids = {\n"
        "                item.job_id\n"
        "                for item in self.service.router.store.list_jobs(\n"
        "                    work_session_id=channel.work_session_id\n"
        "                )\n"
        "            }\n"
        "            if action_kind is not None:\n"
        "                progress_target, execution_record = await create_execution_target(\n"
        "                    message,\n"
        "                    location,\n"
        "                    channel,\n"
        "                    action_kind,\n"
        "                )\n"
        "                await remember_target(location, progress_target)\n"
        "            else:\n"
        "                await remember_target(location, message.channel)\n",
    )
    replace_once(
        path,
        "                    async with message.channel.typing():\n"
        "                        result = await asyncio.to_thread(\n"
        "                            self.service.handle_message,\n"
        "                            location,\n"
        "                            message_id=str(message.id),\n"
        "                            actor_id=str(message.author.id),\n"
        "                            text=str(message.content),\n"
        "                            title=title,\n"
        "                        )\n"
        "                    await send_chunks(message.channel, result.message)\n",
        "                    async with progress_target.typing():\n"
        "                        result = await asyncio.to_thread(\n"
        "                            self.service.handle_message,\n"
        "                            location,\n"
        "                            message_id=str(message.id),\n"
        "                            actor_id=str(message.author.id),\n"
        "                            text=str(message.content),\n"
        "                            title=title,\n"
        "                        )\n"
        "                    await send_chunks(progress_target, result.message)\n"
        "                    if action_kind is not None:\n"
        "                        after_jobs = self.service.router.store.list_jobs(\n"
        "                            work_session_id=channel.work_session_id\n"
        "                        )\n"
        "                        new_job_ids = [\n"
        "                            item.job_id\n"
        "                            for item in after_jobs\n"
        "                            if item.job_id not in before_job_ids\n"
        "                        ]\n"
        "                        target_id = str(getattr(progress_target, \"id\", message.channel.id))\n"
        "                        if execution_record is not None and new_job_ids:\n"
        "                            execution_registry.bind_jobs(\n"
        "                                execution_record.thread_id,\n"
        "                                new_job_ids,\n"
        "                            )\n"
        "                            execution_registry.set_status(\n"
        "                                execution_record.thread_id,\n"
        "                                \"watching\",\n"
        "                            )\n"
        "                        for job_id in new_job_ids:\n"
        "                            start_job_watcher(\n"
        "                                thread_id=target_id,\n"
        "                                target=progress_target,\n"
        "                                job_id=job_id,\n"
        "                            )\n"
        "                        if execution_record is not None and not new_job_ids:\n"
        "                            execution_registry.set_status(\n"
        "                                execution_record.thread_id,\n"
        "                                \"completed\",\n"
        "                            )\n",
    )
    replace_once(
        path,
        "                except UnmappedDiscordChannelError as exc:\n"
        "                    await send_chunks(message.channel, str(exc))\n"
        "                except Exception as exc:\n"
        "                    await send_chunks(\n"
        "                        message.channel,\n"
        "                        f\"**処理に失敗しました。** `{type(exc).__name__}: {exc}`\",\n"
        "                    )\n"
        "                    await log(\n",
        "                except UnmappedDiscordChannelError as exc:\n"
        "                    await send_chunks(progress_target, str(exc))\n"
        "                except Exception as exc:\n"
        "                    await send_chunks(\n"
        "                        progress_target,\n"
        "                        f\"**処理に失敗しました。** `{type(exc).__name__}: {exc}`\",\n"
        "                    )\n"
        "                    if execution_record is not None:\n"
        "                        execution_registry.set_status(\n"
        "                            execution_record.thread_id,\n"
        "                            \"failed\",\n"
        "                        )\n"
        "                    await log(\n",
    )
    insert_before_once(
        path,
        "        @agent.command(\n            name=\"finish\",\n",
        '''        @agent.command(
            name="help",
            description="現行コマンドと自然文操作の使い方を表示します。",
        )
        async def help_command(interaction: Any) -> None:
            await reply(
                interaction,
                build_help_message(
                    self.service,
                    interaction_location(interaction),
                ),
            )

        @agent.command(
            name="readiness",
            description="Codex、Compute、保存先、Kaggle等の実行準備を確認します。",
        )
        async def readiness(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                message = await asyncio.to_thread(
                    build_readiness_message,
                    self.service,
                    interaction_location(interaction),
                )
                await reply(interaction, message)
            except Exception as exc:
                await reply(
                    interaction,
                    f"**Readiness確認に失敗しました。** `{type(exc).__name__}: {exc}`",
                )

''',
    )
    insert_before_once(
        path,
        "        @agent.command(\n            name=\"approve_compute\",\n",
        '''        job_group = app_commands.Group(
            name="job",
            description="この案件の実験Jobを確認します。",
        )

        @job_group.command(
            name="list",
            description="この案件の実験Jobを一覧表示します。",
        )
        async def job_list(interaction: Any) -> None:
            await interaction.response.defer(thinking=True)
            try:
                message = await asyncio.to_thread(
                    build_job_list_message,
                    self.service,
                    interaction_location(interaction),
                )
                await reply(interaction, message)
            except Exception as exc:
                await reply(
                    interaction,
                    f"**Job一覧取得に失敗しました。** `{type(exc).__name__}: {exc}`",
                )

''',
    )
    replace_once(
        path,
        "        tree.add_command(agent)\n",
        "        agent.add_command(job_group)\n"
        "        tree.add_command(agent)\n",
    )
    replace_once(
        path,
        "            await tree.sync()\n"
        "            await client.change_presence(\n",
        "            await tree.sync()\n"
        "            for record in execution_registry.list():\n"
        "                if record.status != \"watching\":\n"
        "                    continue\n"
        "                try:\n"
        "                    target = client.get_channel(int(record.thread_id))\n"
        "                    if target is None:\n"
        "                        target = await client.fetch_channel(int(record.thread_id))\n"
        "                except Exception:\n"
        "                    continue\n"
        "                session_targets[record.work_session_id] = target\n"
        "                for job_id in record.job_ids:\n"
        "                    start_job_watcher(\n"
        "                        thread_id=record.thread_id,\n"
        "                        target=target,\n"
        "                        job_id=job_id,\n"
        "                    )\n"
        "            await client.change_presence(\n",
    )
    replace_once(
        path,
        "def _requires_fresh_turn(text: str) -> bool:\n"
        "    return (\n"
        "        _explicit_run_intent(text)\n"
        "        or _explicit_submit_intent(text)\n"
        "        or _explicit_paper_intent(text)\n"
        "    )\n",
        "def _execution_action_kind(text: str) -> str | None:\n"
        "    if _explicit_submit_intent(text):\n"
        "        return \"submission\"\n"
        "    if _explicit_paper_intent(text):\n"
        "        return \"paper\"\n"
        "    if _explicit_run_intent(text):\n"
        "        return \"experiment\"\n"
        "    return None\n\n\n"
        "def _requires_fresh_turn(text: str) -> bool:\n"
        "    return _execution_action_kind(text) is not None\n",
    )


def patch_codex_events() -> None:
    path = "harness/codex_app_server_service.py"
    replace_once(
        path,
        "            if item_type == \"commandExecution\":\n"
        "                payload.update(\n",
        "            if item_type == \"agentMessage\":\n"
        "                payload.update(\n"
        "                    {\n"
        "                        \"text\": str(item.get(\"text\") or \"\")[:4000],\n"
        "                        \"phase\": str(item.get(\"phase\") or \"\"),\n"
        "                    }\n"
        "                )\n"
        "            elif item_type == \"commandExecution\":\n"
        "                payload.update(\n",
    )
    replace_once(
        path,
        "    if method == \"item/completed\":\n"
        "        item_type = str(payload.get(\"item_type\") or \"\")\n"
        "        if item_type == \"commandExecution\":\n",
        "    if method == \"item/completed\":\n"
        "        item_type = str(payload.get(\"item_type\") or \"\")\n"
        "        if item_type == \"agentMessage\":\n"
        "            # Only user-facing commentary is forwarded. Raw reasoning\n"
        "            # items and the final answer remain on their existing paths.\n"
        "            text = str(payload.get(\"text\") or \"\").strip()\n"
        "            if str(payload.get(\"phase\") or \"\") == \"commentary\" and text:\n"
        "                return text[:1900]\n"
        "            return None\n"
        "        if item_type == \"commandExecution\":\n",
    )


def patch_configuration_and_docs() -> None:
    replace_once(
        "main.py",
        "                create_threads=_bool_env(\"DISCORD_CREATE_THREADS\", False),\n",
        "                create_threads=_bool_env(\"DISCORD_EXECUTION_THREADS\", True),\n",
    )
    replace_once(
        ".env.example",
        "# In routed mode, parent-channel messages create one dedicated Thread/WorkSession.\n"
        "DISCORD_CREATE_THREADS=false\n",
        "# Explicit execution requests create one child Discord Thread used as a live\n"
        "# progress log while retaining the parent channel's Project/WorkSession/Codex\n"
        "# context. Discord does not support nested Threads, so requests sent inside a\n"
        "# Thread reuse that Thread.\n"
        "DISCORD_EXECUTION_THREADS=true\n"
        "# Historical setting retained for configuration compatibility.\n"
        "DISCORD_CREATE_THREADS=false\n",
    )
    replace_once(
        "docs/natural_channel_workflow.md",
        "`DISCORD_CREATE_THREADS` defaults to `false`. The Edge never creates a new\n"
        "Discord thread merely because a user sends a message. Creating and organizing\n"
        "channels is a Discord-side operation controlled by the user.\n",
        "`DISCORD_EXECUTION_THREADS=true` creates a child Discord Thread only for an\n"
        "explicit execution, submission, or paper-generation request. The configured\n"
        "parent channel remains the single Project, WorkSession, and durable Codex chat;\n"
        "the child Thread is a scoped live log for that one execution. It receives safe\n"
        "user-facing Codex commentary, command/file milestones, the final response, and\n"
        "subsequent Job status changes. It does not receive hidden chain-of-thought or raw\n"
        "reasoning tokens.\n\n"
        "Discord does not support nested Threads. When the configured work context is\n"
        "already a Discord Thread, that Thread is reused as the execution log. The\n"
        "historical `DISCORD_CREATE_THREADS` variable is retained only for configuration\n"
        "compatibility.\n",
    )
    replace_once(
        "docs/natural_channel_workflow.md",
        "/agent setup\n/agent channel\n/agent status\n/agent finish\n",
        "/agent setup\n/agent channel\n/agent status\n/agent help\n/agent readiness\n/agent job list\n/agent finish\n",
    )
    replace_once(
        "tests/test_natural_channel_workflow.py",
        "    assert \"create_threads=_bool_env(\\\"DISCORD_CREATE_THREADS\\\", False)\" in main\n",
        "    assert \"create_threads=_bool_env(\\\"DISCORD_EXECUTION_THREADS\\\", True)\" in main\n"
        "    assert 'name=\"help\"' in adapter\n"
        "    assert 'name=\"readiness\"' in adapter\n"
        "    assert 'name=\"job\"' in adapter and 'name=\"list\"' in adapter\n",
    )


def cleanup() -> None:
    for relative in (
        "tools/discord_execution_threads.patch",
        "tools/apply_discord_execution_threads.py",
        ".github/workflows/apply-discord-execution-threads.yml",
    ):
        (ROOT / relative).unlink(missing_ok=True)


def main() -> None:
    patch_discord_adapter()
    patch_codex_events()
    patch_configuration_and_docs()
    cleanup()


if __name__ == "__main__":
    main()
