import asyncio
import threading
import time

import pytest

from harness.commands import Command, CommandContext, CommandResult
from harness.worker import AsyncCommandWorker, WorkerQueueFullError


def test_worker_keeps_event_loop_responsive_and_serializes_mutations():
    async def scenario():
        main_thread = threading.get_ident()
        calls = []

        def blocking_handler(command, context):
            calls.append((command.name, "start", time.monotonic(), threading.get_ident()))
            time.sleep(0.15)
            calls.append((command.name, "end", time.monotonic(), threading.get_ident()))
            return CommandResult(ok=True, mode=None, message=command.name)

        worker = AsyncCommandWorker(blocking_handler, max_queue_size=4)
        heartbeat = 0

        async def tick():
            nonlocal heartbeat
            for _ in range(20):
                await asyncio.sleep(0.01)
                heartbeat += 1

        first, second, _ = await asyncio.gather(
            worker.submit(Command("first"), CommandContext()),
            worker.submit(Command("second"), CommandContext()),
            tick(),
        )
        await worker.close()
        assert first.message == "first"
        assert second.message == "second"
        assert heartbeat == 20
        first_end = next(item[2] for item in calls if item[0] == "first" and item[1] == "end")
        second_start = next(item[2] for item in calls if item[0] == "second" and item[1] == "start")
        assert second_start >= first_end
        assert all(item[3] != main_thread for item in calls)

    asyncio.run(scenario())


def test_worker_rejects_when_bounded_queue_is_full():
    async def scenario():
        release = threading.Event()

        def blocking_handler(command, context):
            release.wait(timeout=2)
            return CommandResult(ok=True, mode=None, message=command.name)

        worker = AsyncCommandWorker(blocking_handler, max_queue_size=1)
        first = asyncio.create_task(worker.submit(Command("first"), CommandContext()))
        await asyncio.sleep(0.02)
        second = asyncio.create_task(worker.submit(Command("second"), CommandContext()))
        await asyncio.sleep(0.02)
        with pytest.raises(WorkerQueueFullError):
            await worker.submit(Command("third"), CommandContext())
        release.set()
        await asyncio.gather(first, second)
        await worker.close()

    asyncio.run(scenario())
