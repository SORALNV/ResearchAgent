from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from typing import Callable

from harness.commands import Command, CommandContext, CommandResult


class WorkerClosedError(RuntimeError):
    pass


class WorkerQueueFullError(RuntimeError):
    pass


@dataclass
class _WorkItem:
    sequence: int
    command: Command
    context: CommandContext
    future: asyncio.Future[CommandResult]


class AsyncCommandWorker:
    """Serialize orchestrator mutations while blocking work runs off the event loop."""

    def __init__(
        self,
        handler: Callable[[Command, CommandContext], CommandResult],
        *,
        max_queue_size: int = 32,
        name: str = "research-command-worker",
    ) -> None:
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")
        self._handler = handler
        self._max_queue_size = max_queue_size
        self._name = name
        self._queue: asyncio.Queue[_WorkItem | None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sequence = itertools.count(1)
        self._closed = False

    async def start(self) -> None:
        if self._closed:
            raise WorkerClosedError(f"{self._name} is closed")
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError(f"{self._name} cannot move to another event loop")
        if self._task is not None and not self._task.done():
            return
        self._loop = loop
        self._queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._task = loop.create_task(self._run(), name=self._name)

    async def submit(self, command: Command, context: CommandContext) -> CommandResult:
        await self.start()
        if self._closed or self._queue is None or self._loop is None:
            raise WorkerClosedError(f"{self._name} is closed")
        future: asyncio.Future[CommandResult] = self._loop.create_future()
        item = _WorkItem(next(self._sequence), command, context, future)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise WorkerQueueFullError(
                f"{self._name} queue is full ({self._max_queue_size})"
            ) from exc
        return await future

    async def close(self, *, drain: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue is None or self._task is None:
            return
        if drain:
            await self._queue.join()
        else:
            while True:
                try:
                    pending = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if pending is not None and not pending.future.done():
                    pending.future.cancel()
                self._queue.task_done()
        await self._queue.put(None)
        await self._task
        self._task = None

    @property
    def pending_count(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                result = await asyncio.to_thread(self._handler, item.command, item.context)
            except asyncio.CancelledError:
                if not item.future.done():
                    item.future.cancel()
                raise
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            else:
                if not item.future.done():
                    item.future.set_result(result)
            finally:
                self._queue.task_done()
