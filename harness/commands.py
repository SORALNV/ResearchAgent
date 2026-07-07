from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Command:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandContext:
    actor: str = "local"
    source: str = "cli"
    correlation_id: str = field(default_factory=lambda: f"CMD-{uuid.uuid4().hex[:10]}")


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    mode: str | None
    message: str
    data: dict[str, Any] = field(default_factory=dict)

