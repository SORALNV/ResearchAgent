from __future__ import annotations

import shlex

from harness.commands import Command


ALIASES = {}


def parse_research_command(text: str) -> Command:
    parts = shlex.split(text)
    if len(parts) < 2 or parts[0] != "/re":
        raise ValueError("expected /re <new|plan|start|status|pause|resume|redirect|idea|search|papers|paper|eval|cost|doctor|runs|accept|revise|approve|reject|stop>")
    prefix = parts[0]
    action = parts[1]
    if action == "new":
        if len(parts) != 2:
            raise ValueError("/re new does not accept extra arguments. Use /re plan next.")
        return Command("new_session")
    if action == "plan":
        if len(parts) != 2:
            raise ValueError("/re plan does not accept text. Send a normal message after switching modes.")
        return Command("enter_plan")
    name = ALIASES.get(action, action)
    rest = parts[2:]
    if action in {"new", "redirect", "idea", "search"}:
        if not rest:
            raise ValueError(f"/re {action} requires <text>")
        key = "query" if action == "search" else "text"
        return Command(name, {key: " ".join(rest)})
    if action == "paper":
        if len(rest) != 1:
            raise ValueError("/re paper requires <paper_id>")
        return Command("paper", {"paper_id": rest[0]})
    if action == "approve":
        if len(rest) != 1:
            raise ValueError("/re approve requires <id>")
        return Command("approve", {"approval_id": rest[0]})
    if action == "accept":
        if len(rest) != 1:
            raise ValueError("/re accept requires <gate_id>")
        return Command("accept", {"gate_id": rest[0]})
    if action == "revise":
        if len(rest) < 2:
            raise ValueError("/re revise requires <gate_id> <reason>")
        return Command("revise", {"gate_id": rest[0], "reason": " ".join(rest[1:])})
    if action == "reject":
        if len(rest) < 2:
            raise ValueError("/re reject requires <id> <reason>")
        return Command("reject", {"approval_id": rest[0], "reason": " ".join(rest[1:])})
    if action in {"start", "status", "pause", "resume", "stop", "papers", "eval", "cost", "doctor", "runs"}:
        if rest:
            raise ValueError(f"/re {action} does not accept extra arguments")
        return Command(name)
    raise ValueError(f"unknown /re command: {action}")
