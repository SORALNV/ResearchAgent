from __future__ import annotations

import re
from typing import Iterable


_LABEL_ONLY = re.compile(r"^([^#>*`\-][^:：]{0,30})[:：]\s*$")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def compact_discord_markdown(text: str, *, max_chars: int = 12000) -> str:
    """Make prose compact without damaging fenced code blocks.

    Discord renders Markdown but vertical label/value cards become hard to scan.
    Outside code fences, this joins a label with its value, removes repeated blank
    lines, and converts headings to compact bold lead-ins.
    """

    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""
    parts = normalized.split("```")
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if index % 2:
            rendered.append("```" + part + "```")
        else:
            rendered.append(_compact_plain(part))
    result = "".join(rendered).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    if max_chars > 0 and len(result) > max_chars:
        result = result[: max(0, max_chars - 32)].rstrip() + "\n…（以降は省略）"
    return result


def compact_join(parts: Iterable[str]) -> str:
    return compact_discord_markdown("\n\n".join(str(item).strip() for item in parts if str(item).strip()))


def _compact_plain(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        heading = _HEADING.match(stripped)
        if heading:
            output.append(f"**{heading.group(1).strip()}**")
            index += 1
            continue
        label = _LABEL_ONLY.match(stripped)
        if label:
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            if next_index < len(lines):
                value = lines[next_index].strip()
                if not value.startswith(("- ", "* ", "> ", "```", "#")):
                    output.append(f"**{label.group(1).strip()}:** {value}")
                    index = next_index + 1
                    continue
        if not stripped:
            if output and output[-1] != "":
                output.append("")
        else:
            output.append(stripped)
        index += 1
    while output and output[-1] == "":
        output.pop()
    return "\n".join(output)
