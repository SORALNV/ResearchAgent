from __future__ import annotations

import re
import time
from pathlib import Path


class AgentMemory:
    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = Path(memory_dir)
        self.entries_dir = self.memory_dir / "entries"
        self.index_path = self.memory_dir / "MEMORY.md"

    def save(self, title: str, body: str) -> Path:
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        clean_title = title.strip() or "memory"
        clean_body = body.strip()
        filename = self._unique_filename(clean_title)
        entry_path = self.entries_dir / filename
        entry_path.write_text(f"# {clean_title}\n\n{clean_body}\n", encoding="utf-8")

        summary = self._summary(clean_body)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(f"- [{clean_title}](entries/{filename}) — {summary}\n")
        return entry_path

    def digest(self, max_chars: int = 2000) -> str:
        if not self.index_path.exists():
            return "記憶はまだありません。"
        text = self.index_path.read_text(encoding="utf-8").strip()
        if not text:
            return "記憶はまだありません。"
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def _unique_filename(self, title: str) -> str:
        date_prefix = time.strftime("%Y-%m-%d", time.localtime())
        slug = _slugify(title)
        base = f"{date_prefix}_{slug}"
        candidate = f"{base}.md"
        index = 2
        while (self.entries_dir / candidate).exists():
            candidate = f"{base}-{index}.md"
            index += 1
        return candidate

    @staticmethod
    def _summary(body: str) -> str:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:160]
        return "要点なし"


def _slugify(value: str, max_length: int = 64) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    if not slug:
        slug = "memory"
    return slug[:max_length].strip("-") or "memory"
