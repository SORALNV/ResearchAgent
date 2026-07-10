from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentEntry:
    exp_id: str
    name: str
    hypothesis: str
    status: str
    round_id: int
    metrics: dict[str, object]
    dir: str
    created_at: str
    updated_at: str
    success_metric: str = ""
    priority: str = ""


class ExperimentStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read_all(self) -> list[ExperimentEntry]:
        if not self.path.exists():
            return []
        entries: list[ExperimentEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    continue
                entries.append(ExperimentEntry(**data))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return entries

    def append(self, entry: ExperimentEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")

    def upsert(self, entry: ExperimentEntry) -> None:
        entries = self.read_all()
        replaced = False
        updated: list[ExperimentEntry] = []
        for existing in entries:
            if existing.exp_id == entry.exp_id:
                updated.append(entry)
                replaced = True
            else:
                updated.append(existing)
        if not replaced:
            updated.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for item in updated:
                handle.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")

    def next_exp_id(self) -> str:
        max_seen = 0
        pattern = re.compile(r"^EXP-(\d+)$")
        for entry in self.read_all():
            match = pattern.match(entry.exp_id)
            if not match:
                continue
            max_seen = max(max_seen, int(match.group(1)))
        return f"EXP-{max_seen + 1:03d}"

    def planned_backlog(self) -> list[ExperimentEntry]:
        priority_order = {"high": 0, "mid": 1, "low": 2, "": 3}
        return sorted(
            [entry for entry in self.read_all() if entry.status == "planned"],
            key=lambda entry: (
                priority_order.get(entry.priority, priority_order[""]),
                entry.exp_id,
            ),
        )
