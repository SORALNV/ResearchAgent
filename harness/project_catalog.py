from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from harness.state import ResearchSession, utc_timestamp


@dataclass
class VersionInfo:
    version_label: str
    major: int
    minor: int
    session_id: str
    project_name: str
    goal: str
    mode: str
    round_id: int
    met: int
    total: int
    completed_reason: str
    parent_version: str
    path: str


class ProjectCatalog:
    def __init__(self, archive_path: Path) -> None:
        self.archive_path = archive_path

    def scan(self) -> list[VersionInfo]:
        if not self.archive_path.exists():
            return []
        infos: list[VersionInfo] = []
        for path in self.archive_path.iterdir():
            if not path.is_dir():
                continue
            parsed = _parse_version_from_name(path.name)
            if parsed is None:
                continue
            version_label, major, minor = parsed
            state_path = path / "state.json"
            session_id = _parse_session_id(path.name)
            if not state_path.exists():
                infos.append(
                    VersionInfo(
                        version_label=version_label,
                        major=major,
                        minor=minor,
                        session_id=session_id or "unknown",
                        project_name="",
                        goal="state.jsonなし",
                        mode="broken",
                        round_id=0,
                        met=0,
                        total=0,
                        completed_reason="",
                        parent_version="",
                        path=str(path),
                    )
                )
                continue
            try:
                session = ResearchSession.from_dict(
                    json.loads(state_path.read_text(encoding="utf-8"))
                )
            except Exception:
                infos.append(
                    VersionInfo(
                        version_label=version_label,
                        major=major,
                        minor=minor,
                        session_id=session_id or "unknown",
                        project_name="",
                        goal="state.json読込失敗",
                        mode="broken",
                        round_id=0,
                        met=0,
                        total=0,
                        completed_reason="",
                        parent_version="",
                        path=str(path),
                    )
                )
                continue
            criteria = session.last_criteria_status or []
            infos.append(
                VersionInfo(
                    version_label=session.version_label or version_label,
                    major=major,
                    minor=minor,
                    session_id=session.session_id,
                    project_name=session.project_name,
                    goal=session.research_goal,
                    mode=session.mode.value.lower(),
                    round_id=session.round_id,
                    met=sum(1 for item in criteria if str(item.get("status", "")).lower() == "met"),
                    total=len(criteria),
                    completed_reason=session.completed_reason or "",
                    parent_version=session.parent_version,
                    path=str(path),
                )
            )
        return sorted(infos, key=lambda info: (info.major, info.minor))

    def render_index(self, infos: list[VersionInfo]) -> str:
        lines = [
            "# Research Runs Index",
            "",
            f"- Generated At: {utc_timestamp()}",
            f"- Versions: {len(infos)}",
            "",
        ]
        by_major: dict[int, list[VersionInfo]] = {}
        for info in infos:
            by_major.setdefault(info.major, []).append(info)
        for major in sorted(by_major):
            group = sorted(by_major[major], key=lambda item: item.minor)
            head = group[0]
            title = _display_title(head)[:40]
            lines.extend([f"## V{major:03d} {title}", ""])
            for info in group:
                indent = "  - " if info.minor > 0 else "- "
                parent = f" ← {info.parent_version}" if info.parent_version else ""
                reason = f" {info.completed_reason}" if info.completed_reason else ""
                lines.append(
                    f"{indent}{info.version_label} ({info.session_id}) {info.mode} "
                    f"round={info.round_id} met={info.met}/{info.total}{reason}{parent}"
                )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def write_index(self) -> Path:
        self.archive_path.mkdir(parents=True, exist_ok=True)
        path = self.archive_path / "INDEX.md"
        path.write_text(self.render_index(self.scan()), encoding="utf-8")
        return path


def _parse_version_from_name(name: str) -> tuple[str, int, int] | None:
    match = re.match(r"^V(\d{3})\.(\d+)", name, re.IGNORECASE)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    return f"V{major:03d}.{minor}", major, minor


def _parse_session_id(name: str) -> str:
    parts = name.split("_", 2)
    if len(parts) >= 2:
        return parts[1]
    return ""


def _display_title(info: VersionInfo) -> str:
    if info.project_name and info.project_name != "Research Harness MVP":
        return info.project_name
    return info.goal or info.project_name or "untitled-research"
