from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from harness.control_plane import ConflictError, Domain
from harness.discord_channel_map import ChannelDomainMap, ChannelResolution, DiscordLocation
from harness.state import utc_timestamp


class ChannelSessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


def _normalize_domain(value: Domain | str) -> Domain:
    if isinstance(value, Domain):
        return value
    normalized = str(value).strip().lower()
    aliases = {"research": Domain.RESEARCH, "kaggle": Domain.KAGGLE}
    if normalized not in aliases:
        raise ValueError("mode must be research or kaggle")
    return aliases[normalized]


def _snowflake(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text or not text.isdigit():
        raise ValueError(f"{name} must be a Discord snowflake")
    return text


@dataclass(frozen=True)
class ChannelSessionConfig:
    """Persistent definition of one Discord channel as one work context."""

    conversation_id: str
    guild_id: str
    channel_id: str
    parent_channel_id: str | None
    domain: Domain
    subject: str
    target_ref: str = ""
    project_id: str = ""
    work_session_id: str = ""
    codex_thread_id: str | None = None
    status: ChannelSessionStatus = ChannelSessionStatus.ACTIVE
    created_by: str = "system"
    created_at: str = ""
    updated_at: str = ""
    archived_at: str | None = None

    @property
    def mode(self) -> str:
        return self.domain.value

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["domain"] = self.domain.value
        payload["mode"] = self.domain.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelSessionConfig":
        domain = _normalize_domain(data.get("domain") or data.get("mode") or "")
        conversation_id = _snowflake(
            data.get("conversation_id") or data.get("channel_id"),
            "conversation_id",
        )
        channel_id = _snowflake(
            data.get("channel_id") or conversation_id,
            "channel_id",
        )
        guild_id = _snowflake(data.get("guild_id") or "0", "guild_id")
        parent = data.get("parent_channel_id")
        parent_id = _snowflake(parent, "parent_channel_id") if parent else None
        subject = " ".join(str(data.get("subject") or data.get("topic") or "").split())
        if not subject:
            subject = f"Discord {domain.value} channel {conversation_id}"
        now = utc_timestamp()
        return cls(
            conversation_id=conversation_id,
            guild_id=guild_id,
            channel_id=channel_id,
            parent_channel_id=parent_id,
            domain=domain,
            subject=subject,
            target_ref=" ".join(
                str(
                    data.get("target_ref")
                    or data.get("target")
                    or data.get("competition")
                    or ""
                ).split()
            ),
            project_id=str(data.get("project_id") or "").strip(),
            work_session_id=str(data.get("work_session_id") or "").strip(),
            codex_thread_id=(
                str(data["codex_thread_id"]).strip()
                if data.get("codex_thread_id")
                else None
            ),
            status=ChannelSessionStatus(
                str(data.get("status") or ChannelSessionStatus.ACTIVE.value)
            ),
            created_by=str(data.get("created_by") or "system"),
            created_at=str(data.get("created_at") or now),
            updated_at=str(data.get("updated_at") or now),
            archived_at=(
                str(data["archived_at"]) if data.get("archived_at") else None
            ),
        )


class ChannelSessionRegistry:
    """Atomic JSON registry for channel -> domain/topic/chat bindings."""

    SCHEMA_VERSION = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / "channel_sessions.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @classmethod
    def from_environment(
        cls,
        root: str | Path,
        environ: Mapping[str, str] | None = None,
    ) -> "ChannelSessionRegistry":
        source = dict(os.environ if environ is None else environ)
        registry = cls(root)
        registry.bootstrap(source)
        return registry

    def get(self, location: DiscordLocation) -> ChannelSessionConfig | None:
        return self.get_by_conversation_id(location.conversation_id)

    def get_by_conversation_id(
        self,
        conversation_id: str,
    ) -> ChannelSessionConfig | None:
        key = _snowflake(conversation_id, "conversation_id")
        with self._lock:
            return self._read_all().get(key)

    def active(self, conversation_id: str) -> ChannelSessionConfig | None:
        value = self.get_by_conversation_id(conversation_id)
        return value if value and value.status == ChannelSessionStatus.ACTIVE else None

    def list(self) -> tuple[ChannelSessionConfig, ...]:
        with self._lock:
            values = list(self._read_all().values())
        return tuple(sorted(values, key=lambda item: (item.created_at, item.conversation_id)))

    def setup(
        self,
        location: DiscordLocation,
        *,
        domain: Domain | str,
        subject: str,
        target_ref: str = "",
        actor_id: str = "system",
    ) -> ChannelSessionConfig:
        normalized_domain = _normalize_domain(domain)
        normalized_subject = " ".join(str(subject).split()).strip()
        if not normalized_subject:
            raise ValueError("subject must be non-empty")
        normalized_target = " ".join(str(target_ref).split()).strip()
        now = utc_timestamp()
        conversation_id = _snowflake(location.conversation_id, "conversation_id")
        with self._lock:
            values = self._read_all()
            existing = values.get(conversation_id)
            if existing is not None:
                same = (
                    existing.status == ChannelSessionStatus.ACTIVE
                    and existing.domain == normalized_domain
                    and existing.subject == normalized_subject
                    and existing.target_ref == normalized_target
                )
                if same:
                    return existing
                if existing.status == ChannelSessionStatus.ARCHIVED:
                    raise ConflictError(
                        "this Discord channel is archived; create a new channel for a new topic"
                    )
                raise ConflictError(
                    "this Discord channel is already assigned to another active topic"
                )
            config = ChannelSessionConfig(
                conversation_id=conversation_id,
                guild_id=_snowflake(location.guild_id, "guild_id"),
                channel_id=_snowflake(location.channel_id, "channel_id"),
                parent_channel_id=(
                    _snowflake(location.parent_channel_id, "parent_channel_id")
                    if location.parent_channel_id
                    else None
                ),
                domain=normalized_domain,
                subject=normalized_subject,
                target_ref=normalized_target,
                project_id=_project_id(location, normalized_domain),
                created_by=str(actor_id).strip() or "system",
                created_at=now,
                updated_at=now,
            )
            values[conversation_id] = config
            self._write_all(values)
            return config

    def bind_runtime(
        self,
        conversation_id: str,
        *,
        project_id: str,
        work_session_id: str,
        codex_thread_id: str | None = None,
    ) -> ChannelSessionConfig:
        key = _snowflake(conversation_id, "conversation_id")
        with self._lock:
            values = self._read_all()
            current = values.get(key)
            if current is None:
                raise KeyError(f"unknown channel session: {key}")
            if current.status != ChannelSessionStatus.ACTIVE:
                raise ConflictError("archived channel session cannot be rebound")
            updated = replace(
                current,
                project_id=str(project_id),
                work_session_id=str(work_session_id),
                codex_thread_id=(
                    str(codex_thread_id) if codex_thread_id else current.codex_thread_id
                ),
                updated_at=utc_timestamp(),
            )
            values[key] = updated
            self._write_all(values)
            return updated

    def archive(self, conversation_id: str) -> ChannelSessionConfig:
        key = _snowflake(conversation_id, "conversation_id")
        with self._lock:
            values = self._read_all()
            current = values.get(key)
            if current is None:
                raise KeyError(f"unknown channel session: {key}")
            if current.status == ChannelSessionStatus.ARCHIVED:
                return current
            now = utc_timestamp()
            updated = replace(
                current,
                status=ChannelSessionStatus.ARCHIVED,
                archived_at=now,
                updated_at=now,
            )
            values[key] = updated
            self._write_all(values)
            return updated

    def bootstrap(self, environ: Mapping[str, str]) -> None:
        raw = str(environ.get("DISCORD_CHANNEL_SESSIONS_JSON") or "").strip()
        records: list[Mapping[str, Any]] = []
        if raw:
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("DISCORD_CHANNEL_SESSIONS_JSON must be valid JSON") from exc
            if isinstance(decoded, list):
                records.extend(item for item in decoded if isinstance(item, Mapping))
            elif isinstance(decoded, Mapping):
                for channel_id, item in decoded.items():
                    if isinstance(item, Mapping):
                        records.append({"channel_id": str(channel_id), **dict(item)})
                    else:
                        records.append({"channel_id": str(channel_id), "mode": str(item)})
            else:
                raise ValueError(
                    "DISCORD_CHANNEL_SESSIONS_JSON must be an object or array"
                )

        legacy = ChannelDomainMap.from_environment(environ).to_dict()
        known = {
            str(item.get("conversation_id") or item.get("channel_id") or "")
            for item in records
        }
        for channel_id, domain in legacy.items():
            if channel_id not in known:
                records.append(
                    {
                        "channel_id": channel_id,
                        "conversation_id": channel_id,
                        "guild_id": str(environ.get("DISCORD_GUILD_ID") or "0"),
                        "mode": domain,
                        "subject": f"Discord {domain} channel {channel_id}",
                    }
                )

        if not records:
            return
        with self._lock:
            values = self._read_all()
            changed = False
            for raw_record in records:
                record = ChannelSessionConfig.from_dict(raw_record)
                if record.conversation_id in values:
                    continue
                values[record.conversation_id] = record
                changed = True
            if changed:
                self._write_all(values)

    def _read_all(self) -> dict[str, ChannelSessionConfig]:
        if not self.path.is_file():
            return {}
        try:
            decoded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        raw_channels = decoded.get("channels") if isinstance(decoded, Mapping) else None
        if not isinstance(raw_channels, Mapping):
            return {}
        result: dict[str, ChannelSessionConfig] = {}
        for key, value in raw_channels.items():
            if not isinstance(value, Mapping):
                continue
            try:
                result[str(key)] = ChannelSessionConfig.from_dict(value)
            except (TypeError, ValueError):
                continue
        return result

    def _write_all(self, values: Mapping[str, ChannelSessionConfig]) -> None:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "channels": {
                key: value.to_dict()
                for key, value in sorted(values.items(), key=lambda item: item[0])
            },
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        temporary.replace(self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


class ChannelSessionDomainMap:
    """Resolve dynamic channel sessions before the legacy environment map."""

    def __init__(
        self,
        base: ChannelDomainMap,
        registry: ChannelSessionRegistry,
    ) -> None:
        self.base = base
        self.registry = registry

    def resolve(
        self,
        channel_id: str,
        *,
        parent_channel_id: str | None = None,
    ) -> ChannelResolution:
        channel = _snowflake(channel_id, "channel_id")
        direct = self.registry.active(channel)
        if direct is not None:
            return ChannelResolution(direct.domain, channel, False)
        if parent_channel_id is not None:
            parent = _snowflake(parent_channel_id, "parent_channel_id")
            inherited = self.registry.active(parent)
            if inherited is not None:
                return ChannelResolution(inherited.domain, parent, True)
        return self.base.resolve(channel, parent_channel_id=parent_channel_id)

    def to_dict(self) -> dict[str, str]:
        result = dict(self.base.to_dict())
        for item in self.registry.list():
            if item.status == ChannelSessionStatus.ACTIVE:
                result[item.conversation_id] = item.domain.value
        return dict(sorted(result.items()))

    def __bool__(self) -> bool:
        return bool(self.to_dict())


def _project_id(location: DiscordLocation, domain: Domain) -> str:
    digest = hashlib.sha256(
        f"channel-session:{location.guild_id}:{location.conversation_id}:{domain.value}".encode(
            "utf-8"
        )
    ).hexdigest()[:20]
    return f"PRJ-CHANNEL-{domain.value.upper()}-{digest}"
