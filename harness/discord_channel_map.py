from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Mapping, MutableMapping

from harness.control_plane import Domain


_CHANNEL_ID_PATTERN = re.compile(r"^[0-9]{1,32}$")


class ChannelRoutingError(RuntimeError):
    """Base class for Discord channel/domain routing failures."""


class UnmappedDiscordChannelError(ChannelRoutingError):
    """Raised when no explicit domain mapping covers a Discord location."""


@dataclass(frozen=True)
class ChannelResolution:
    domain: Domain
    route_channel_id: str
    inherited_from_parent: bool


@dataclass(frozen=True)
class DiscordLocation:
    guild_id: str
    channel_id: str
    parent_channel_id: str | None = None
    thread_id: str | None = None
    forum_post_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "guild_id", snowflake(self.guild_id, "guild_id"))
        object.__setattr__(
            self,
            "channel_id",
            snowflake(self.channel_id, "channel_id"),
        )
        for field_name in ("parent_channel_id", "thread_id", "forum_post_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, snowflake(value, field_name))
        if self.thread_id is not None and self.forum_post_id is not None:
            raise ValueError("set either thread_id or forum_post_id, not both")

    @property
    def conversation_id(self) -> str:
        return self.thread_id or self.forum_post_id or self.channel_id

    def identity_ref(self) -> dict[str, str]:
        if self.thread_id:
            return {"thread_id": self.thread_id}
        if self.forum_post_id:
            return {"forum_post_id": self.forum_post_id}
        return {"conversation_id": self.channel_id}

    def external_ref(self) -> dict[str, str]:
        value = {
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "parent_channel_id": self.parent_channel_id or "",
            **self.identity_ref(),
        }
        return {key: item for key, item in value.items() if item}


class ChannelDomainMap:
    """Strict Discord channel-to-domain map.

    Exact channel IDs win. Otherwise a thread inherits its mapped parent.
    Unknown channels fail closed.
    """

    def __init__(self, routes: Mapping[str, Domain | str] | None = None) -> None:
        normalized: dict[str, Domain] = {}
        for channel_id, domain in dict(routes or {}).items():
            self._add(normalized, channel_id, domain)
        self._routes = normalized

    @classmethod
    def parse(cls, raw: str) -> "ChannelDomainMap":
        raw = raw.strip()
        if not raw:
            return cls()
        if raw.startswith("{"):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "DISCORD_CHANNEL_DOMAIN_MAP must be a JSON object or "
                    "channel=domain comma list"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError("DISCORD_CHANNEL_DOMAIN_MAP JSON must be an object")
            return cls({str(key): str(item) for key, item in value.items()})

        result = cls()
        routes: dict[str, Domain] = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            separator = "=" if "=" in entry else ":" if ":" in entry else None
            if separator is None:
                raise ValueError(
                    f"invalid Discord channel mapping {entry!r}; use channel=domain"
                )
            channel_id, domain = (part.strip() for part in entry.split(separator, 1))
            cls._add(routes, channel_id, domain)
        result._routes = routes
        return result

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "ChannelDomainMap":
        source = dict(os.environ if environ is None else environ)
        parsed = cls.parse(source.get("DISCORD_CHANNEL_DOMAIN_MAP", ""))
        routes = dict(parsed._routes)
        for name, domain in (
            ("DISCORD_RESEARCH_CHANNEL_IDS", Domain.RESEARCH),
            ("DISCORD_KAGGLE_CHANNEL_IDS", Domain.KAGGLE),
        ):
            for channel_id in csv_values(source.get(name, "")):
                cls._add(routes, channel_id, domain)

        legacy = source.get("DISCORD_CHANNEL_ID", "").strip()
        if legacy and not routes:
            cls._add(routes, legacy, Domain.RESEARCH)
        return cls(routes)

    @staticmethod
    def _add(
        routes: MutableMapping[str, Domain],
        channel_id: str,
        domain: Domain | str,
    ) -> None:
        normalized_id = snowflake(channel_id, "channel_id")
        normalized_domain = normalize_domain(domain)
        if normalized_domain not in {Domain.RESEARCH, Domain.KAGGLE}:
            raise ValueError(
                "Discord channels must resolve to research or kaggle, not hybrid"
            )
        current = routes.get(normalized_id)
        if current is not None and current != normalized_domain:
            raise ValueError(
                f"Discord channel {normalized_id} is mapped to both "
                f"{current.value} and {normalized_domain.value}"
            )
        routes[normalized_id] = normalized_domain

    def resolve(
        self,
        channel_id: str,
        *,
        parent_channel_id: str | None = None,
    ) -> ChannelResolution:
        channel = snowflake(channel_id, "channel_id")
        direct = self._routes.get(channel)
        if direct is not None:
            return ChannelResolution(direct, channel, False)
        if parent_channel_id is not None:
            parent = snowflake(parent_channel_id, "parent_channel_id")
            inherited = self._routes.get(parent)
            if inherited is not None:
                return ChannelResolution(inherited, parent, True)
        raise UnmappedDiscordChannelError(
            f"Discord channel {channel} is not mapped to research or kaggle"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            channel_id: domain.value
            for channel_id, domain in sorted(self._routes.items())
        }

    def __bool__(self) -> bool:
        return bool(self._routes)


def normalize_domain(value: Domain | str) -> Domain:
    return value if isinstance(value, Domain) else Domain(str(value).strip().lower())


def snowflake(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not _CHANNEL_ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"{label} must be a numeric Discord snowflake")
    return normalized


def csv_values(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
