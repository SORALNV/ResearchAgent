from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from harness.state import utc_timestamp


SCHEMA_VERSION = 1


class MethodCardStatus(str, Enum):
    LOCAL = "local"
    TASK_CANDIDATE = "task_candidate"
    VERIFIED = "verified"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class MethodConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceKind(str, Enum):
    SUPPORT = "support"
    COUNTER = "counter"


class ValidationKind(str, Enum):
    CV = "cv"
    HOLDOUT = "holdout"
    PRIVATE_LB = "private_lb"
    PUBLIC_LB = "public_lb"
    LEADERBOARD = "leaderboard"
    UNKNOWN = "unknown"


_PROMOTION_VALIDATION_KINDS = {
    ValidationKind.CV,
    ValidationKind.HOLDOUT,
    ValidationKind.PRIVATE_LB,
}


@dataclass(frozen=True)
class MethodScope:
    task_family: str = "unknown"
    modality: str = "unknown"
    metric_family: str = "unknown"
    conditions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_family": self.task_family,
            "modality": self.modality,
            "metric_family": self.metric_family,
            "conditions": list(self.conditions),
            "tags": list(self.tags),
        }

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | None) -> "MethodScope":
        data = dict(value or {})
        return cls(
            task_family=_clean_text(data.get("task_family") or data.get("task") or "unknown"),
            modality=_clean_text(data.get("modality") or "unknown"),
            metric_family=_clean_text(data.get("metric_family") or data.get("metric") or "unknown"),
            conditions=_text_tuple(data.get("conditions")),
            tags=_text_tuple(data.get("tags")),
        )

    def canonical_key(self) -> dict[str, Any]:
        return {
            "task_family": self.task_family.lower(),
            "modality": self.modality.lower(),
            "metric_family": self.metric_family.lower(),
            "conditions": sorted(item.lower() for item in self.conditions),
            "tags": sorted(item.lower() for item in self.tags),
        }


@dataclass(frozen=True)
class MethodEvidence:
    result_ref: str
    competition: str
    memo_id: str
    outcome: str
    validation_kind: ValidationKind = ValidationKind.UNKNOWN
    metric_name: str = ""
    metric_value: float | None = None
    metric_delta: float | None = None
    independent_key: str = ""
    observed_at: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        if not str(self.result_ref).strip():
            raise ValueError("method evidence requires result_ref")
        object.__setattr__(self, "result_ref", str(self.result_ref).strip())
        object.__setattr__(self, "competition", _clean_text(self.competition or "unknown"))
        object.__setattr__(self, "memo_id", str(self.memo_id).strip())
        object.__setattr__(self, "outcome", _clean_text(self.outcome or "unknown"))
        object.__setattr__(self, "validation_kind", ValidationKind(self.validation_kind))
        object.__setattr__(self, "metric_name", _clean_text(self.metric_name))
        object.__setattr__(
            self,
            "independent_key",
            str(self.independent_key or self.result_ref).strip(),
        )

    @property
    def identity(self) -> str:
        return f"{self.result_ref}\0{self.independent_key}\0{self.validation_kind.value}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["validation_kind"] = self.validation_kind.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MethodEvidence":
        return cls(
            result_ref=str(data["result_ref"]),
            competition=str(data.get("competition") or "unknown"),
            memo_id=str(data.get("memo_id") or ""),
            outcome=str(data.get("outcome") or "unknown"),
            validation_kind=ValidationKind(
                str(data.get("validation_kind") or ValidationKind.UNKNOWN.value)
            ),
            metric_name=str(data.get("metric_name") or ""),
            metric_value=_optional_float(data.get("metric_value")),
            metric_delta=_optional_float(data.get("metric_delta")),
            independent_key=str(data.get("independent_key") or ""),
            observed_at=str(data.get("observed_at") or utc_timestamp()),
        )


@dataclass(frozen=True)
class MethodCandidate:
    claim: str
    scope: MethodScope = field(default_factory=MethodScope)
    evidence_kind: EvidenceKind = EvidenceKind.SUPPORT
    next_falsification: str = ""
    rationale: str = ""
    method_id: str = ""

    def __post_init__(self) -> None:
        claim = _clean_text(self.claim)
        if not claim:
            raise ValueError("method candidate requires a non-empty claim")
        object.__setattr__(self, "claim", claim)
        object.__setattr__(self, "evidence_kind", EvidenceKind(self.evidence_kind))
        object.__setattr__(self, "next_falsification", _clean_text(self.next_falsification))
        object.__setattr__(self, "rationale", _clean_text(self.rationale))
        object.__setattr__(self, "method_id", str(self.method_id).strip())

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "MethodCandidate":
        return cls(
            claim=str(value.get("claim") or value.get("lesson") or ""),
            scope=MethodScope.from_value(
                value.get("scope") if isinstance(value.get("scope"), Mapping) else value
            ),
            evidence_kind=EvidenceKind(
                str(value.get("evidence_kind") or EvidenceKind.SUPPORT.value)
            ),
            next_falsification=str(value.get("next_falsification") or ""),
            rationale=str(value.get("rationale") or ""),
            method_id=str(value.get("method_id") or ""),
        )


@dataclass(frozen=True)
class MethodCard:
    method_id: str
    revision: int
    claim: str
    scope: MethodScope
    status: MethodCardStatus
    confidence: MethodConfidence
    evidence: tuple[MethodEvidence, ...] = ()
    counterevidence: tuple[MethodEvidence, ...] = ()
    source_memo_ids: tuple[str, ...] = ()
    next_falsification: str = ""
    rationale: str = ""
    transition_reason: str = ""
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "method_id": self.method_id,
            "revision": self.revision,
            "claim": self.claim,
            "scope": self.scope.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "counterevidence": [item.to_dict() for item in self.counterevidence],
            "source_memo_ids": list(self.source_memo_ids),
            "next_falsification": self.next_falsification,
            "rationale": self.rationale,
            "transition_reason": self.transition_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MethodCard":
        return cls(
            method_id=str(data["method_id"]),
            revision=max(1, int(data.get("revision") or 1)),
            claim=str(data["claim"]),
            scope=MethodScope.from_value(
                data.get("scope") if isinstance(data.get("scope"), Mapping) else None
            ),
            status=MethodCardStatus(str(data.get("status") or MethodCardStatus.LOCAL.value)),
            confidence=MethodConfidence(
                str(data.get("confidence") or MethodConfidence.LOW.value)
            ),
            evidence=tuple(
                MethodEvidence.from_dict(item)
                for item in data.get("evidence", [])
                if isinstance(item, Mapping)
            ),
            counterevidence=tuple(
                MethodEvidence.from_dict(item)
                for item in data.get("counterevidence", [])
                if isinstance(item, Mapping)
            ),
            source_memo_ids=_text_tuple(data.get("source_memo_ids")),
            next_falsification=str(data.get("next_falsification") or ""),
            rationale=str(data.get("rationale") or ""),
            transition_reason=str(data.get("transition_reason") or ""),
            created_at=str(data.get("created_at") or utc_timestamp()),
            updated_at=str(data.get("updated_at") or utc_timestamp()),
        )


class MethodCardStore:
    """Append-only, cross-competition methodology store with a generated Markdown view."""

    _locks_guard = threading.RLock()
    _locks: dict[str, threading.RLock] = {}

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / "method_cards.jsonl"
        self.markdown_path = self.root / "KAGGLE_METHODBOOK.md"
        self.root.mkdir(parents=True, exist_ok=True)
        key = str(self.path)
        with self._locks_guard:
            self._lock = self._locks.setdefault(key, threading.RLock())

    @classmethod
    def from_environment(
        cls,
        project_root: str | Path,
        environ: Mapping[str, str] | None = None,
    ) -> "MethodCardStore":
        source = dict(os.environ if environ is None else environ)
        root = Path(source.get("METHODBOOK_DIR") or "knowledge").expanduser()
        if not root.is_absolute():
            root = Path(project_root).expanduser().resolve() / root
        return cls(root)

    def get(self, method_id: str) -> MethodCard | None:
        return self._read_latest().get(str(method_id).strip())

    def list(
        self,
        *,
        statuses: Iterable[MethodCardStatus | str] | None = None,
    ) -> tuple[MethodCard, ...]:
        allowed = (
            {MethodCardStatus(item) for item in statuses}
            if statuses is not None
            else None
        )
        cards = list(self._read_latest().values())
        if allowed is not None:
            cards = [item for item in cards if item.status in allowed]
        return tuple(sorted(cards, key=lambda item: (item.status.value, item.method_id)))

    def search(
        self,
        query: str,
        *,
        task_family: str = "",
        modality: str = "",
        metric_family: str = "",
        limit: int = 8,
        include_inactive: bool = False,
    ) -> tuple[MethodCard, ...]:
        query_tokens = _tokens(query)
        task = _clean_text(task_family).lower()
        modal = _clean_text(modality).lower()
        metric = _clean_text(metric_family).lower()
        scored: list[tuple[int, MethodCard]] = []
        for card in self.list():
            if not include_inactive and card.status in {
                MethodCardStatus.DEPRECATED,
                MethodCardStatus.REJECTED,
            }:
                continue
            haystack = " ".join(
                [
                    card.claim,
                    card.scope.task_family,
                    card.scope.modality,
                    card.scope.metric_family,
                    *card.scope.conditions,
                    *card.scope.tags,
                ]
            ).lower()
            lexical = sum(5 for token in query_tokens if token and token in haystack)
            scope_score = 0
            if task and task == card.scope.task_family.lower():
                scope_score += 25
            if modal and modal == card.scope.modality.lower():
                scope_score += 15
            if metric and metric == card.scope.metric_family.lower():
                scope_score += 15
            global_card = card.scope.task_family.lower() in {"", "any", "general", "unknown"}
            if query_tokens and lexical == 0 and scope_score == 0 and not global_card:
                continue
            status_score = {
                MethodCardStatus.VERIFIED: 40,
                MethodCardStatus.TASK_CANDIDATE: 25,
                MethodCardStatus.LOCAL: 10,
                MethodCardStatus.DEPRECATED: -20,
                MethodCardStatus.REJECTED: -40,
            }[card.status]
            evidence_score = min(15, len(card.evidence) * 3) - min(
                15, len(card.counterevidence) * 4
            )
            scored.append((status_score + lexical + scope_score + evidence_score, card))
        scored.sort(key=lambda pair: (-pair[0], pair[1].method_id))
        return tuple(item for _, item in scored[: max(1, int(limit))])

    def record(
        self,
        candidate: MethodCandidate | Mapping[str, Any],
        evidence: MethodEvidence,
    ) -> MethodCard:
        value = (
            candidate
            if isinstance(candidate, MethodCandidate)
            else MethodCandidate.from_value(candidate)
        )
        method_id = value.method_id or method_card_id(value.claim, value.scope)
        with self._lock:
            existing = self._read_latest_unlocked().get(method_id)
            if existing is not None and (
                existing.claim != value.claim
                or existing.scope.canonical_key() != value.scope.canonical_key()
            ):
                raise ValueError(
                    f"method_id {method_id} already refers to a different claim or scope"
                )
            supports = list(existing.evidence if existing else ())
            counters = list(existing.counterevidence if existing else ())
            target = supports if value.evidence_kind == EvidenceKind.SUPPORT else counters
            if not any(item.identity == evidence.identity for item in target):
                target.append(evidence)
            memo_ids = list(existing.source_memo_ids if existing else ())
            if evidence.memo_id and evidence.memo_id not in memo_ids:
                memo_ids.append(evidence.memo_id)

            current_status = existing.status if existing else MethodCardStatus.LOCAL
            status, confidence = _infer_status(
                current_status,
                tuple(supports),
                tuple(counters),
            )
            created_at = existing.created_at if existing else utc_timestamp()
            revision = (existing.revision + 1) if existing else 1
            card = MethodCard(
                method_id=method_id,
                revision=revision,
                claim=value.claim,
                scope=value.scope,
                status=status,
                confidence=confidence,
                evidence=tuple(supports),
                counterevidence=tuple(counters),
                source_memo_ids=tuple(memo_ids),
                next_falsification=(
                    value.next_falsification
                    or (existing.next_falsification if existing else "")
                ),
                rationale=value.rationale or (existing.rationale if existing else ""),
                transition_reason="automatic evidence update",
                created_at=created_at,
                updated_at=utc_timestamp(),
            )
            if existing is not None and _card_semantics(existing) == _card_semantics(card):
                return existing
            self._append_revision_unlocked(card)
            self._render_markdown_unlocked()
            return card

    def transition(
        self,
        method_id: str,
        target: MethodCardStatus | str,
        *,
        reason: str,
    ) -> MethodCard:
        target_status = MethodCardStatus(target)
        explanation = _clean_text(reason)
        if not explanation:
            raise ValueError("method-card transition requires a reason")
        allowed = {
            MethodCardStatus.LOCAL: {
                MethodCardStatus.TASK_CANDIDATE,
                MethodCardStatus.VERIFIED,
                MethodCardStatus.DEPRECATED,
                MethodCardStatus.REJECTED,
            },
            MethodCardStatus.TASK_CANDIDATE: {
                MethodCardStatus.LOCAL,
                MethodCardStatus.VERIFIED,
                MethodCardStatus.DEPRECATED,
                MethodCardStatus.REJECTED,
            },
            MethodCardStatus.VERIFIED: {
                MethodCardStatus.TASK_CANDIDATE,
                MethodCardStatus.DEPRECATED,
                MethodCardStatus.REJECTED,
            },
            MethodCardStatus.DEPRECATED: set(),
            MethodCardStatus.REJECTED: set(),
        }
        with self._lock:
            current = self._read_latest_unlocked().get(str(method_id).strip())
            if current is None:
                raise KeyError(f"unknown method card: {method_id}")
            if target_status == current.status:
                return current
            if target_status not in allowed[current.status]:
                raise ValueError(
                    f"invalid method-card transition: {current.status.value} -> {target_status.value}"
                )
            updated = replace(
                current,
                revision=current.revision + 1,
                status=target_status,
                confidence=_confidence_for_status(target_status),
                transition_reason=explanation,
                updated_at=utc_timestamp(),
            )
            self._append_revision_unlocked(updated)
            self._render_markdown_unlocked()
            return updated

    def render_markdown(self) -> Path:
        with self._lock:
            self._render_markdown_unlocked()
        return self.markdown_path

    def _read_latest(self) -> dict[str, MethodCard]:
        with self._lock:
            return self._read_latest_unlocked()

    def _read_latest_unlocked(self) -> dict[str, MethodCard]:
        if not self.path.is_file():
            return {}
        result: dict[str, MethodCard] = {}
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return {}
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, Mapping):
                continue
            raw_card = record.get("card") if isinstance(record.get("card"), Mapping) else record
            try:
                card = MethodCard.from_dict(raw_card)
            except (KeyError, TypeError, ValueError):
                continue
            current = result.get(card.method_id)
            if current is None or card.revision >= current.revision:
                result[card.method_id] = card
        return result

    def _append_revision_unlocked(self, card: MethodCard) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "method_card_revision",
            "card": card.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

    def _render_markdown_unlocked(self) -> None:
        cards = list(self._read_latest_unlocked().values())
        groups = [
            MethodCardStatus.VERIFIED,
            MethodCardStatus.TASK_CANDIDATE,
            MethodCardStatus.LOCAL,
            MethodCardStatus.DEPRECATED,
            MethodCardStatus.REJECTED,
        ]
        lines = [
            "# Kaggle MethodBook",
            "",
            "`method_cards.jsonl`が正本です。このMarkdownは自動生成ビューです。",
            "Public LBだけの証拠は`verified`昇格に使用しません。",
            "",
        ]
        for status in groups:
            values = sorted(
                (item for item in cards if item.status == status),
                key=lambda item: item.method_id,
            )
            lines.extend([f"## {status.value}", ""])
            if not values:
                lines.extend(["- なし", ""])
                continue
            for card in values:
                scope = (
                    f"{card.scope.task_family} / {card.scope.modality} / "
                    f"{card.scope.metric_family}"
                )
                lines.append(
                    f"- **{card.method_id}** `{card.confidence.value}` — {card.claim}"
                )
                lines.append(
                    f"  - scope: `{scope}`; support {len(card.evidence)}; "
                    f"counter {len(card.counterevidence)}"
                )
                if card.next_falsification:
                    lines.append(f"  - next falsification: {card.next_falsification}")
            lines.append("")
        temporary = self.markdown_path.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        temporary.replace(self.markdown_path)


def method_card_id(claim: str, scope: MethodScope) -> str:
    payload = {
        "claim": _clean_text(claim).lower(),
        "scope": scope.canonical_key(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"KM-{digest.upper()}"


def _infer_status(
    current: MethodCardStatus,
    supports: Sequence[MethodEvidence],
    counters: Sequence[MethodEvidence],
) -> tuple[MethodCardStatus, MethodConfidence]:
    if current in {MethodCardStatus.DEPRECATED, MethodCardStatus.REJECTED}:
        return current, _confidence_for_status(current)
    robust_support = [
        item for item in supports if item.validation_kind in _PROMOTION_VALIDATION_KINDS
    ]
    robust_counter = [
        item for item in counters if item.validation_kind in _PROMOTION_VALIDATION_KINDS
    ]
    independent_support = {item.independent_key for item in robust_support}
    competitions = {
        item.competition.lower()
        for item in robust_support
        if item.competition and item.competition.lower() != "unknown"
    }
    if robust_counter:
        if len(independent_support) >= 2 and len(robust_support) > len(robust_counter):
            return MethodCardStatus.TASK_CANDIDATE, MethodConfidence.MEDIUM
        return MethodCardStatus.LOCAL, MethodConfidence.LOW
    if len(independent_support) >= 2 and len(competitions) >= 2:
        return MethodCardStatus.VERIFIED, MethodConfidence.HIGH
    if len(independent_support) >= 2:
        return MethodCardStatus.TASK_CANDIDATE, MethodConfidence.MEDIUM
    return MethodCardStatus.LOCAL, MethodConfidence.LOW


def _confidence_for_status(status: MethodCardStatus) -> MethodConfidence:
    if status == MethodCardStatus.VERIFIED:
        return MethodConfidence.HIGH
    if status == MethodCardStatus.TASK_CANDIDATE:
        return MethodConfidence.MEDIUM
    return MethodConfidence.LOW


def _card_semantics(card: MethodCard) -> dict[str, Any]:
    value = card.to_dict()
    for key in ("revision", "updated_at", "transition_reason"):
        value.pop(key, None)
    return value


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    raw = [value] if isinstance(value, str) else value if isinstance(value, Sequence) else []
    return tuple(
        dict.fromkeys(text for item in raw if (text := _clean_text(item)))
    )


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tokens(value: str) -> tuple[str, ...]:
    text = str(value or "").lower()
    raw = re.findall(r"[a-z0-9_+.-]+|[\u3040-\u30ff\u3400-\u9fff]+", text)
    return tuple(dict.fromkeys(item for item in raw if len(item) >= 2))
