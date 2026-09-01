from __future__ import annotations

import json
from typing import Any, Mapping

from harness.channel_sessions import ChannelSessionConfig
from harness.control_plane import Domain
from harness.discord_thread_router import DiscordIngressResult
from harness.kaggle_methodbook import MethodCardStore
from harness.natural_channel_service_v2 import (
    NaturalChannelService,
    NaturalConversationHandler,
)


class MethodBookConversationHandler(NaturalConversationHandler):
    """Inject only relevant, non-terminal MethodCards into Kaggle conversations."""

    def __init__(self, *args: Any, method_store: MethodCardStore, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.method_store = method_store

    def _build_prompt(
        self,
        ingress: DiscordIngressResult,
        channel: ChannelSessionConfig | None,
    ) -> str:
        base = super()._build_prompt(ingress, channel)
        if self.domain != Domain.KAGGLE:
            return base
        user_text = str(ingress.event.payload.get("text") or "")
        subject = channel.subject if channel else ingress.route.work_session.title
        target = channel.target_ref if channel else ""
        query = " ".join(item for item in (subject, target, user_text) if item)
        task_family, modality, metric_family = _recent_scope(
            self.store,
            ingress.route.work_session.work_session_id,
        )
        cards = self.method_store.search(
            query,
            task_family=task_family,
            modality=modality,
            metric_family=metric_family,
            limit=8,
        )
        payload = [
            {
                "method_id": card.method_id,
                "claim": card.claim,
                "scope": card.scope.to_dict(),
                "status": card.status.value,
                "confidence": card.confidence.value,
                "support_count": len(card.evidence),
                "counterevidence_count": len(card.counterevidence),
                "source_competitions": sorted(
                    {
                        item.competition
                        for item in card.evidence
                        if item.competition and item.competition != "unknown"
                    }
                ),
                "next_falsification": card.next_falsification,
            }
            for card in cards
        ]
        return (
            base
            + "\n\n以下は過去実験から抽出したMethodBook候補です。"
            "検証済みでも現在のコンペでの成功を保証しません。scopeを外して一般化せず、"
            "採用する場合はjob proposalのmetadata.method_card_idsへmethod_idを記録し、"
            "最安の反証実験を先に設計してください。\n"
            "<UNTRUSTED_METHODBOOK>\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n</UNTRUSTED_METHODBOOK>\n"
        )


def attach_methodbook_context(
    service: NaturalChannelService,
    method_store: MethodCardStore | None,
) -> NaturalChannelService:
    if method_store is None:
        return service
    current = service.dispatcher.handlers.get(Domain.KAGGLE)
    if isinstance(current, MethodBookConversationHandler):
        service.method_store = method_store
        return service
    if not isinstance(current, NaturalConversationHandler):
        raise TypeError("Kaggle dispatcher is not a NaturalConversationHandler")
    enhanced = MethodBookConversationHandler(
        current.config,
        current.registry,
        current.domain,
        current.store,
        executor=current.executor,
        workspace_root=current.workspace_root,
        method_store=method_store,
    )
    service.dispatcher.handlers[Domain.KAGGLE] = enhanced
    service.method_store = method_store
    return service


def _recent_scope(store: Any, work_session_id: str) -> tuple[str, str, str]:
    try:
        jobs = store.list_jobs(work_session_id=work_session_id)
    except Exception:
        return "", "", ""
    for job in reversed(jobs):
        payload = getattr(getattr(job, "spec", None), "payload", {}) or {}
        metadata = payload.get("proposal_metadata")
        metadata_map = metadata if isinstance(metadata, Mapping) else {}
        task_family = _first_text(
            payload.get("task_family"),
            metadata_map.get("task_family"),
        )
        modality = _first_text(
            payload.get("modality"),
            metadata_map.get("modality"),
        )
        metric_family = _first_text(
            payload.get("metric_family"),
            metadata_map.get("metric_family"),
        )
        if task_family or modality or metric_family:
            return task_family, modality, metric_family
    return "", "", ""


def _first_text(*values: Any) -> str:
    for value in values:
        text = " ".join(str(value or "").split())
        if text:
            return text
    return ""
