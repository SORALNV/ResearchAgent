from __future__ import annotations

import os
from typing import Any, Mapping

from harness.config import HarnessConfig
from harness.iteration_memo import (
    IterationMemoEngine,
    ProviderIterationMemoPlanner,
    RuleBasedIterationMemoPlanner,
)
from harness.kaggle_methodbook import MethodCardStore
from harness.learning_feedback import LearningResultFeedbackAdapter
from harness.methodbook_natural import attach_methodbook_context
from harness.methodbook_planner import MethodBookAwareMemoPlanner
from harness.natural_channel_service_v2 import NaturalChannelService


def attach_iteration_learning(
    service: NaturalChannelService,
    config: HarnessConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> NaturalChannelService:
    """Attach one shared MethodBook to Compute feedback and Kaggle chat prompts."""

    source = dict(os.environ if environ is None else environ)
    if not _bool_value(source.get("METHODBOOK_ENABLED"), True):
        return service

    existing = getattr(service, "iteration_learning_adapter", None)
    if isinstance(existing, LearningResultFeedbackAdapter):
        attach_methodbook_context(service, existing.method_store)
        service.iteration_memo_engine = existing.memo_engine
        return service

    compute = getattr(service, "compute", None)
    feedback = getattr(compute, "feedback", None)
    scheduler = getattr(compute, "scheduler", None)
    if feedback is None or scheduler is None:
        return service
    if isinstance(feedback, LearningResultFeedbackAdapter):
        service.iteration_learning_adapter = feedback
        service.iteration_memo_engine = feedback.memo_engine
        attach_methodbook_context(service, feedback.method_store)
        return service

    method_store = MethodCardStore.from_environment(config.project_root, source)
    provider_planner = None
    if _bool_value(source.get("ITERATION_MEMO_PROVIDER_ENABLED"), True) and _provider_configured(
        config, source
    ):
        provider_planner = ProviderIterationMemoPlanner(config)
    planner = (
        MethodBookAwareMemoPlanner(provider_planner, method_store)
        if provider_planner is not None
        else None
    )
    fallback_planner = MethodBookAwareMemoPlanner(
        RuleBasedIterationMemoPlanner(),
        method_store,
    )
    memo_engine = IterationMemoEngine(
        service.router.store,
        method_store.root,
        method_store,
        planner=planner,
        fallback_planner=fallback_planner,
    )
    learning = LearningResultFeedbackAdapter(feedback, memo_engine)

    # ComputeStack is intentionally frozen, but scheduler.feedback is the active
    # execution dependency. Keep the public stack view consistent without
    # replacing the stack object shared by the existing service wrappers.
    scheduler.feedback = learning
    object.__setattr__(compute, "feedback", learning)
    service.iteration_learning_adapter = learning
    service.iteration_memo_engine = memo_engine
    attach_methodbook_context(service, method_store)
    method_store.render_markdown()
    return service


def _provider_configured(config: HarnessConfig, source: Mapping[str, str]) -> bool:
    commands = (
        getattr(config, "main_agent_command", ""),
        getattr(config, "sub_agent_command", ""),
        getattr(config, "review_agent_command", ""),
        getattr(config, "fresh_agent_command", ""),
    )
    if any(str(item or "").strip() for item in commands):
        return True
    return any(
        str(source.get(name) or "").strip()
        for name in (
            "OPENAI_API_KEY",
            "CODEX_APP_SERVER_COMMAND",
            "AGENT_RUNTIME_ORDER",
            "REVIEW_AGENT_RUNTIME_ORDER",
        )
    )


def _bool_value(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
