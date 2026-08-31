from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from harness.control_plane import Domain
from harness.discord_channel_map import normalize_domain


DECISION_EVENT_PREFIX = "human.decision."


class HumanDecisionKind(str, Enum):
    HYPOTHESIS = "hypothesis"
    RESULT_INTERPRETATION = "result_interpretation"
    KAGGLE_SUBMISSION = "kaggle_submission"
    RESEARCH_PAPER = "research_paper"


class HumanDecisionVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"


class ControlledAction(str, Enum):
    START_EXPERIMENT = "start_experiment"
    CONTINUE_FROM_RESULT = "continue_from_result"
    SUBMIT_KAGGLE = "submit_kaggle"
    START_PAPER_DRAFT = "start_paper_draft"


@dataclass(frozen=True)
class HumanGateResult:
    allowed: bool
    action: ControlledAction
    required_decision: HumanDecisionKind
    subject_ref: str
    verdict: HumanDecisionVerdict | None = None
    event_id: str | None = None
    reason: str = ""


class HumanResponsibilityPolicy:
    """Research-direction decisions that cannot be delegated to an Agent."""

    _ACTION_DECISIONS = {
        ControlledAction.START_EXPERIMENT: HumanDecisionKind.HYPOTHESIS,
        ControlledAction.CONTINUE_FROM_RESULT: HumanDecisionKind.RESULT_INTERPRETATION,
        ControlledAction.SUBMIT_KAGGLE: HumanDecisionKind.KAGGLE_SUBMISSION,
        ControlledAction.START_PAPER_DRAFT: HumanDecisionKind.RESEARCH_PAPER,
    }

    @classmethod
    def required_decisions(cls, domain: Domain | str) -> tuple[HumanDecisionKind, ...]:
        normalized = normalize_domain(domain)
        shared = (
            HumanDecisionKind.HYPOTHESIS,
            HumanDecisionKind.RESULT_INTERPRETATION,
        )
        if normalized == Domain.KAGGLE:
            return (*shared, HumanDecisionKind.KAGGLE_SUBMISSION)
        if normalized == Domain.RESEARCH:
            return (*shared, HumanDecisionKind.RESEARCH_PAPER)
        raise ValueError("human responsibility policy requires research or kaggle")

    @classmethod
    def decision_for_action(
        cls,
        domain: Domain | str,
        action: ControlledAction | str,
    ) -> HumanDecisionKind:
        normalized_domain = normalize_domain(domain)
        normalized_action = ControlledAction(action)
        if normalized_domain not in {Domain.RESEARCH, Domain.KAGGLE}:
            raise ValueError("controlled actions require research or kaggle")
        if (
            normalized_action == ControlledAction.SUBMIT_KAGGLE
            and normalized_domain != Domain.KAGGLE
        ):
            raise ValueError("submit_kaggle is only valid in the kaggle domain")
        if (
            normalized_action == ControlledAction.START_PAPER_DRAFT
            and normalized_domain != Domain.RESEARCH
        ):
            raise ValueError("start_paper_draft is only valid in the research domain")
        return cls._ACTION_DECISIONS[normalized_action]

    @classmethod
    def human_tasks(cls, domain: Domain | str) -> tuple[str, ...]:
        normalized = normalize_domain(domain)
        if normalized not in {Domain.RESEARCH, Domain.KAGGLE}:
            raise ValueError("human responsibility policy requires research or kaggle")
        final = (
            "提出してよいかの最終判断"
            if normalized == Domain.KAGGLE
            else "論文としてまとめるかの判断"
        )
        return (
            "AIと相談した上で、何を試すかの仮説を選ぶ",
            "実験結果を解釈し、次の判断を確定する",
            final,
        )

    @classmethod
    def agent_tasks(cls, domain: Domain | str) -> tuple[str, ...]:
        normalized = normalize_domain(domain)
        common = (
            "調査、実装、テスト、エラー修正、実験実行、ログ整理",
            "結果の要約、反証、比較、次の仮説候補の提案",
            "checkpoint、artifact、再開、失敗実験の保存",
        )
        if normalized == Domain.KAGGLE:
            return (
                *common,
                "submission候補の形式・SHA-256検証",
                "人間がexact SHA-256を承認した後のKaggle提出、履歴照合、LB記録",
            )
        if normalized == Domain.RESEARCH:
            return (
                *common,
                "人間が論文化を承認した後の根拠束作成、文献整理、草稿、レビュー、改稿、Markdown/LaTeX成果物生成",
            )
        raise ValueError("agent responsibility policy requires research or kaggle")

    @classmethod
    def metadata(cls, domain: Domain | str) -> dict[str, Any]:
        normalized = normalize_domain(domain)
        return {
            "domain": normalized.value,
            "human_only": [
                item.value for item in cls.required_decisions(normalized)
            ],
            "human_tasks": list(cls.human_tasks(normalized)),
            "agent_tasks": list(cls.agent_tasks(normalized)),
            "scope": "research_direction",
        }


def normalize_subject_ref(kind: HumanDecisionKind, subject_ref: str) -> str:
    normalized = str(subject_ref).strip()
    if not normalized:
        raise ValueError("subject_ref must be non-empty")
    if kind != HumanDecisionKind.KAGGLE_SUBMISSION:
        return normalized
    digest = normalized.removeprefix("sha256:").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(
            "Kaggle submission decisions must bind to an exact SHA-256 digest"
        )
    return f"sha256:{digest}"


def decision_event_type(kind: HumanDecisionKind) -> str:
    return DECISION_EVENT_PREFIX + kind.value
