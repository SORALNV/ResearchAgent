from __future__ import annotations

from pathlib import Path

from harness.planning import render_planning_scout
from harness.state import ResearchSession


def render_research_brief(session: ResearchSession) -> str:
    questions = "\n".join(f"- {question}" for question in session.planning_questions)
    answers = "\n".join(
        f"- {key}: {value}" for key, value in session.planning_answers.items()
    ) or "- 未確定"
    ideas = "\n".join(f"- {idea}" for idea in session.accepted_ideas) or "- なし"
    redirects = "\n".join(f"- {item}" for item in session.redirects) or "- なし"
    approvals = "\n".join(
        f"- {item.approval_id}: {item.operation} ({item.status})"
        for item in session.approval_requests.values()
    ) or "- なし"
    phase_gates = "\n".join(
        f"- {gate.gate_id}: {gate.phase} ({gate.status}) {gate.reason}"
        for gate in session.phase_gates.values()
    ) or "- なし"
    scout = render_planning_scout(session.planning_scout) if session.planning_scout else "## Similar Research Scout\n\n- 未実行: plan中の会話で必要になれば自動検索します。"
    return f"""# Research Brief

## Project
{session.project_name}

## Session
- ID: {session.session_id}
- Version: {session.version_label or "未設定"}
- Research Dir: {session.research_dir or "未設定"}
- Mode: {session.mode.value}
- Phase: {session.phase}
- Round: {session.round_id}

## Goal
{session.research_goal}

## Planning Questions
{questions}

## Planning Answers
{answers}

{scout}

## Current Question
{session.current_question or "未設定"}

## Constraints And Redirects
{redirects}

## Accepted Ideas
{ideas}

## Approval Requests
{approvals}

## Phase Gates
{phase_gates}

## Next Action
{session.next_action or "未設定"}
"""


class ResearchBriefWriter:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, session: ResearchSession) -> str:
        body = render_research_brief(session)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(body, encoding="utf-8")
        return body
