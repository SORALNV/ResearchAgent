from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from harness.compute_feedback import (
    HypothesisProposal,
    ResultFeedbackEngine,
    find_hypothesis_proposal,
    latest_compute_context,
    normalize_hypothesis_proposal,
)
from harness.compute_scheduler import (
    ComputeScheduler,
    ComputeStack,
    build_default_compute_stack,
)
from harness.config import HarnessConfig
from harness.control_plane import (
    ConflictError,
    Domain,
    EventLane,
    Job,
    JobStatus,
)
from harness.discord_channel_map import DiscordLocation
from harness.discord_thread_router import (
    DiscordChannelDispatcher,
    DiscordIngressResult,
    DiscordThreadRouter,
)
from harness.domain_consultation import (
    DomainConsultationHandler,
    DomainConsultationResponse,
)
from harness.human_decision_policy import (
    ControlledAction,
    HumanDecisionKind,
    HumanDecisionVerdict,
)
from harness.routed_discord_adapter import (
    RoutedDecisionReply,
    RoutedDiscordService,
)


@dataclass(frozen=True)
class ComputeDecisionOutcome:
    decision: RoutedDecisionReply
    job: Job | None = None
    reason: str = ""


class ComputeAwareDomainConsultationHandler(DomainConsultationHandler):
    """Expose collected results to the AI and persist executable job proposals."""

    def __call__(self, ingress: DiscordIngressResult) -> DomainConsultationResponse:
        response = super().__call__(ingress)
        raw_proposals = _parse_job_proposals(response.message)
        proposals: list[HypothesisProposal] = []
        for index, raw in enumerate(raw_proposals[:5], 1):
            try:
                proposal = normalize_hypothesis_proposal(
                    raw,
                    domain=self.domain,
                    parent_job_id=(
                        str(raw["parent_job_id"])
                        if raw.get("parent_job_id")
                        else None
                    ),
                    parent_result_ref=(
                        str(raw["parent_result_ref"])
                        if raw.get("parent_result_ref")
                        else None
                    ),
                    seed=f"discord:{ingress.event.event_id}:{index}",
                )
            except (TypeError, ValueError):
                continue
            self.store.append_event(
                event_type=ResultFeedbackEngine.PROPOSAL_EVENT,
                lane=EventLane.DATA,
                project_id=ingress.route.project.project_id,
                work_session_id=ingress.route.work_session.work_session_id,
                actor="agent:discord-consultation",
                payload={
                    "source_event_id": ingress.event.event_id,
                    "proposal": proposal.to_dict(),
                    "requires_human_hypothesis_decision": True,
                    "requires_result_interpretation_first": bool(
                        proposal.parent_result_ref
                    ),
                },
                idempotency_key=(
                    f"discord:{ingress.event.event_id}:proposal:"
                    f"{proposal.subject_ref}"
                ),
            )
            proposals.append(proposal)

        message = _strip_job_proposal_block(response.message)
        if proposals:
            message = (
                message.rstrip()
                + "\n\n実行可能な仮説候補:\n"
                + "\n".join(
                    f"- `{item.subject_ref}`: {item.title}"
                    for item in proposals
                )
                + "\n試す候補は `/agent hypothesis` で人間が確定してください。"
            )
        return DomainConsultationResponse(
            domain=response.domain,
            work_session_id=response.work_session_id,
            request_event_id=response.request_event_id,
            response_event_id=response.response_event_id,
            message=message,
            runtime_provider=response.runtime_provider,
            cached=response.cached,
        )

    def _build_prompt(self, ingress: DiscordIngressResult) -> str:
        base = super()._build_prompt(ingress)
        context = latest_compute_context(
            self.store,
            work_session_id=ingress.route.work_session.work_session_id,
            limit=150,
        )
        return (
            base
            + "\n\n現在のCompute/実験コンテキスト:\n<UNTRUSTED_COMPUTE_CONTEXT>\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
            + "\n</UNTRUSTED_COMPUTE_CONTEXT>\n\n"
            + _proposal_protocol(self.domain)
        )


class AutonomousRoutedDiscordService(RoutedDiscordService):
    """Routed Discord service with accepted-hypothesis job creation."""

    def __init__(
        self,
        router: DiscordThreadRouter,
        dispatcher: DiscordChannelDispatcher,
        compute: ComputeStack,
    ) -> None:
        super().__init__(router, dispatcher)
        self.compute = compute
        self.scheduler: ComputeScheduler = compute.scheduler

    def start(self) -> None:
        self.scheduler.start(recover=True)

    def stop(self, *, wait: bool = False) -> None:
        self.scheduler.stop(wait=wait, cancel_active=False)

    def record_decision(
        self,
        location: DiscordLocation,
        *,
        title: str,
        kind: HumanDecisionKind | str,
        verdict: HumanDecisionVerdict | str,
        subject_ref: str,
        note: str,
        actor_id: str,
        message_id: str,
        actor_is_human: bool,
        project_id: str | None = None,
    ) -> RoutedDecisionReply:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        normalized_kind = HumanDecisionKind(kind)
        normalized_verdict = HumanDecisionVerdict(verdict)
        proposal: HypothesisProposal | None = None
        if (
            normalized_kind == HumanDecisionKind.HYPOTHESIS
            and normalized_verdict == HumanDecisionVerdict.ACCEPT
        ):
            proposal = find_hypothesis_proposal(
                self.router.store,
                work_session_id=route.work_session.work_session_id,
                subject_ref=subject_ref,
            )
            if proposal and proposal.parent_result_ref:
                interpretation = self.router.check_human_gate(
                    route,
                    action=ControlledAction.CONTINUE_FROM_RESULT,
                    subject_ref=proposal.parent_result_ref,
                )
                if not interpretation.allowed:
                    raise PermissionError(
                        "result interpretation must be accepted before selecting "
                        f"a child hypothesis from {proposal.parent_result_ref}"
                    )

        decision = super().record_decision(
            location,
            title=title,
            kind=normalized_kind,
            verdict=normalized_verdict,
            subject_ref=subject_ref,
            note=note,
            actor_id=actor_id,
            message_id=message_id,
            actor_is_human=actor_is_human,
            project_id=project_id,
        )
        if (
            normalized_kind == HumanDecisionKind.HYPOTHESIS
            and normalized_verdict == HumanDecisionVerdict.ACCEPT
        ):
            if proposal is None:
                self.router.store.append_event(
                    event_type="experiment.hypothesis.accepted_without_job_spec",
                    lane=EventLane.STATUS,
                    project_id=route.project.project_id,
                    work_session_id=route.work_session.work_session_id,
                    actor=f"discord-human:{actor_id}",
                    payload={
                        "subject_ref": decision.subject_ref,
                        "decision_event_id": decision.event_id,
                        "reason": (
                            "No structured hypothesis proposal exists. Ask the AI "
                            "to produce an executable proposal before retrying."
                        ),
                    },
                    idempotency_key=(
                        f"decision:{decision.event_id}:missing-job-spec"
                    ),
                )
            else:
                job = self._create_job_for_decision(route, proposal, decision)
                self.scheduler.enqueue(job.job_id)
        return decision

    def approve_compute(
        self,
        location: DiscordLocation,
        *,
        title: str,
        job_id: str,
        actor_id: str,
        project_id: str | None = None,
    ) -> Job:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        job = self.router.store.get_job(job_id)
        if job.spec.work_session_id != route.work_session.work_session_id:
            raise PermissionError("job does not belong to this Discord WorkSession")
        return self.scheduler.approve_job(
            job_id,
            actor=f"discord-human:{actor_id}",
        )

    def cancel_compute(
        self,
        location: DiscordLocation,
        *,
        title: str,
        job_id: str,
        actor_id: str,
        project_id: str | None = None,
    ) -> Job:
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        job = self.router.store.get_job(job_id)
        if job.spec.work_session_id != route.work_session.work_session_id:
            raise PermissionError("job does not belong to this Discord WorkSession")
        return self.scheduler.cancel_job(
            job_id,
            actor=f"discord-human:{actor_id}",
        )

    def status(
        self,
        location: DiscordLocation,
        *,
        title: str,
        project_id: str | None = None,
    ) -> str:
        base = super().status(
            location,
            title=title,
            project_id=project_id,
        )
        route = self.router.resolve_work_session(
            location,
            title=title,
            project_id=project_id,
        )
        jobs = self.router.store.list_jobs(
            work_session_id=route.work_session.work_session_id
        )
        context = latest_compute_context(
            self.router.store,
            work_session_id=route.work_session.work_session_id,
            limit=150,
        )
        job_lines = [
            (
                f"- {job.job_id}: {job.status.value}; "
                f"backend={job.backend_id or '-'}; "
                f"result={job.checkpoint_ref or '-'}"
            )
            for job in jobs[-10:]
        ] or ["- なし"]
        proposal_lines = []
        for item in context["pending_hypothesis_proposals"][-10:]:
            proposal = item.get("proposal")
            if isinstance(proposal, Mapping):
                proposal_lines.append(
                    f"- {proposal.get('subject_ref')}: {proposal.get('title')}"
                )
        if not proposal_lines:
            proposal_lines = ["- なし"]
        return "\n".join(
            [
                base,
                "Compute Scheduler:",
                json.dumps(self.scheduler.snapshot(), ensure_ascii=False),
                "Jobs:",
                *job_lines,
                "仮説候補（人間の選択待ち）:",
                *proposal_lines,
            ]
        )

    def _create_job_for_decision(
        self,
        route,
        proposal: HypothesisProposal,
        decision: RoutedDecisionReply,
    ) -> Job:
        if proposal.domain != route.domain:
            raise ValueError("hypothesis proposal domain does not match the channel")
        spec = proposal.to_job_spec(
            project_id=route.project.project_id,
            work_session_id=route.work_session.work_session_id,
        )
        digest = hashlib.sha256(
            (
                route.work_session.work_session_id
                + "\0"
                + proposal.subject_ref
            ).encode("utf-8")
        ).hexdigest()[:20]
        job_id = f"JOB-HYP-{digest}"
        try:
            job = self.router.store.create_job(spec, job_id=job_id)
        except ConflictError:
            job = self.router.store.get_job(job_id)
            if job.spec.to_dict() != spec.to_dict():
                raise ConflictError(
                    "accepted hypothesis is already bound to a different JobSpec"
                )
        self.router.store.append_event(
            event_type="experiment.hypothesis.accepted",
            lane=EventLane.CONTROL,
            project_id=route.project.project_id,
            work_session_id=route.work_session.work_session_id,
            job_id=job.job_id,
            actor="compute:discord-service",
            payload={
                "subject_ref": proposal.subject_ref,
                "decision_event_id": decision.event_id,
                "job_id": job.job_id,
                "backend_preferences": list(spec.backend_preferences),
                "resources": spec.resources.to_dict(),
            },
            idempotency_key=f"decision:{decision.event_id}:job:{job.job_id}",
        )
        return job


def build_autonomous_routed_service(
    config: HarnessConfig,
    router: DiscordThreadRouter,
) -> AutonomousRoutedDiscordService:
    compute = build_default_compute_stack(config, router.store)
    dispatcher = DiscordChannelDispatcher(
        router,
        {
            Domain.RESEARCH: ComputeAwareDomainConsultationHandler(
                config,
                router.store,
                Domain.RESEARCH,
            ),
            Domain.KAGGLE: ComputeAwareDomainConsultationHandler(
                config,
                router.store,
                Domain.KAGGLE,
            ),
        },
    )
    return AutonomousRoutedDiscordService(router, dispatcher, compute)


def _proposal_protocol(domain: Domain) -> str:
    default_preferences = (
        ["kaggle_notebook", "remote_gpu", "local_gpu", "local_cpu"]
        if domain == Domain.KAGGLE
        else ["remote_gpu", "local_gpu", "local_cpu"]
    )
    return f"""実行可能な仮説を提案する場合は、通常の説明の末尾に次のJSONを1つだけ fenced code block で付けてください。
人間の仮説選択を代行してはいけません。提案しない場合はJSONを付けないでください。

```json
{{
  "job_proposals": [
    {{
      "subject_ref": "hypothesis:短い一意ID",
      "title": "短い実験名",
      "hypothesis": "検証可能な仮説",
      "implementation_prompt": "Codexが実装できる具体的要件",
      "entrypoint": "既存コードならargv文字列。新規実装なら空文字",
      "smoke_command": "GPU長時間実行前に通す短いコマンド",
      "resources": {{
        "cpu_cores": 4,
        "memory_mb": 8192,
        "gpu_count": 1,
        "gpu_memory_mb": 12000,
        "accelerator": "gpu",
        "ephemeral_storage_mb": 20480,
        "network_required": false,
        "labels": ["training"]
      }},
      "backend_preferences": {json.dumps(default_preferences)},
      "max_runtime_seconds": 7200,
      "outputs": ["result.json", "metrics.json", "progress.json"],
      "priority": 0,
      "parent_job_id": null,
      "parent_result_ref": null,
      "metadata": {{"success_condition": "採択条件", "failure_condition": "棄却条件"}}
    }}
  ]
}}
```

resultから派生した候補では parent_result_ref を正確な result:<job>:<hash> にし、可能なら parent_job_id も設定してください。
"""


def _parse_job_proposals(text: str) -> list[dict[str, Any]]:
    candidates = [text]
    if "```" in text:
        parts = text.split("```")
        candidates = [parts[index] for index in range(1, len(parts), 2)] + candidates
    for candidate in candidates:
        stripped = candidate.strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and isinstance(
            value.get("job_proposals"), list
        ):
            return [
                dict(item)
                for item in value["job_proposals"]
                if isinstance(item, Mapping)
            ]
    return []


def _strip_job_proposal_block(text: str) -> str:
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    kept: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            kept.append(part)
            continue
        candidate = part.strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            kept.append("```" + part + "```")
            continue
        if not (
            isinstance(value, Mapping)
            and isinstance(value.get("job_proposals"), list)
        ):
            kept.append("```" + part + "```")
    return "".join(kept).strip()
