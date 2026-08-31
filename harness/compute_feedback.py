from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from harness.compute_models import (
    FeedbackPlanner,
    canonical_json_hash,
    json_copy,
    safe_relative_path,
)
from harness.config import HarnessConfig
from harness.control_plane import (
    ControlPlaneStore,
    Domain,
    Event,
    EventLane,
    Job,
    JobSpec,
    ResourceRequirements,
)
from harness.process_manager import ProcessCancellationController
from harness.provider_runtime import ProviderAwareAgentCommandExecutor
from harness.state import ResearchSession, utc_timestamp


@dataclass(frozen=True)
class HypothesisProposal:
    subject_ref: str
    domain: Domain
    title: str
    hypothesis: str
    implementation_prompt: str = ""
    entrypoint: str | tuple[str, ...] = ""
    smoke_command: str | tuple[str, ...] = ""
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    backend_preferences: tuple[str, ...] = ()
    max_runtime_seconds: int | None = None
    outputs: tuple[str, ...] = ()
    priority: int = 0
    parent_job_id: str | None = None
    parent_result_ref: str | None = None
    experiment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_ref": self.subject_ref,
            "domain": self.domain.value,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "implementation_prompt": self.implementation_prompt,
            "entrypoint": (
                list(self.entrypoint)
                if isinstance(self.entrypoint, tuple)
                else self.entrypoint
            ),
            "smoke_command": (
                list(self.smoke_command)
                if isinstance(self.smoke_command, tuple)
                else self.smoke_command
            ),
            "resources": self.resources.to_dict(),
            "backend_preferences": list(self.backend_preferences),
            "max_runtime_seconds": self.max_runtime_seconds,
            "outputs": list(self.outputs),
            "priority": self.priority,
            "parent_job_id": self.parent_job_id,
            "parent_result_ref": self.parent_result_ref,
            "experiment_id": self.experiment_id,
            "metadata": json_copy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HypothesisProposal":
        domain = Domain(str(data["domain"]))
        subject_ref = str(data.get("subject_ref") or "").strip()
        if not subject_ref:
            raise ValueError("hypothesis proposal requires subject_ref")
        if not subject_ref.startswith("hypothesis:"):
            subject_ref = "hypothesis:" + subject_ref
        title = str(data.get("title") or "").strip()
        hypothesis = str(data.get("hypothesis") or title).strip()
        if not title or not hypothesis:
            raise ValueError("hypothesis proposal requires title and hypothesis")
        return cls(
            subject_ref=subject_ref,
            domain=domain,
            title=title,
            hypothesis=hypothesis,
            implementation_prompt=str(
                data.get("implementation_prompt") or ""
            ).strip(),
            entrypoint=_command_value(data.get("entrypoint")),
            smoke_command=_command_value(data.get("smoke_command")),
            resources=ResourceRequirements.from_dict(
                data.get("resources")
                if isinstance(data.get("resources"), Mapping)
                else None
            ),
            backend_preferences=tuple(
                str(item)
                for item in data.get("backend_preferences", [])
                if str(item).strip()
            ),
            max_runtime_seconds=(
                int(data["max_runtime_seconds"])
                if data.get("max_runtime_seconds") is not None
                else None
            ),
            outputs=tuple(
                item
                for item in (
                    safe_relative_path(str(value))
                    for value in data.get("outputs", [])
                )
                if item
            ),
            priority=int(data.get("priority") or 0),
            parent_job_id=(
                str(data["parent_job_id"])
                if data.get("parent_job_id")
                else None
            ),
            parent_result_ref=(
                str(data["parent_result_ref"])
                if data.get("parent_result_ref")
                else None
            ),
            experiment_id=(
                str(data["experiment_id"])
                if data.get("experiment_id")
                else None
            ),
            metadata=json_copy(
                data.get("metadata")
                if isinstance(data.get("metadata"), Mapping)
                else {}
            ),
        )

    def to_job_spec(
        self,
        *,
        project_id: str,
        work_session_id: str,
    ) -> JobSpec:
        payload: dict[str, Any] = {
            "title": self.title,
            "hypothesis": self.hypothesis,
            "implementation_prompt": self.implementation_prompt,
            "entrypoint": (
                list(self.entrypoint)
                if isinstance(self.entrypoint, tuple)
                else self.entrypoint
            ),
            "smoke_command": (
                list(self.smoke_command)
                if isinstance(self.smoke_command, tuple)
                else self.smoke_command
            ),
            "outputs": list(self.outputs),
            "hypothesis_subject_ref": self.subject_ref,
            "parent_result_ref": self.parent_result_ref,
            "proposal_metadata": json_copy(self.metadata),
        }
        return JobSpec(
            project_id=project_id,
            work_session_id=work_session_id,
            domain=self.domain,
            kind="experiment",
            payload=payload,
            resources=self.resources,
            backend_preferences=self.backend_preferences,
            max_runtime_seconds=self.max_runtime_seconds,
            priority=self.priority,
            parent_job_id=self.parent_job_id,
            experiment_id=self.experiment_id or self.subject_ref,
            requires_approval=False,
        )


@dataclass(frozen=True)
class FeedbackOutcome:
    result_ref: str
    result_event_id: str
    proposal_event_ids: tuple[str, ...]
    proposals: tuple[HypothesisProposal, ...]
    result: dict[str, Any]
    feedback_path: str


class ProviderFeedbackPlanner:
    """Use the configured planning provider to turn results into hypotheses."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        executor: Any | None = None,
    ) -> None:
        self.config = config
        self.cancellation = ProcessCancellationController(
            config.agent_cancel_grace_seconds
        )
        self.executor = executor or ProviderAwareAgentCommandExecutor(
            config,
            threading.RLock(),
            self.cancellation,
        )

    def propose(
        self,
        *,
        job: Job,
        result: Mapping[str, Any],
        result_ref: str,
    ) -> list[dict[str, Any]]:
        workspace = Path(
            str(job.spec.payload.get("workspace") or self.config.project_root)
        ).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        session = ResearchSession.new(
            str(job.spec.payload.get("hypothesis") or job.spec.kind),
            project_name=(
                "KaggleAgent"
                if job.spec.domain == Domain.KAGGLE
                else "ResearchAgent"
            ),
        )
        session.session_id = job.spec.work_session_id
        session.research_dir = str(workspace)
        invocation = self.executor.run(
            session=session,
            role="planning",
            stage="compute_result_feedback",
            prompt=_feedback_prompt(job, result, result_ref),
            command_text=(
                self.config.main_agent_command
                or self.config.sub_agent_command
                or self.config.review_agent_command
            ),
            sandbox="read-only",
            task_id=job.job_id,
            working_dir=workspace,
        )
        if not bool(getattr(invocation, "ok", False)):
            return []
        return _parse_proposal_output(
            str(getattr(invocation, "output", "") or "")
        )


class RuleBasedFeedbackPlanner:
    """Fail-safe fallback that always creates a reviewable next-step proposal."""

    def propose(
        self,
        *,
        job: Job,
        result: Mapping[str, Any],
        result_ref: str,
    ) -> list[dict[str, Any]]:
        metric = _primary_metric(result)
        metric_text = (
            f"{metric['name']}={metric['value']}"
            if metric is not None
            else "primary metric is not structured"
        )
        return [
            {
                "title": "結果を再検証し、単一要因だけ変更する次実験",
                "hypothesis": (
                    f"{result_ref} の {metric_text} を再現確認し、"
                    "最も影響が大きいと推定した要因を1つだけ変更すれば、"
                    "因果の切り分けと再現性を改善できる"
                ),
                "implementation_prompt": (
                    "親実験のコードと成果物を読み、同じ評価手順を維持したまま、"
                    "変更点を1つに限定したchild experimentを実装する。"
                    "result.json、metrics.json、progress.jsonを出力し、"
                    "変更点・再現条件・失敗条件を記録する。"
                ),
                "entrypoint": "",
                "smoke_command": "",
                "resources": job.spec.resources.to_dict(),
                "backend_preferences": list(job.spec.backend_preferences),
                "max_runtime_seconds": job.spec.max_runtime_seconds,
                "outputs": ["result.json", "metrics.json", "progress.json"],
                "priority": job.spec.priority,
                "metadata": {"planner": "rule_based_fallback"},
            }
        ]


class ResultFeedbackEngine:
    """Persist collected results and make them available as next hypotheses."""

    RESULT_EVENT = "experiment.result.collected"
    PROPOSAL_EVENT = "experiment.hypothesis.proposed"
    FEEDBACK_EVENT = "experiment.feedback.generated"

    def __init__(
        self,
        store: ControlPlaneStore,
        root_dir: str | Path,
        *,
        planner: FeedbackPlanner | None = None,
        fallback_planner: FeedbackPlanner | None = None,
        max_proposals: int = 5,
    ) -> None:
        self.store = store
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.planner = planner
        self.fallback_planner = fallback_planner or RuleBasedFeedbackPlanner()
        self.max_proposals = max(1, max_proposals)

    def integrate(
        self,
        *,
        job: Job,
        collected_result: Mapping[str, Any] | None,
        artifacts_dir: str | Path,
        artifact_refs: tuple[str, ...] | list[str] = (),
        backend: str,
    ) -> FeedbackOutcome:
        artifact_root = Path(artifacts_dir).expanduser().resolve()
        result = _load_structured_result(artifact_root, collected_result)
        digest = canonical_json_hash(
            {
                "job_id": job.job_id,
                "backend": backend,
                "result": result,
                "artifacts": list(artifact_refs),
            }
        )
        result_ref = f"result:{job.job_id}:{digest[:16]}"
        result_event = self.store.append_event(
            event_type=self.RESULT_EVENT,
            lane=EventLane.DATA,
            project_id=job.spec.project_id,
            work_session_id=job.spec.work_session_id,
            job_id=job.job_id,
            actor=f"compute:{backend}",
            payload={
                "result_ref": result_ref,
                "backend": backend,
                "result": result,
                "artifact_refs": list(artifact_refs),
                "artifacts_dir": str(artifact_root),
                "requires_human_interpretation": True,
            },
            idempotency_key=f"compute:{job.job_id}:result:{digest}",
        )

        raw_proposals = _embedded_proposals(result)
        if not raw_proposals and self.planner is not None:
            try:
                raw_proposals = self.planner.propose(
                    job=job,
                    result=result,
                    result_ref=result_ref,
                )
            except Exception:
                raw_proposals = []
        if not raw_proposals:
            raw_proposals = self.fallback_planner.propose(
                job=job,
                result=result,
                result_ref=result_ref,
            )

        proposals: list[HypothesisProposal] = []
        proposal_events: list[Event] = []
        for index, raw in enumerate(raw_proposals[: self.max_proposals], 1):
            try:
                proposal = normalize_hypothesis_proposal(
                    raw,
                    domain=job.spec.domain,
                    parent_job_id=job.job_id,
                    parent_result_ref=result_ref,
                    seed=f"{digest}:{index}",
                )
            except (TypeError, ValueError):
                continue
            event = self.store.append_event(
                event_type=self.PROPOSAL_EVENT,
                lane=EventLane.DATA,
                project_id=job.spec.project_id,
                work_session_id=job.spec.work_session_id,
                job_id=job.job_id,
                actor="agent:feedback",
                payload={
                    "result_ref": result_ref,
                    "proposal": proposal.to_dict(),
                    "requires_human_hypothesis_decision": True,
                    "requires_result_interpretation_first": True,
                },
                idempotency_key=(
                    f"compute:{job.job_id}:proposal:{proposal.subject_ref}"
                ),
            )
            proposals.append(proposal)
            proposal_events.append(event)

        feedback_path = (
            self.root_dir
            / "feedback"
            / job.spec.work_session_id
            / f"{job.job_id}.json"
        )
        _atomic_json(
            feedback_path,
            {
                "generated_at": utc_timestamp(),
                "result_ref": result_ref,
                "result_event_id": result_event.event_id,
                "result": result,
                "proposals": [item.to_dict() for item in proposals],
                "proposal_event_ids": [item.event_id for item in proposal_events],
            },
        )
        self.store.append_event(
            event_type=self.FEEDBACK_EVENT,
            lane=EventLane.STATUS,
            project_id=job.spec.project_id,
            work_session_id=job.spec.work_session_id,
            job_id=job.job_id,
            actor="agent:feedback",
            payload={
                "result_ref": result_ref,
                "proposal_count": len(proposals),
                "proposal_subject_refs": [
                    item.subject_ref for item in proposals
                ],
                "feedback_path": str(feedback_path),
                "next_required_human_action": "result_interpretation",
            },
            idempotency_key=f"compute:{job.job_id}:feedback:{digest}",
        )
        return FeedbackOutcome(
            result_ref=result_ref,
            result_event_id=result_event.event_id,
            proposal_event_ids=tuple(item.event_id for item in proposal_events),
            proposals=tuple(proposals),
            result=result,
            feedback_path=str(feedback_path),
        )


def normalize_hypothesis_proposal(
    raw: Mapping[str, Any],
    *,
    domain: Domain,
    parent_job_id: str | None,
    parent_result_ref: str | None,
    seed: str,
) -> HypothesisProposal:
    value = dict(raw)
    title = str(value.get("title") or value.get("name") or "").strip()
    hypothesis = str(
        value.get("hypothesis") or value.get("description") or title
    ).strip()
    if not title or not hypothesis:
        raise ValueError("proposal requires title and hypothesis")
    raw_subject = str(
        value.get("subject_ref") or value.get("hypothesis_id") or ""
    ).strip()
    if raw_subject:
        subject_ref = (
            raw_subject
            if raw_subject.startswith("hypothesis:")
            else "hypothesis:" + raw_subject
        )
    else:
        proposal_digest = hashlib.sha256(
            (seed + "\0" + title + "\0" + hypothesis).encode("utf-8")
        ).hexdigest()[:20]
        subject_ref = f"hypothesis:{proposal_digest}"
    resources = ResourceRequirements.from_dict(
        value.get("resources")
        if isinstance(value.get("resources"), Mapping)
        else None
    )
    outputs_raw = value.get("outputs") or []
    if isinstance(outputs_raw, str):
        outputs_raw = [outputs_raw]
    outputs = tuple(
        item
        for item in (
            safe_relative_path(str(output)) for output in outputs_raw
        )
        if item
    )
    return HypothesisProposal(
        subject_ref=subject_ref,
        domain=domain,
        title=title,
        hypothesis=hypothesis,
        implementation_prompt=str(
            value.get("implementation_prompt")
            or value.get("implementation")
            or ""
        ).strip(),
        entrypoint=_command_value(value.get("entrypoint")),
        smoke_command=_command_value(value.get("smoke_command")),
        resources=resources,
        backend_preferences=tuple(
            str(item)
            for item in value.get("backend_preferences", [])
            if str(item).strip()
        ),
        max_runtime_seconds=(
            int(value["max_runtime_seconds"])
            if value.get("max_runtime_seconds") is not None
            else None
        ),
        outputs=outputs,
        priority=int(value.get("priority") or 0),
        parent_job_id=parent_job_id,
        parent_result_ref=parent_result_ref,
        experiment_id=(
            str(value["experiment_id"])
            if value.get("experiment_id")
            else subject_ref
        ),
        metadata=json_copy(
            value.get("metadata")
            if isinstance(value.get("metadata"), Mapping)
            else {}
        ),
    )


def find_hypothesis_proposal(
    store: ControlPlaneStore,
    *,
    work_session_id: str,
    subject_ref: str,
    limit: int = 2000,
) -> HypothesisProposal | None:
    normalized = str(subject_ref).strip()
    if normalized and not normalized.startswith("hypothesis:"):
        normalized = "hypothesis:" + normalized
    for event in reversed(
        store.latest_events(
            work_session_id=work_session_id,
            lanes=[EventLane.DATA],
            limit=max(1, limit),
        )
    ):
        if event.event_type != ResultFeedbackEngine.PROPOSAL_EVENT:
            continue
        raw = event.payload.get("proposal")
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("subject_ref") or "") != normalized:
            continue
        try:
            return HypothesisProposal.from_dict(raw)
        except (TypeError, ValueError):
            return None
    return None


def latest_compute_context(
    store: ControlPlaneStore,
    *,
    work_session_id: str,
    limit: int = 100,
) -> dict[str, Any]:
    jobs = store.list_jobs(work_session_id=work_session_id)
    events = store.latest_events(
        work_session_id=work_session_id,
        lanes=[EventLane.DATA, EventLane.STATUS],
        limit=max(1, limit),
    )
    results: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    for event in events:
        if event.event_type == ResultFeedbackEngine.RESULT_EVENT:
            results.append(
                {
                    "event_id": event.event_id,
                    "job_id": event.job_id,
                    **dict(event.payload),
                }
            )
        elif event.event_type == ResultFeedbackEngine.PROPOSAL_EVENT:
            proposals.append(
                {
                    "event_id": event.event_id,
                    "job_id": event.job_id,
                    **dict(event.payload),
                }
            )
    return {
        "jobs": [item.to_dict() for item in jobs[-50:]],
        "recent_results": results[-10:],
        "pending_hypothesis_proposals": proposals[-20:],
    }


def _embedded_proposals(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = result.get("next_hypotheses") or result.get("hypothesis_proposals")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _load_structured_result(
    artifact_root: Path,
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = json_copy(supplied or {})
    for name in ("result.json", "metrics.json"):
        path = artifact_root / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        if name == "metrics.json":
            result.setdefault("metrics", dict(value))
        else:
            result.update(dict(value))
    result.setdefault("collected_at", utc_timestamp())
    return result


def _feedback_prompt(
    job: Job,
    result: Mapping[str, Any],
    result_ref: str,
) -> str:
    domain_rules = (
        "KaggleではCV固定、リーク回避、再現性、submissionを勝手に行わないことを守る。"
        if job.spec.domain == Domain.KAGGLE
        else "研究では反証可能性、先行研究との差分、再現性を重視する。"
    )
    return f"""あなたはResearchAgentの実験結果レビュー担当です。
{domain_rules}
人間が担うのは、何を試すかの最終選択、結果解釈の確定、そしてKaggle提出または論文化の最終判断だけです。
あなたは結果から次に検証すべき仮説候補を最大3件生成してください。ただし自動承認・自動実行はしません。

Job:
{json.dumps(job.to_dict(), ensure_ascii=False, indent=2)}

result_ref: {result_ref}
Result:
{json.dumps(dict(result), ensure_ascii=False, indent=2)}

JSONだけを返してください。形式:
{{
  "proposals": [
    {{
      "title": "短い名称",
      "hypothesis": "検証可能な仮説",
      "implementation_prompt": "Codexが実装可能な具体的要件",
      "entrypoint": "未実装なら空文字。既存ならargv文字列",
      "smoke_command": "短いsmoke test。未定なら空文字",
      "resources": {{"cpu_cores": 2, "memory_mb": 4096, "gpu_count": 1, "gpu_memory_mb": 12000, "accelerator": "gpu", "network_required": false, "labels": ["training"]}},
      "backend_preferences": [],
      "max_runtime_seconds": 3600,
      "outputs": ["result.json", "metrics.json", "progress.json"],
      "priority": 0,
      "metadata": {{"expected_signal": "改善または棄却条件"}}
    }}
  ]
}}
"""


def _parse_proposal_output(text: str) -> list[dict[str, Any]]:
    candidates = [text.strip()]
    if "```" in text:
        parts = text.split("```")
        candidates.extend(parts[index].strip() for index in range(1, len(parts), 2))
    for candidate in candidates:
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and isinstance(value.get("proposals"), list):
            return [
                dict(item)
                for item in value["proposals"]
                if isinstance(item, Mapping)
            ]
    return []


def _primary_metric(result: Mapping[str, Any]) -> dict[str, Any] | None:
    value = result.get("primary_metric")
    if isinstance(value, Mapping) and value.get("name") is not None:
        return dict(value)
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        for name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and not isinstance(
                metric_value, bool
            ):
                return {
                    "name": str(name),
                    "value": metric_value,
                    "direction": "unknown",
                }
    return None


def _command_value(value: Any) -> str | tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return str(value or "").strip()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
