from __future__ import annotations

from typing import Any, Mapping, Sequence

from harness.control_plane import Job
from harness.iteration_memo import (
    IterationMemoPlanner,
    IterationOutcome,
    classify_outcome,
    observe_primary_metric,
)
from harness.kaggle_methodbook import EvidenceKind, MethodCardStore


class MethodBookAwareMemoPlanner:
    """Bind reused MethodCards back to the outcome of the experiment that used them."""

    def __init__(
        self,
        delegate: IterationMemoPlanner,
        method_store: MethodCardStore,
    ) -> None:
        self.delegate = delegate
        self.method_store = method_store

    def summarize(
        self,
        *,
        job: Job,
        result: Mapping[str, Any],
        result_ref: str,
        proposals: Sequence[Mapping[str, Any]],
        backend: str,
    ) -> Mapping[str, Any]:
        raw = dict(
            self.delegate.summarize(
                job=job,
                result=result,
                result_ref=result_ref,
                proposals=proposals,
                backend=backend,
            )
        )
        metric = observe_primary_metric(job, result)
        outcome = classify_outcome(result, metric)
        evidence_kind = (
            EvidenceKind.SUPPORT
            if outcome == IterationOutcome.IMPROVED
            else EvidenceKind.COUNTER
            if outcome == IterationOutcome.REGRESSED
            else None
        )
        candidates = [
            dict(item)
            for item in raw.get("method_candidates", [])
            if isinstance(item, Mapping)
        ]
        known_ids = {
            str(item.get("method_id") or "").strip()
            for item in candidates
            if str(item.get("method_id") or "").strip()
        }
        referenced_ids = _referenced_method_ids(job)
        if evidence_kind is not None:
            for method_id in referenced_ids:
                if method_id in known_ids:
                    continue
                card = self.method_store.get(method_id)
                if card is None:
                    continue
                candidates.append(
                    {
                        "method_id": card.method_id,
                        "claim": card.claim,
                        "scope": card.scope.to_dict(),
                        "evidence_kind": evidence_kind.value,
                        "next_falsification": card.next_falsification,
                        "rationale": (
                            f"This experiment explicitly reused {card.method_id} and "
                            f"finished as {outcome.value} on {metric.name or 'the primary metric'}."
                        ),
                    }
                )
                known_ids.add(method_id)
        raw["method_candidates"] = candidates
        # The provider may explain the result, but it is not allowed to decide
        # whether the stored metric improved or regressed. Recompute that from
        # the immutable Job/result pair and overwrite any model-supplied value.
        raw["outcome"] = outcome.value
        raw["observed_outcome"] = outcome.value
        raw["referenced_method_card_ids"] = list(referenced_ids)
        return raw


def _referenced_method_ids(job: Job) -> tuple[str, ...]:
    payload = job.spec.payload
    metadata = payload.get("proposal_metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    raw_values: list[Any] = []
    for value in (
        payload.get("method_card_ids"),
        metadata_map.get("method_card_ids"),
    ):
        if isinstance(value, str):
            raw_values.extend(item.strip() for item in value.split(","))
        elif isinstance(value, Sequence):
            raw_values.extend(value)
    return tuple(
        dict.fromkeys(
            text
            for item in raw_values
            if (text := str(item or "").strip())
        )
    )
