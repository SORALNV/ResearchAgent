from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from harness.papers import PaperStore


_CITATION_PATTERN = re.compile(r"\[(P-\d{3})\]")


@dataclass(frozen=True)
class EvalResult:
    total_questions: int
    papers_available: int
    required_source_type_hits: int
    must_not_claim_hits: int
    citation_ready: bool
    summary: str
    expected_keyword_hits: int = 0
    expected_keyword_total: int = 0
    valid_citation_count: int = 0
    invalid_citations: tuple[str, ...] = ()
    execution_calls: int = 0
    execution_successes: int = 0
    execution_success_rate: float = 0.0
    review_calls: int = 0
    review_successes: int = 0
    artifact_integrity_ok: bool = True
    artifact_error_count: int = 0
    overall_score: float = 0.0
    question_details: tuple[dict[str, object], ...] = field(default_factory=tuple)


def run_golden_eval(
    golden_path: Path,
    paper_store: PaperStore,
    *,
    report_text: str = "",
    ledger_text: str = "",
    journal_entries: Iterable[dict[str, Any]] | None = None,
    research_dir: Path | None = None,
) -> EvalResult:
    """Evaluate the produced answer, citation integrity, and execution trace.

    The legacy two-argument call remains supported. If no generated answer is
    supplied, paper summaries are used as a compatibility corpus.
    """

    questions = _read_jsonl(golden_path)
    papers = paper_store.read_all()
    source_types = {paper.source for paper in papers}
    valid_ids = {paper.paper_id for paper in papers}
    paper_corpus = "\n".join(paper.summary for paper in papers)
    answer_corpus = "\n".join(
        item for item in (report_text, ledger_text) if item.strip()
    )
    evaluation_corpus = answer_corpus or paper_corpus
    lowered_corpus = evaluation_corpus.lower()

    required_hits = 0
    forbidden_hits = 0
    keyword_hits = 0
    keyword_total = 0
    details: list[dict[str, object]] = []

    for question in questions:
        required = {
            str(item) for item in question.get("required_source_types", [])
        }
        source_hit = bool(required & source_types)
        if source_hit:
            required_hits += 1

        expected = [
            str(item) for item in question.get("expected_keywords", [])
        ]
        hit_keywords = [
            keyword for keyword in expected if keyword.lower() in lowered_corpus
        ]
        keyword_hits += len(hit_keywords)
        keyword_total += len(expected)

        forbidden = [
            str(item) for item in question.get("must_not_claim", [])
        ]
        found_forbidden = [
            claim for claim in forbidden if claim.lower() in lowered_corpus
        ]
        forbidden_hits += len(found_forbidden)

        details.append(
            {
                "question": str(question.get("question") or ""),
                "source_type_hit": source_hit,
                "expected_keyword_hits": hit_keywords,
                "expected_keyword_total": len(expected),
                "forbidden_claim_hits": found_forbidden,
            }
        )

    cited_ids = set(_CITATION_PATTERN.findall(answer_corpus))
    invalid = tuple(sorted(cited_ids - valid_ids))
    valid_citation_count = len(cited_ids & valid_ids)
    if answer_corpus:
        citation_ready = bool(papers) and bool(cited_ids) and not invalid
    else:
        citation_ready = bool(papers) and all(
            paper.paper_id.startswith("P-") for paper in papers
        )

    execution_calls = 0
    execution_successes = 0
    review_calls = 0
    review_successes = 0
    for invocation in _iter_invocations(journal_entries or []):
        stage = str(invocation.get("stage") or "")
        if not stage:
            continue
        execution_calls += 1
        ok = _invocation_ok(invocation)
        if ok:
            execution_successes += 1
        if stage == "review":
            review_calls += 1
            if ok and _review_output_looks_structured(
                str(invocation.get("output") or "")
            ):
                review_successes += 1

    execution_success_rate = (
        execution_successes / execution_calls if execution_calls else 0.0
    )
    artifact_integrity_ok, artifact_errors = _artifact_integrity(research_dir)

    keyword_coverage = keyword_hits / keyword_total if keyword_total else 1.0
    source_coverage = required_hits / len(questions) if questions else 1.0
    citation_score = 1.0 if citation_ready else 0.0
    safety_score = 1.0 if forbidden_hits == 0 else 0.0
    execution_score = execution_success_rate if execution_calls else 0.5
    artifact_score = 1.0 if artifact_integrity_ok else 0.0
    overall_score = round(
        100
        * (
            0.30 * keyword_coverage
            + 0.15 * source_coverage
            + 0.20 * citation_score
            + 0.15 * safety_score
            + 0.10 * execution_score
            + 0.10 * artifact_score
        ),
        1,
    )

    summary = (
        f"questions={len(questions)}, papers={len(papers)}, "
        f"required_source_type_hits={required_hits}, "
        f"must_not_claim_hits={forbidden_hits}, "
        f"citation_ready={citation_ready}\n"
        f"answer_keyword_coverage={keyword_hits}/{keyword_total}, "
        f"valid_citations={valid_citation_count}, "
        f"invalid_citations={list(invalid)}, "
        f"execution_success={execution_successes}/{execution_calls}, "
        f"review_success={review_successes}/{review_calls}, "
        f"artifact_integrity_ok={artifact_integrity_ok}, "
        f"overall_score={overall_score}"
    )
    return EvalResult(
        total_questions=len(questions),
        papers_available=len(papers),
        required_source_type_hits=required_hits,
        must_not_claim_hits=forbidden_hits,
        citation_ready=citation_ready,
        summary=summary,
        expected_keyword_hits=keyword_hits,
        expected_keyword_total=keyword_total,
        valid_citation_count=valid_citation_count,
        invalid_citations=invalid,
        execution_calls=execution_calls,
        execution_successes=execution_successes,
        execution_success_rate=execution_success_rate,
        review_calls=review_calls,
        review_successes=review_successes,
        artifact_integrity_ok=artifact_integrity_ok,
        artifact_error_count=len(artifact_errors),
        overall_score=overall_score,
        question_details=tuple(details),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    result: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _iter_invocations(
    journal_entries: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for entry in journal_entries:
        sessions = entry.get("conversation_sessions")
        if not isinstance(sessions, list):
            continue
        for invocation in sessions:
            if isinstance(invocation, dict):
                yield invocation


def _invocation_ok(invocation: dict[str, Any]) -> bool:
    try:
        returncode = int(invocation.get("returncode") or 0)
    except (TypeError, ValueError):
        return False
    return (
        returncode == 0
        and not bool(invocation.get("skipped"))
        and not bool(invocation.get("timed_out"))
        and not bool(invocation.get("cancelled"))
    )


def _review_output_looks_structured(output: str) -> bool:
    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        value = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and value.get("verdict") in {"accept", "revise"}


def _artifact_integrity(
    research_dir: Path | None,
) -> tuple[bool, list[str]]:
    if research_dir is None:
        return True, []
    final_root = research_dir / "artifacts" / "final"
    if not final_root.exists():
        return True, []

    errors: list[str] = []
    manifests = list(final_root.rglob("promotion_manifest.json"))
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{manifest}: {type(exc).__name__}")
            continue
        for item in payload.get("errors", []):
            errors.append(str(item))
        for item in payload.get("promoted", []):
            if not isinstance(item, dict):
                errors.append(f"{manifest}: invalid promoted entry")
                continue
            destination = Path(str(item.get("destination") or ""))
            if not destination.is_file():
                errors.append(f"missing promoted artifact: {destination}")
    return not errors, errors
