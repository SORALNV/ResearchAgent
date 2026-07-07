from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from harness.papers import PaperStore


@dataclass(frozen=True)
class EvalResult:
    total_questions: int
    papers_available: int
    required_source_type_hits: int
    must_not_claim_hits: int
    citation_ready: bool
    summary: str


def run_golden_eval(golden_path: Path, paper_store: PaperStore) -> EvalResult:
    questions = _read_jsonl(golden_path)
    papers = paper_store.read_all()
    source_types = {paper.source for paper in papers}
    required_hits = 0
    must_not_hits = 0
    for question in questions:
        required = set(question.get("required_source_types", []))
        if required & source_types:
            required_hits += 1
        for forbidden in question.get("must_not_claim", []):
            forbidden_lower = forbidden.lower()
            if any(forbidden_lower in paper.summary.lower() for paper in papers):
                must_not_hits += 1
    citation_ready = bool(papers) and all(paper.paper_id.startswith("P-") for paper in papers)
    return EvalResult(
        total_questions=len(questions),
        papers_available=len(papers),
        required_source_type_hits=required_hits,
        must_not_claim_hits=must_not_hits,
        citation_ready=citation_ready,
        summary=(
            f"questions={len(questions)}, papers={len(papers)}, "
            f"required_source_type_hits={required_hits}, must_not_claim_hits={must_not_hits}, "
            f"citation_ready={citation_ready}"
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

