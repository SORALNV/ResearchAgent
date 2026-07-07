from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from harness.state import utc_timestamp


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    url: str | None
    doi: str | None
    arxiv_id: str | None
    abstract: str
    summary: str
    source: str
    retrieved_at: str
    relevance_score: float
    confidence: str
    used_in_rounds: list[int] = field(default_factory=list)

    def identity_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        return f"title:{normalize_title(self.title)}"


def normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def summarize_abstract(title: str, abstract: str, max_chars: int = 260) -> str:
    text = " ".join(abstract.split())
    if not text:
        return f"{title}: abstract未取得のため要約は未確認。"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


class PaperStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read_all(self) -> list[Paper]:
        if not self.path.exists():
            return []
        papers = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                papers.append(Paper(**json.loads(line)))
        return papers

    def write_all(self, papers: Iterable[Paper]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for paper in papers:
                handle.write(json.dumps(asdict(paper), ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(self.path)

    def upsert_many(self, candidates: Iterable[Paper]) -> tuple[list[Paper], list[Paper]]:
        existing = self.read_all()
        by_key = {paper.identity_key(): paper for paper in existing}
        next_index = self._next_index(existing)
        inserted: list[Paper] = []
        updated: list[Paper] = []
        for candidate in candidates:
            key = candidate.identity_key()
            if key in by_key:
                merged = merge_papers(by_key[key], candidate)
                by_key[key] = merged
                updated.append(merged)
            else:
                candidate.paper_id = f"P-{next_index:03d}"
                next_index += 1
                by_key[key] = candidate
                inserted.append(candidate)
        self.write_all(sorted(by_key.values(), key=lambda paper: paper.paper_id))
        return inserted, updated

    def get(self, paper_id: str) -> Paper | None:
        for paper in self.read_all():
            if paper.paper_id == paper_id:
                return paper
        return None

    def _next_index(self, papers: list[Paper]) -> int:
        max_seen = 0
        for paper in papers:
            if paper.paper_id.startswith("P-"):
                try:
                    max_seen = max(max_seen, int(paper.paper_id[2:]))
                except ValueError:
                    continue
        return max_seen + 1


def merge_papers(existing: Paper, incoming: Paper) -> Paper:
    existing.authors = existing.authors or incoming.authors
    existing.year = existing.year or incoming.year
    existing.venue = existing.venue or incoming.venue
    existing.url = existing.url or incoming.url
    existing.doi = existing.doi or incoming.doi
    existing.arxiv_id = existing.arxiv_id or incoming.arxiv_id
    existing.abstract = existing.abstract or incoming.abstract
    existing.summary = existing.summary or incoming.summary
    existing.relevance_score = max(existing.relevance_score, incoming.relevance_score)
    existing.confidence = existing.confidence if existing.confidence == "high" else incoming.confidence
    existing.used_in_rounds = sorted(set(existing.used_in_rounds + incoming.used_in_rounds))
    return existing


def make_paper(
    *,
    title: str,
    authors: list[str],
    year: int | None,
    venue: str | None,
    url: str | None,
    doi: str | None,
    arxiv_id: str | None,
    abstract: str,
    source: str,
    relevance_score: float,
    confidence: str,
) -> Paper:
    return Paper(
        paper_id="P-000",
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        url=url,
        doi=doi,
        arxiv_id=arxiv_id,
        abstract=abstract,
        summary=summarize_abstract(title, abstract),
        source=source,
        retrieved_at=utc_timestamp(),
        relevance_score=relevance_score,
        confidence=confidence,
        used_in_rounds=[],
    )

