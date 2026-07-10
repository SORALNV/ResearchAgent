from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import re
from typing import Protocol

from harness.papers import Paper, make_paper


class PaperSearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[Paper]:
        ...


@dataclass
class FakePaperSearchProvider:
    name: str = "fake"

    def search(self, query: str, max_results: int = 5) -> list[Paper]:
        fixtures = [
            make_paper(
                title="Research Harnesses for Tool-Using Agents",
                authors=["A. Example", "B. Tester"],
                year=2025,
                venue="FakeConf",
                url="https://example.test/papers/research-harnesses",
                doi="10.0000/example-harness",
                arxiv_id=None,
                abstract=(
                    "This paper studies lightweight harnesses for tool-using research agents, "
                    "with emphasis on restartability, audit logs, and human approval gates."
                ),
                source=self.name,
                relevance_score=0.92,
                confidence="mid",
            ),
            make_paper(
                title="Citation Grounding for Automated Literature Review",
                authors=["C. Source"],
                year=2024,
                venue="FakeWorkshop",
                url="https://example.test/papers/citation-grounding",
                doi="10.0000/example-citation",
                arxiv_id=None,
                abstract=(
                    "The work evaluates citation identifiers for reducing unsupported claims "
                    "in automated literature review systems."
                ),
                source=self.name,
                relevance_score=0.86,
                confidence="mid",
            ),
        ]
        lowered = query.lower()
        ranked = [
            paper
            for paper in fixtures
            if any(token in paper.title.lower() or token in paper.abstract.lower() for token in lowered.split())
        ] or fixtures
        return ranked[:max_results]


@dataclass
class ArxivPaperSearchProvider:
    name: str = "arxiv"
    base_url: str = "https://export.arxiv.org/api/query"
    timeout_seconds: int = 15

    def search(self, query: str, max_results: int = 5) -> list[Paper]:
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        request = urllib.request.Request(
            f"{self.base_url}?{params}",
            headers={"User-Agent": "ResearchAgentHarness/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            xml_body = response.read()
        return _rank_and_filter_by_query(self._parse(xml_body, max_results), query, max_results)

    def _parse(self, xml_body: bytes, max_results: int) -> list[Paper]:
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(xml_body)
        papers: list[Paper] = []
        for index, entry in enumerate(root.findall("atom:entry", ns), start=1):
            title = _text(entry, "atom:title", ns)
            abstract = _text(entry, "atom:summary", ns)
            url = _text(entry, "atom:id", ns) or None
            arxiv_id = url.rstrip("/").split("/")[-1] if url else None
            published = _text(entry, "atom:published", ns)
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [
                _text(author, "atom:name", ns)
                for author in entry.findall("atom:author", ns)
                if _text(author, "atom:name", ns)
            ]
            doi = _text(entry, "arxiv:doi", ns) or None
            papers.append(
                make_paper(
                    title=title,
                    authors=authors,
                    year=year,
                    venue="arXiv",
                    url=url,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    abstract=abstract,
                    source=self.name,
                    relevance_score=max(0.0, 1.0 - (index - 1) * 0.05),
                    confidence="mid",
                )
            )
            if len(papers) >= max_results:
                break
        return papers


def _text(element: ET.Element, path: str, ns: dict[str, str]) -> str:
    found = element.find(path, ns)
    if found is None or found.text is None:
        return ""
    return " ".join(found.text.split())


_QUERY_STOPWORDS = {
    "about",
    "docs",
    "documentation",
    "natural",
    "language",
    "official",
    "paper",
    "papers",
    "possible",
    "research",
    "search",
    "study",
    "using",
    "with",
}


def _rank_and_filter_by_query(papers: list[Paper], query: str, max_results: int) -> list[Paper]:
    significant_terms = _significant_query_terms(query)
    if not significant_terms:
        return papers[:max_results]

    ranked: list[tuple[float, Paper]] = []
    for paper in papers:
        score = _lexical_relevance_score(paper, significant_terms)
        if score <= 0:
            continue
        paper.relevance_score = score
        ranked.append((score, paper))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [paper for _, paper in ranked[:max_results]]


def _significant_query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in _tokens(query):
        if len(token) < 4 or token in _QUERY_STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _lexical_relevance_score(paper: Paper, significant_terms: list[str]) -> float:
    haystack = set(
        _tokens(
            " ".join(
                [
                    paper.title,
                    paper.abstract,
                    paper.venue or "",
                    paper.arxiv_id or "",
                    paper.doi or "",
                ]
            )
        )
    )
    matched = sum(1 for term in significant_terms if term in haystack)
    if len(significant_terms) >= 3 and matched < 2:
        return 0.0
    if matched == 0:
        return 0.0
    return matched / len(significant_terms)


def _tokens(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", text.lower()):
        tokens.append(token)
        if token.endswith("s") and len(token) > 4:
            tokens.append(token[:-1])
    return tokens
