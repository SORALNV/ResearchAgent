from __future__ import annotations

import re
from typing import Any

from harness.papers import Paper

NOVELTY_BLOCKING_STATUSES = {"unclear", "crowded", "needs_human_decision"}


def generate_planning_query(goal: str) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", goal)
    if words:
        return " ".join(words[:8])
    return goal.strip()


def evaluate_novelty_gate(papers: list[Paper]) -> dict[str, Any]:
    real_papers = [paper for paper in papers if paper.source != "fake"]
    fake_papers = [paper for paper in papers if paper.source == "fake"]
    high_overlap = [
        paper
        for paper in real_papers
        if paper.relevance_score >= 0.9 and paper.confidence in {"mid", "high"}
    ]

    if not papers:
        status = "unclear"
        rationale = "関連文献が取得できていないため、新規性は未確認。"
    elif not real_papers:
        status = "needs_human_decision"
        rationale = "fake provider由来の候補のみで、実在文献に基づく新規性判断はできない。"
    elif len(real_papers) < 3:
        status = "unclear"
        rationale = "実在文献の候補が3件未満のため、類似研究の網羅性が不足している。"
    elif len(high_overlap) >= 3:
        status = "crowded"
        rationale = "高関連度の実在文献が複数あり、既存研究と近すぎる可能性が高い。"
    else:
        status = "supported"
        rationale = "実在文献に基づく比較対象があり、差分仮説を立てて進められる。"

    return {
        "novelty_status": status,
        "blocking": status in NOVELTY_BLOCKING_STATUSES,
        "rationale": rationale,
        "real_paper_count": len(real_papers),
        "fake_paper_count": len(fake_papers),
        "evidence_paper_ids": [paper.paper_id for paper in real_papers[:5]],
        "blocking_statuses": sorted(NOVELTY_BLOCKING_STATUSES),
    }


def build_planning_scout(goal: str, query: str, papers: list[Paper]) -> dict[str, Any]:
    novelty_gate = evaluate_novelty_gate(papers)
    if not papers:
        return {
            "query": query,
            **novelty_gate,
            **_build_novelty_v2_sections(goal, []),
            "similar_research": [],
            "evidence_papers": [],
            "suggested_directions": [
                "類似研究は未確認。検索語を変えて再調査するか、スコープを狭めてから再検索する。"
            ],
            "differentiation_points": [
                "未確認: 類似研究が見つかっていないため、独自性判断は保留。"
            ],
            "risks": [
                "出典候補がない状態で新規性を断定しない。"
            ],
            "questions_for_sora": [
                "この研究で最も重視したい差分は、性能・再現性・使いやすさ・コストのどれですか？"
            ],
            "decision_prompt": "類似研究が未確認のため、会話の中で追加調査してから開始判断するのがおすすめ。",
            "decision_options": [
                "検索観点を変えて追加調査する",
                "研究スコープを狭める",
                "根拠不足を理解した上で /re start するかSoraが判断する",
            ],
        }

    similar = [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "summary": paper.summary,
            "confidence": paper.confidence,
            "source": paper.source,
            "source_kind": "fake" if paper.source == "fake" else "real",
            "relevance_score": paper.relevance_score,
        }
        for paper in papers[:5]
    ]
    evidence_papers = [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "source": paper.source,
            "relevance_score": paper.relevance_score,
        }
        for paper in papers
        if paper.source != "fake"
    ][:5]
    first = evidence_papers[0] if evidence_papers else similar[0]
    second = evidence_papers[1] if len(evidence_papers) > 1 else (similar[1] if len(similar) > 1 else None)
    first_id = first["paper_id"]
    first_title = first["title"]
    directions = [
        (
            f"{first_title} [{first_id}] に近い問題設定があるため、"
            "まず再現・比較可能な最小実験を作る。"
        ),
        (
            f"{first_title} [{first_id}] の観点を踏まえ、"
            "今回の研究ではデータ条件・運用条件・評価指標のどこを変えるかを明示する。"
        ),
    ]
    if second:
        directions.append(
            f"{second['title']} [{second['paper_id']}] と比較して、既存研究との差分を1つに絞る。"
        )
    return {
        "query": query,
        **novelty_gate,
        **_build_novelty_v2_sections(goal, papers),
        "similar_research": similar,
        "evidence_papers": evidence_papers,
        "suggested_directions": directions,
        "differentiation_points": [
            "出典付きで既存研究との差分を1つ以上定義する。",
            "未確認の新規性は「要検証」として扱い、RESEARCHで検証する。",
            "評価指標を既存研究と比較できる形にする。",
        ],
        "risks": [
            "出典IDなしで「新しい」「有効」と断定しない。",
            "類似研究のabstractだけでは詳細手法が未確認の可能性がある。",
        ],
        "questions_for_sora": [
            "この類似研究群に対して、どの差分を狙いますか？",
            "最初の成果物は再現実験、比較表、実装プロトタイプ、調査レポートのどれにしますか？",
            "既存研究に勝つことより、未検証条件を明確にする方針でよいですか？",
        ],
        "decision_prompt": (
            f"提案: {first_title} [{first_id}] を主要比較対象にし、"
            "差分を1つに絞ってから /re start する。"
        ),
        "decision_options": [
            "この差分で進めるなら /re start",
            "比較対象を変えるなら会話でそのまま指示する",
            "文献根拠を増やすなら会話で調べたい観点を伝える",
        ],
    }


def _build_novelty_v2_sections(goal: str, papers: list[Paper]) -> dict[str, Any]:
    real_papers = [paper for paper in papers if paper.source != "fake"]
    candidates = real_papers or papers
    primary = candidates[0] if candidates else None
    if primary is None:
        return {
            "primary_comparison": {
                "paper_id": None,
                "title": "未確認",
                "source_kind": "none",
                "reason": "比較対象となる文献がまだ取得できていない。",
            },
            "overlap_points": [
                "未確認: 類似研究が取得できていないため、重複点は判断できない。"
            ],
            "differentiation_hypotheses": [
                "未確認: 追加文献検索後に差分仮説を立てる。"
            ],
            "weakness_points": [
                "文献根拠が不足しているため、研究開始判断が弱い。"
            ],
            "required_decisions": [
                "検索queryを変えて追加調査するか、スコープを狭めるかを決める。"
            ],
        }

    source_kind = "fake" if primary.source == "fake" else "real"
    return {
        "primary_comparison": {
            "paper_id": primary.paper_id,
            "title": primary.title,
            "source": primary.source,
            "source_kind": source_kind,
            "reason": (
                f"最も関連度が高い候補として {primary.title} [{primary.paper_id}] を主要比較対象にする。"
            ),
        },
        "overlap_points": [
            f"{primary.title} [{primary.paper_id}] と研究テーマ「{goal}」は、問題設定または研究支援フローの一部が重なる可能性がある。",
            "abstractレベルの比較であり、手法詳細の重複は未確認。",
        ],
        "differentiation_hypotheses": [
            f"{primary.title} [{primary.paper_id}] を基準に、対象ユーザー、運用制約、評価指標のいずれかを明確に変える。",
            "既存研究に勝つことより、未検証条件を出典付きで明示する。",
        ],
        "weakness_points": [
            "取得文献のabstractだけでは、実装詳細と評価条件が未確認。",
            "差分が複数に散ると研究の焦点が弱くなる。",
            "実在文献が少ない場合、新規性判断は保留すべき。",
        ],
        "required_decisions": [
            "主要比較対象をこの文献に固定してよいか。",
            "差分を性能、再現性、運用コスト、UX、評価方法のどれに置くか。",
            "最初の成果物を比較表、再現実験、プロトタイプ、調査レポートのどれにするか。",
        ],
    }


def render_planning_scout(scout: dict[str, Any]) -> str:
    lines = ["## Similar Research Scout", "", f"- Query: {scout.get('query') or '未設定'}"]
    lines.append(f"- Novelty Status: {scout.get('novelty_status') or '未評価'}")
    lines.append(f"- Gate: {'BLOCKED' if scout.get('blocking') else 'PASS'}")
    lines.append(f"- Rationale: {scout.get('rationale') or '未確認'}")
    lines.append(
        f"- Evidence Papers: real={scout.get('real_paper_count', 0)} / fake={scout.get('fake_paper_count', 0)}"
    )
    similar = scout.get("similar_research") or []
    lines.append("")
    lines.append("### Similar Research")
    if similar:
        for item in similar:
            source_kind = item.get("source_kind") or ("fake" if item.get("source") == "fake" else "real")
            lines.append(
                f"- [{item['paper_id']}] {item['title']} ({item.get('year') or '未確認'}, {source_kind}): {item.get('summary') or '要約なし'}"
            )
    else:
        lines.append("- 未確認")
    lines.append("")
    lines.append("### Primary Comparison")
    primary = scout.get("primary_comparison") or {}
    lines.append(
        f"- [{primary.get('paper_id') or '未確認'}] {primary.get('title') or '未確認'} "
        f"({primary.get('source_kind') or 'unknown'}): {primary.get('reason') or '未確認'}"
    )
    for title, key in [
        ("Overlap Points", "overlap_points"),
        ("Differentiation Hypotheses", "differentiation_hypotheses"),
        ("Weakness Points", "weakness_points"),
        ("Required Decisions", "required_decisions"),
        ("Suggested Directions", "suggested_directions"),
        ("Differentiation Points", "differentiation_points"),
        ("Risks", "risks"),
        ("Questions For Sora", "questions_for_sora"),
    ]:
        lines.append("")
        lines.append(f"### {title}")
        for item in scout.get(key, []) or ["未確認"]:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("### Decision Prompt")
    lines.append(scout.get("decision_prompt") or "未確認")
    lines.append("")
    lines.append("### Decision Options")
    for item in scout.get("decision_options", []) or ["未確認"]:
        lines.append(f"- {item}")
    return "\n".join(lines)
