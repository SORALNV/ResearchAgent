from pathlib import Path

import pytest

from harness.kaggle_methodbook import (
    EvidenceKind,
    MethodCandidate,
    MethodCardStatus,
    MethodCardStore,
    MethodEvidence,
    MethodScope,
    ValidationKind,
)


def _candidate(kind: EvidenceKind = EvidenceKind.SUPPORT) -> MethodCandidate:
    return MethodCandidate(
        claim="CatBoost native categorical improves tabular CV when raw categories are preserved",
        scope=MethodScope(
            task_family="tabular",
            modality="structured",
            metric_family="auc",
            conditions=("fixed CVSpec", "raw categorical columns"),
            tags=("catboost", "categorical"),
        ),
        evidence_kind=kind,
        next_falsification="repeat with another seed and a second competition",
    )


def _evidence(
    ref: str,
    competition: str,
    *,
    kind: ValidationKind = ValidationKind.CV,
    delta: float = 0.001,
) -> MethodEvidence:
    return MethodEvidence(
        result_ref=ref,
        competition=competition,
        memo_id="MEMO-" + ref.rsplit(":", 1)[-1],
        outcome="improved" if delta >= 0 else "regressed",
        validation_kind=kind,
        metric_name="cv_auc",
        metric_value=0.84 + delta,
        metric_delta=delta,
        independent_key=ref,
    )


def test_method_card_promotes_only_after_independent_evidence(tmp_path: Path) -> None:
    store = MethodCardStore(tmp_path / "knowledge")
    first = store.record(_candidate(), _evidence("result:a:1", "comp-a"))
    assert first.status == MethodCardStatus.LOCAL

    duplicate = store.record(_candidate(), _evidence("result:a:1", "comp-a"))
    assert duplicate.revision == first.revision

    second = store.record(_candidate(), _evidence("result:a:2", "comp-a"))
    assert second.status == MethodCardStatus.TASK_CANDIDATE

    third = store.record(_candidate(), _evidence("result:b:1", "comp-b"))
    assert third.status == MethodCardStatus.VERIFIED
    assert len(third.evidence) == 3
    assert store.markdown_path.is_file()
    rendered = store.markdown_path.read_text(encoding="utf-8")
    assert "method_cards.jsonl" in rendered
    assert third.method_id in rendered


def test_public_leaderboard_evidence_never_promotes_a_card(tmp_path: Path) -> None:
    store = MethodCardStore(tmp_path / "knowledge")
    card = store.record(
        _candidate(),
        _evidence(
            "result:lb:1",
            "comp-a",
            kind=ValidationKind.PUBLIC_LB,
        ),
    )
    card = store.record(
        _candidate(),
        _evidence(
            "result:lb:2",
            "comp-b",
            kind=ValidationKind.PUBLIC_LB,
        ),
    )
    assert card.status == MethodCardStatus.LOCAL


def test_counterevidence_downgrades_verified_claim_for_revalidation(tmp_path: Path) -> None:
    store = MethodCardStore(tmp_path / "knowledge")
    card = store.record(_candidate(), _evidence("result:a:1", "comp-a"))
    card = store.record(_candidate(), _evidence("result:b:1", "comp-b"))
    assert card.status == MethodCardStatus.VERIFIED

    card = store.record(
        _candidate(EvidenceKind.COUNTER),
        _evidence("result:c:1", "comp-c", delta=-0.003),
    )
    assert card.status == MethodCardStatus.TASK_CANDIDATE
    assert len(card.counterevidence) == 1


def test_search_excludes_terminal_cards_and_uses_scope(tmp_path: Path) -> None:
    store = MethodCardStore(tmp_path / "knowledge")
    card = store.record(_candidate(), _evidence("result:a:1", "comp-a"))
    assert store.search("CatBoost categorical", task_family="tabular") == (card,)

    rejected = store.transition(
        card.method_id,
        MethodCardStatus.REJECTED,
        reason="independent holdout contradicted the claim",
    )
    assert rejected.status == MethodCardStatus.REJECTED
    assert store.search("CatBoost categorical", task_family="tabular") == ()
    assert store.search(
        "CatBoost categorical",
        task_family="tabular",
        include_inactive=True,
    ) == (rejected,)
    with pytest.raises(ValueError):
        store.transition(
            card.method_id,
            MethodCardStatus.VERIFIED,
            reason="terminal records do not reopen",
        )


def test_append_only_store_ignores_a_truncated_tail(tmp_path: Path) -> None:
    store = MethodCardStore(tmp_path / "knowledge")
    card = store.record(_candidate(), _evidence("result:a:1", "comp-a"))
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version":1,"card":')
    reloaded = MethodCardStore(store.root)
    assert reloaded.get(card.method_id) == card
