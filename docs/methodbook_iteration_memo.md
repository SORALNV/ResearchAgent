# MethodBook and Iteration Memo

ResearchAgent turns completed Kaggle experiments into durable, cross-competition methodology. The feature is attached to the existing natural-channel, Compute, Control Plane, and final-action paths; it does not introduce another execution mode.

## Runtime flow

```text
Discord Kaggle channel
  -> natural conversation and explicit experiment choice
  -> ComputeScheduler
  -> ResultFeedbackEngine
  -> IterationMemoEngine
  -> MethodCardStore
  -> relevant MethodCards injected into a later Kaggle conversation
```

Experiment completion remains authoritative. Memo generation is a post-result learning step: if it fails, the experiment result is preserved and `kaggle.iteration.memo.failed` is recorded instead of converting the experiment to failure.

## Storage

The default root is `PROJECT_ROOT/knowledge`, configurable with `METHODBOOK_DIR`.

```text
knowledge/
├── method_cards.jsonl                 # append-only source of truth
├── KAGGLE_METHODBOOK.md               # generated view
└── competitions/
    └── <competition>/
        ├── MEMO.md                    # generated view
        └── iterations/
            └── MEMO-....json          # one immutable iteration memo
```

`method_cards.jsonl` is the canonical MethodBook. Markdown files are regenerated views and must not be edited as independent sources of truth.

## Iteration memo

For every completed Kaggle experiment, the memo records:

- the exact Job and `result_ref`;
- competition, task family, modality, metric family, backend, and hypothesis;
- primary metric, baseline, direction, normalized improvement delta, and validation kind;
- outcome: `improved`, `neutral`, `regressed`, `failed`, or `inconclusive`;
- reusable lessons, anti-patterns, next quality gates, reusable assets, discard list, and one next best action;
- referenced artifacts and next-hypothesis proposals;
- MethodCards created or updated by this result.

A configured provider may generalize the evidence into a structured memo. A deterministic rule-based planner is always available as a fallback. Provider output is treated as a candidate interpretation; metric direction and evidence classification are recomputed from the stored Job and result.

## MethodCard model

A MethodCard is a conditional claim, not an unconditional recipe. Its identity is derived from the normalized claim and scope unless an existing `method_id` is explicitly reused.

```json
{
  "method_id": "KM-...",
  "claim": "CatBoost native categorical improves tabular AUC when raw categories are preserved",
  "scope": {
    "task_family": "tabular",
    "modality": "structured",
    "metric_family": "auc",
    "conditions": ["fixed CVSpec", "raw categorical columns"],
    "tags": ["catboost", "categorical"]
  },
  "status": "task_candidate",
  "confidence": "medium",
  "evidence": [],
  "counterevidence": [],
  "next_falsification": "repeat with another seed and a second competition"
}
```

Statuses are:

| Status | Meaning |
|---|---|
| `local` | One competition/run or evidence that is not independently validated |
| `task_candidate` | At least two independent CV/holdout/private-LB confirmations in the same task context |
| `verified` | Independent robust confirmations across at least two competitions |
| `deprecated` | Retained for provenance but no longer recommended |
| `rejected` | Contradicted or invalid; excluded from normal retrieval |

Public leaderboard evidence is stored but cannot promote a card. Robust counterevidence can downgrade a previously verified card for revalidation. Truncated JSONL tails are ignored so an interrupted final append does not erase earlier revisions.

## Reusing a MethodCard

Kaggle channel prompts receive only relevant, non-terminal cards, selected from the current text and recent Job scope. The Agent is told that even a verified card is not a guarantee and must design the cheapest falsification first.

When a proposal uses an existing card, it records the reference in the Job proposal:

```json
{
  "metadata": {
    "method_card_ids": ["KM-..."]
  }
}
```

After that experiment, the referenced card receives support evidence when the robust primary metric improves and counterevidence when it regresses. Neutral, failed, or inconclusive outcomes are preserved in the memo but do not automatically change the reused card's support/counter sets.

## Control Plane events

- `kaggle.iteration.memo.created`
- `kaggle.iteration.memo.failed`
- `kaggle.method.card.updated`

These events contain references and structured records, while full experiment logs and artifacts remain in their existing registries.

## Configuration

```env
METHODBOOK_ENABLED=true
METHODBOOK_DIR=knowledge
ITERATION_MEMO_PROVIDER_ENABLED=true
```

Set `METHODBOOK_ENABLED=false` to disable the learning layer without changing the existing Compute or Discord paths. With provider memo generation disabled or unavailable, deterministic memo generation remains active.

## Safety and human boundary

This feature does not submit to Kaggle, spend on paid Compute, alter credentials, or publish externally. Existing operational approvals remain in force. It also does not change the human research-direction boundary: the human still chooses the hypothesis, interprets the experiment result, and makes the final Kaggle submission decision. MethodCards are evidence-backed suggestions used by the Agent to improve later proposals, not automatic authorization to execute or submit.

The implementation adapts the iterative-learning principles discussed in paperthin, but does not install or auto-update paperthin at runtime. Reproducibility depends only on ResearchAgent source, its persisted MethodBook, and the recorded provider/runtime configuration.
