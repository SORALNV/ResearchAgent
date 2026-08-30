# Runtime safety, quality evaluation, and convergence

## Active execution path

`MockAgentRunner` remains the compatibility facade, but when any real-agent command is configured it instantiates `harness.multi_agent_runner.MultiAgentRunner`. That class is the hardened runtime and is covered by `tests/test_hardened_runtime_wiring.py`.

The active real-agent path is:

```text
main plan
  -> parallel isolated sub workspaces
  -> fail-closed structured review
  -> reviewer-selected retries
  -> optional fresh / Claude audit
  -> structured main integration
  -> hash-verified artifact promotion
```

A malformed, timed-out, skipped, or non-zero plan/review/integration call is never accepted implicitly. The runner retries protocol generation within `MAX_PROTOCOL_RETRIES`; unresolved failures are converted into approval-blocking operations.

## Checkpoint and recovery

Every stage writes an atomic checkpoint under:

```text
artifacts/checkpoints/Rxxx.json
```

Completed plan, subtask, review, fresh, and consultation calls are reused. An interrupted or protocol-blocked round resumes from the first incomplete or invalid stage instead of repeating all paid calls.

## Process and secret isolation

Agent subprocesses receive an allowlisted environment, not the Discord Bot environment. `DISCORD_*` variables are hard-denied. The default isolated HOME is created under the research run.

A bare `codex` command uses Codex's own sandbox settings. Generic commands use `AGENT_SANDBOX_BACKEND`:

- `auto`: use bubblewrap when available, otherwise reject generic execution.
- `bwrap`: require bubblewrap.
- `none`: no OS wrapper; rejected unless `AGENT_ALLOW_UNSANDBOXED_GENERIC=true`.

For bubblewrap, project/research inputs are read-only, the assigned task workspace and isolated HOME are writable, and network access is denied by default.

## Artifact promotion

Sub-agent files are hashed into `artifact_manifest.json`. Main integration may select artifacts by `task_id` and relative path. Promotion verifies:

- regular-file status
- no symlink or path traversal
- manifest membership
- SHA-256 and size
- no conflicting destination overwrite

Only verified files are copied to `artifacts/final/Rxxx/`.

## Runtime status

`/re status` uses a read-only control lane and remains responsive while the serialized research worker is busy. It displays:

- current stage and checkpoint state
- active Agent process count
- completed/failed/total subtasks
- Agent calls and estimated tokens
- convergence counters
- pending approvals and latest runtime error

`/re cancel`, `/re pause`, and `/re stop` signal active process groups before entering the serialized mutation queue.

## Convergence

Main integration returns:

```json
{
  "round_status": "continue | completed | blocked | failed",
  "progress_score": 0.0,
  "new_evidence_ids": [],
  "unresolved_blockers": []
}
```

`ConvergenceTracker` completes a high-confidence `completed` result early. It opens a human phase gate when progress stagnates or no evidence is added for the configured patience. State is stored in `artifacts/convergence.json`.

## Evaluation

`/re eval` now evaluates generated research content, not only paper metadata. It checks:

- expected-answer keyword coverage
- prohibited claims
- source-type coverage
- citation IDs against `papers.jsonl`
- Agent invocation success rate
- structured review success
- promoted-artifact integrity

The score is diagnostic, not a proof that a research conclusion is correct. Golden questions should be expanded for each intended research domain.
