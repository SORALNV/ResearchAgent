# Real multi-agent execution and Discord worker

## Research round pipeline

When any real agent command is configured, the compatibility `MockAgentRunner` delegates to `MultiAgentRunner`.

Each round executes:

1. **main / plan** — create a structured plan and independent subtasks.
2. **sub / execute** — run up to `SUB_AGENT_COUNT` subtasks concurrently, bounded by `AGENT_PARALLELISM`.
3. **review** — verify evidence, contradictions, reproducibility, failures, and unsafe operations.
4. **sub / retry** — rerun only reviewer-selected tasks, up to `MAX_REVIEW_RETRIES`.
5. **fresh** — add an independent hypothesis, counterexample, or missing comparison axis according to `FRESH_INTERVAL`.
6. **main / integrate** — integrate the latest sub outputs, review, fresh output, and optional Claude consultation.

If a role-specific command is missing, that role falls back to another configured command. Therefore `SUB_AGENT_COMMAND=codex` alone is enough to make the whole round real rather than leaving main/review/fresh as stubs.

A bare `codex` command is expanded without a shell. Read-only roles use a read-only sandbox; sub agents use `workspace-write`.

## Parallel workspace isolation

Each sub task and retry receives its own durable writable directory:

```text
research_runs/Vxxx.../artifacts/agent_workspaces/
└── R001/
    ├── S1/
    │   ├── attempt-01/
    │   └── attempt-02/
    └── S2/
        └── attempt-01/
```

The shared research directory is reference context. Sub agents are instructed not to write into another sub agent's workspace. This prevents concurrent file overwrites and preserves every attempt for audit.

## Review contract

Reviewers should return JSON like:

```json
{
  "verdict": "revise",
  "summary": "S1 needs a direct check",
  "revisions": [
    {"task_id": "S1", "instructions": "add a direct verification"}
  ],
  "confidence": "high"
}
```

Only listed tasks are retried. If revision is requested without valid task IDs, the runner conservatively retries all subtasks rather than silently accepting an unusable review.

## Approval and important-notice propagation

Any real agent can emit:

```text
APPROVAL_REQUIRED: operation=<operation>; reason=<reason>; impact=<impact>; dry_run_result=<not executed>
```

or:

```text
IMPORTANT_NOTICE: operation=long_running_command:<operation>; reason=<reason>; impact=<impact>; dry_run_result=<not executed>
```

The runner scans every role output. Approval-required operations take precedence and are passed to the existing approval gate. Dangerous commands are not executed automatically after approval; existing harness policy remains unchanged.

## Discord worker

The real Discord entry point uses `AsyncCommandWorker` rather than calling the synchronous orchestrator on the Discord event-loop thread:

```text
Discord interaction/message
        ↓
AsyncCommandWorker bounded queue
        ↓
asyncio.to_thread(orchestrator.handle)
        ↓
serialized state/journal mutation
```

There is intentionally one queue consumer. Long-running agent work is off the event loop, but research state, journal, ledger, and report mutations stay serialized. `DISCORD_WORKER_QUEUE_SIZE` bounds queued commands; when full, Discord returns an explicit busy response.

Outbound Discord messages can originate from the worker thread, so the worker-backed adapter schedules sends onto the Discord loop with `call_soon_threadsafe`.

## Recommended configuration

```env
MAIN_AGENT_COMMAND=codex
SUB_AGENT_COMMAND=codex
REVIEW_AGENT_COMMAND=codex
FRESH_AGENT_COMMAND=codex
SUB_AGENT_COUNT=3
AGENT_PARALLELISM=3
MAX_REVIEW_RETRIES=1
FRESH_INTERVAL=1
DISCORD_WORKER_QUEUE_SIZE=32
```

For paid or unattended operation, set finite `MAX_AGENT_CALLS` and `MAX_COMMAND_SECONDS`. A round can make several agent calls when review requests retries.

## Verification

```bash
python -m compileall -q harness main.py
pytest -q
```

The added tests cover subprocess concurrency, selective review retry, role fallback, approval propagation, workspace isolation, worker serialization, bounded queue behavior, and event-loop responsiveness.
