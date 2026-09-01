# ResearchAgent Harness

ResearchAgent is a persistent control plane for running research sessions from Discord or a local CLI. It keeps the research goal, constraints, literature evidence, decisions, approvals, execution traces, and final artifacts in versioned research folders.

The system is designed for **orchestration rather than single-LLM completion**. When real agent commands are configured, a research round uses a real multi-agent pipeline:

```text
main plan
  ↓
multiple sub agents in parallel
  ↓
review
  ↓
selective sub retry when needed
  ↓
fresh independent view
  ↓
main integration
```

If no agent command is configured, the deterministic `MockAgentRunner` remains available for local demos and regression tests.

## Discordチャンネル単位の運用

現在のDiscord Edgeでは、**1チャンネルを1件のKaggleコンペまたは研究テーマ**として扱います。`/agent setup`で`kaggle`または`research`、案件名、対象を登録すると、Project・WorkSession・Codex threadが永続的に紐付きます。会話中に「試して」「実装して」「この案で進めて」と指示すれば実装からJob実行へ進み、「このCSVで提出しよう」「この結果を論文にまとめて」で既存の安全ゲート付き最終処理へ進みます。Discord上に戦略モード／実行モードの切替はありません。

案件終了時は`/agent finish`で内部状態をアーカイブし、Discordチャンネル自体はユーザーが整理します。詳細は[`docs/natural_channel_workflow.md`](docs/natural_channel_workflow.md)を参照してください。

## Design Philosophy

ResearchAgent should behave like a large, persistent `AGENTS.md` plus an execution journal. The harness records and routes work instead of pretending one model can simultaneously be the researcher, executor, reviewer, and final judge.

Planning dialogue clarifies intent and can trigger literature search. Research execution is split across roles such as `main`, `sub`, `review`, `fresh`, and optional Claude consultation. Dangerous operations remain behind the existing approval gate.

## What It Includes

- Discord slash-command entry point and local CLI
- `PLANNING`, `RESEARCH`, `APPROVAL_BLOCKED`, `PAUSED`, and `DONE` modes
- normal Discord messages for PLANNING dialogue
- versioned research folders under `research_runs/V001.0_{session_id}_{slug}/`
- durable `research_brief.md`, `journal.jsonl`, `papers.jsonl`, `research_ledger.jsonl`, `state.json`, reports, and artifacts
- real multi-agent execution when any agent command is configured
- multiple parallel sub agents with isolated per-task workspaces
- reviewer-driven selective retry
- fresh independent perspective and final main integration
- optional Claude consultation
- arXiv literature search and citation IDs
- novelty / phase gates
- dangerous-operation approval gate
- cost counters and hard limits
- golden-question evaluation
- Discord command worker so long-running research does not block the Discord event loop
- GitHub Actions CI for Python 3.11 and 3.12

## Setup

```bash
cd /home/jetson/Code/ResearchAgent
python -m venv .venv
. .venv/bin/activate
pip install -e .[test]
cp .env.example .env
```

For Discord:

```bash
pip install -e .[discord]
```

Set at least:

```env
DISCORD_BOT_TOKEN=...
DISCORD_IMPORTANT_CHANNEL_ID=...
DISCORD_LOG_CHANNEL_ID=...
```

For real Codex-backed research roles:

```env
MAIN_AGENT_COMMAND=codex
SUB_AGENT_COMMAND=codex
REVIEW_AGENT_COMMAND=codex
FRESH_AGENT_COMMAND=codex
```

A missing role falls back to another configured command. Therefore even `SUB_AGENT_COMMAND=codex` alone is enough to switch research rounds from the deterministic mock path to real role-specific Codex calls.

Recommended execution settings:

```env
SUB_AGENT_COUNT=3
AGENT_PARALLELISM=3
MAX_REVIEW_RETRIES=1
FRESH_INTERVAL=1
MAX_COMMAND_SECONDS=300
DISCORD_WORKER_QUEUE_SIZE=32
```

Use finite `MAX_AGENT_CALLS` and `MAX_COMMAND_SECONDS` for paid or unattended operation.

## Local Demo

Without real agent command variables, the local demo intentionally uses deterministic mock execution:

```bash
python main.py --workdir /tmp/research-agent-demo demo --goal "研究ハーネスを検証する"
```

Expected durable outputs include:

```text
/tmp/research-agent-demo/
├── state.json
└── research_runs/
    └── V001.0_{session_id}_{slug}/
        ├── state.json
        ├── journal.jsonl
        ├── research_brief.md
        ├── papers.jsonl
        ├── research_ledger.jsonl
        ├── run_summary.md
        └── artifacts/
            └── report.md
```

The root `state.json` is only the active-session pointer. The durable research record lives in the versioned research folder.

## CLI Commands

```bash
python main.py --workdir /tmp/research-agent-demo re new
python main.py --workdir /tmp/research-agent-demo re plan
python main.py --workdir /tmp/research-agent-demo re start
python main.py --workdir /tmp/research-agent-demo re status
python main.py --workdir /tmp/research-agent-demo re pause
python main.py --workdir /tmp/research-agent-demo re resume
python main.py --workdir /tmp/research-agent-demo re redirect "制約や方針変更"
python main.py --workdir /tmp/research-agent-demo re idea "追加アイデア"
python main.py --workdir /tmp/research-agent-demo re search "research agent citation grounding"
python main.py --workdir /tmp/research-agent-demo re papers
python main.py --workdir /tmp/research-agent-demo re paper P-001
python main.py --workdir /tmp/research-agent-demo re eval
python main.py --workdir /tmp/research-agent-demo re cost
python main.py --workdir /tmp/research-agent-demo re doctor
python main.py --workdir /tmp/research-agent-demo re runs
python main.py --workdir /tmp/research-agent-demo re accept PG-1
python main.py --workdir /tmp/research-agent-demo re revise PG-1 "比較対象を変える"
python main.py --workdir /tmp/research-agent-demo re approve AP-1
python main.py --workdir /tmp/research-agent-demo re reject AP-1 "理由"
python main.py --workdir /tmp/research-agent-demo re stop
```

The internal similar-research scout is triggered from PLANNING dialogue or explicit paper search logic; `/re scout` is not a public command.

## Discord Bot

Start the real bot with:

```bash
python main.py --workdir ./runtime bot
```

Supported slash commands include:

- `/re new`
- `/re plan`
- `/re start`
- `/re status`
- `/re pause`
- `/re resume`
- `/re search <query>`
- `/re papers`
- `/re paper <paper_id>`
- `/re eval`
- `/re cost`
- `/re doctor`
- `/re runs`
- `/re accept <phase_gate_id>`
- `/re revise <phase_gate_id> <reason>`
- `/re approve <approval_id>`
- `/re reject <approval_id> <reason>`
- `/re stop`

After `/re plan`, ordinary messages in the important channel are treated as PLANNING dialogue until the mode changes.

### Discord Worker

Discord no longer calls the synchronous orchestrator directly on the event-loop thread. Commands go through a bounded `AsyncCommandWorker`:

```text
Discord interaction/message
        ↓
AsyncCommandWorker queue
        ↓
asyncio.to_thread(orchestrator.handle)
        ↓
serialized state/journal mutation
```

There is intentionally one command consumer. Long-running agent subprocesses therefore run off the Discord event loop, while state transitions, journal writes, ledger writes, and report generation remain ordered. If the queue reaches `DISCORD_WORKER_QUEUE_SIZE`, the user receives an explicit busy response.

## Mode Flow

Recommended flow:

```text
/re new
/re plan
normal Discord PLANNING dialogue
optional literature search / novelty gate
/re start
RESEARCH
approval or phase gate if required
DONE
/re stop
```

`/re new` closes the prior active theme when needed and creates a new versioned research folder. `/re plan` enables dialogue. `/re start` enters `RESEARCH` only when blocking phase gates are resolved.

## Real Multi-Agent Execution

When at least one agent command is configured, every research round uses real agents.

### 1. main plan

The main role receives the research goal and current question and returns structured independent subtasks.

### 2. parallel sub execution

Up to `SUB_AGENT_COUNT` tasks run concurrently, bounded by `AGENT_PARALLELISM`. Each sub task receives its own durable writable workspace:

```text
artifacts/agent_workspaces/
└── R001/
    ├── S1/
    │   ├── attempt-01/
    │   └── attempt-02/
    └── S2/
        └── attempt-01/
```

This prevents parallel agents from overwriting the same files and preserves every attempt for audit.

### 3. review and selective retry

The review role checks evidence, reproducibility, contradictions, failures, and unsafe operations. A structured review can request specific tasks to run again. Only those tasks are retried, up to `MAX_REVIEW_RETRIES`.

### 4. fresh

According to `FRESH_INTERVAL`, a fresh role adds a deliberately independent hypothesis, counterexample, missing comparison axis, or simpler alternative.

### 5. main integration

The main role integrates the latest sub results, reviews, fresh output, and optional Claude consultation. Failed or unverified results remain visible rather than being silently converted into success.

More implementation detail is in `docs/multi_agent_execution.md`.

## Literature Search and Novelty Gate

`PAPER_PROVIDER=arxiv` uses the arXiv API. Retrieved papers are stored in `papers.jsonl` with stable IDs such as `[P-001]`.

Planning can use these papers to propose:

- a primary comparison
- overlap points
- differentiation hypotheses
- weakness points
- required decisions
- risks
- a `Novelty Status`: `supported`, `unclear`, `crowded`, or `needs_human_decision`

Unclear or crowded novelty can block `/re start` through a Phase Gate. Fake-provider results are useful for tests but are not treated as real evidence.

## Approval Gate Policy

Any real agent can emit:

```text
APPROVAL_REQUIRED: operation=<operation>; reason=<reason>; impact=<impact>; dry_run_result=<not executed>
```

ResearchAgent scans outputs from main, every sub, review, fresh, and optional consultation. Dangerous operations are converted into an `AP-*` request and the session enters `APPROVAL_BLOCKED`.

Approval is required for operations such as:

- file or folder deletion
- important-file overwrite
- writing outside the research archive
- `git push`, PR creation, release actions
- external posting
- paid API use
- secret or `.env` transfer
- `sudo`, `chmod`, `chown`
- untrusted network access

Long-running or large-generation operations can instead emit an important notice.

Approval records permission state and resumes the harness; it does **not** automatically execute the dangerous command. This behavior is intentionally unchanged.

## Research Ledger and Reports

`journal.jsonl` is the full append-only audit stream. `research_ledger.jsonl` summarizes completed research rounds with fields such as:

- `node_id`
- `parent_node_id`
- `phase`
- `hypothesis`
- `action`
- `result`
- `feedback`
- `failure_class`
- `next_action`
- `evidence_ids`
- `selected_as_best`

On `/re stop`, the harness creates `artifacts/report.md` and `run_summary.md`. The report includes sources, novelty status, ledger summary, unverified claims, risks, next steps, and AI provenance.

## Cost Limits

Existing limits remain available:

- `MAX_ROUNDS`
- `MAX_API_CALLS`
- `MAX_TOTAL_TOKENS`
- `MAX_AGENT_CALLS`
- `MAX_COMMAND_SECONDS`
- `MAX_TURNS_PER_CONVERSATION`
- `CONVERSATION_TIMEOUT_SECONDS`
- `CONVERGENCE_PATIENCE`
- `REPORT_INTERVAL_SECONDS`

`MAX_AGENT_CALLS=0`, `MAX_API_CALLS=0`, and `MAX_TOTAL_TOKENS=0` mean no hard limit.

## Doctor

Use:

```bash
python main.py --workdir ./runtime re doctor
```

It reports the project root, research archive directory, Discord channel configuration, paper provider, Codex CLI availability, and configured sub-agent command.

## Evaluation

Golden questions live in:

```text
eval/golden_questions.jsonl
```

Run:

```bash
python main.py --workdir ./runtime re eval
```

## Tests

```bash
python -m compileall -q harness main.py
pytest -q
```

GitHub Actions runs the full test suite on Python 3.11 and Python 3.12 for pull requests. The suite includes the original harness tests plus real subprocess concurrency, reviewer-driven retry, role fallback, approval propagation, workspace isolation, worker queue behavior, and Discord event-loop responsiveness.
