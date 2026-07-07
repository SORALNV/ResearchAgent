# ResearchAgent Harness MVP

ResearchAgent is a small control layer for running research sessions from Discord or a local CLI. The MVP intentionally does not run real Claude, Codex, or sub-agent processes. It uses `MockAgentRunner` to prove the end-to-end harness loop first.

## Design Philosophy

This harness should not try to solve most of the research problem by itself. Treat it as a large, persistent `AGENTS.md` plus an execution journal: it keeps the goal, constraints, source evidence, decisions, approvals, and next actions in one durable place.

The default design is orchestration, not single-LLM completion. Planning dialogue is used to clarify intent and decide which tools or agents to call. Literature search is a tool that the LLM may call when useful. Research execution should be split across roles such as main, sub, review, and fresh so that the system can propose, execute, critique, and add alternative views without depending on one model's answer as final truth.

The harness records and routes work. It should avoid pretending to be the researcher, the reviewer, the executor, and the judge all at once.

## What This MVP Includes

- Discord slash-command entry point with an optional `discord.py` adapter
- Fake Discord adapter for local tests and demos
- `PLANNING`, `RESEARCH`, `APPROVAL_BLOCKED`, `PAUSED`, and `DONE` modes
- `/re new -> PLANNING -> /re start -> RESEARCH -> /re stop` flow
- per-research versioned folders under `research_runs/V001.0_{session_id}_{slug}/`
- per-research `research_brief.md`, `journal.jsonl`, `papers.jsonl`, `state.json`, and `artifacts/`
- deterministic `MockAgentRunner` roles for main, sub, review, fresh, and Claude-stub consultation
- bounded conversation sessions with `max_turns`, timeout, and stagnation stopping
- approval gate dry-run for dangerous operations
- optional real `sub` agent through `SUB_AGENT_COMMAND=codex`
- Phase A literature search with `papers.jsonl`
- citation IDs for paper-backed notes
- session cost counters and hard-limit blocking
- simple golden-question evaluation

## Setup

```bash
cd /home/jetson/Code/ResearchAgent
python -m venv .venv
. .venv/bin/activate
pip install -e .[test]
```

For the real Discord bot entry point:

```bash
pip install -e .[discord]
cp .env.example .env
```

Fill in `DISCORD_BOT_TOKEN` and `DISCORD_CHANNEL_ID` in `.env` or export them in your shell.
For channel separation, set `DISCORD_IMPORTANT_CHANNEL_ID` for high-priority reports and approval requests, and `DISCORD_LOG_CHANNEL_ID` for detailed event logs. `DISCORD_CHANNEL_ID` remains as the legacy fallback for the important channel.

## Local Demo

The local demo uses the Fake Discord adapter and writes artifacts under the chosen workdir.

```bash
python main.py --workdir /tmp/research-agent-demo demo --goal "研究ハーネスMVPを検証する"
```

Expected outputs:

- `/tmp/research-agent-demo/research_runs/V001.0_{session_id}_{slug}/research_brief.md`
- `/tmp/research-agent-demo/research_runs/V001.0_{session_id}_{slug}/journal.jsonl`
- `/tmp/research-agent-demo/research_runs/V001.0_{session_id}_{slug}/papers.jsonl`
- `/tmp/research-agent-demo/research_runs/V001.0_{session_id}_{slug}/state.json`
- `/tmp/research-agent-demo/research_runs/V001.0_{session_id}_{slug}/artifacts/`
- `/tmp/research-agent-demo/state.json`

The root `state.json` is only an active-session pointer. The durable research record lives in the versioned research folder.

The demo runs:

```text
/re new -> /re plan -> /re search -> /re papers -> /re cost -> /re start -> approval gate -> /re approve AP-1 -> /re eval -> /re stop
```

To choose a different archive location:

```bash
python main.py \
  --workdir /tmp/research-agent-runtime \
  --research-archive-dir /data/research-archive \
  re new
python main.py \
  --workdir /tmp/research-agent-runtime \
  --research-archive-dir /data/research-archive \
  re plan
```

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
python main.py --workdir /tmp/research-agent-demo re scout
python main.py --workdir /tmp/research-agent-demo re scout "custom search query"
python main.py --workdir /tmp/research-agent-demo re search "research agent harness citation"
python main.py --workdir /tmp/research-agent-demo re papers
python main.py --workdir /tmp/research-agent-demo re paper P-001
python main.py --workdir /tmp/research-agent-demo re eval
python main.py --workdir /tmp/research-agent-demo re cost
python main.py --workdir /tmp/research-agent-demo re doctor
python main.py --workdir /tmp/research-agent-demo re runs
python main.py --workdir /tmp/research-agent-demo re approve AP-1
python main.py --workdir /tmp/research-agent-demo re reject AP-1 "理由"
python main.py --workdir /tmp/research-agent-demo re stop
```

The older direct commands such as `goal` and `start` remain as developer shortcuts, but the user-facing path is the `re` command group.

## Discord Bot

The real bot is intentionally thin. It converts slash commands into the same `Command` objects used by the CLI, then calls the synchronous orchestrator.

```bash
export DISCORD_BOT_TOKEN="..."
export DISCORD_IMPORTANT_CHANNEL_ID="..."
export DISCORD_LOG_CHANNEL_ID="..."
python main.py --workdir ./runtime bot
```

## Discord Channels

The harness can split Discord output into two channels:

- Important channel: periodic reports, approval requests, cost-limit stops, and user-facing command results.
- Log channel: compact event stream for nearly every `journal.jsonl` event.

Configure them with:

```env
DISCORD_IMPORTANT_CHANNEL_ID=123456789012345678
DISCORD_LOG_CHANNEL_ID=234567890123456789
```

If `DISCORD_LOG_CHANNEL_ID` is empty, detailed logs still go to `journal.jsonl` but are not posted to Discord. If `DISCORD_IMPORTANT_CHANNEL_ID` is empty, `DISCORD_CHANNEL_ID` is used as a fallback.

Supported slash commands:

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

## Mode Flow

Recommended Discord flow:

```text
/re new
/re plan
normal Discord messages for the research idea and PLANNING dialogue
/re start
```

`/re new` finishes the current active theme if needed, generates its closing artifacts, and creates a fresh versioned research folder with the goal still unset. `/re plan` switches into plan mode. After that, normal messages in the important channel are treated as PLANNING dialogue turns until `/re new`, `/re stop`, or `/re start` changes the mode. The first normal message can be the new research idea.

During dialogue mode, the main planning agent should behave like an orchestrator-facing conversation partner. It may decide to use the paper-search tool, propose sub/review/fresh consultation, or ask Sora to clarify constraints. The harness records the dialogue, evidence, decisions, and next actions; it should not treat one LLM response as the final research answer.

The Discord bot presence shows the current user-facing mode:

- `mode: Neutral`: no active planning conversation, finished session, or waiting for `/re plan`
- `mode: plan`: ordinary messages are treated as PLANNING dialogue
- `mode: researching`: research rounds are running or ready to continue
- `mode: blocked`: approval, cost, phase gate, or safety gate is waiting for Sora
- `mode: finalizing`: stop/summary/report cleanup is in progress

After `/re plan`, ordinary Discord messages in the important channel continue the PLANNING wall-ball conversation. On each turn, the LLM may decide to call the paper-search tool and must generate the search query itself. This is the intended place to decide the concrete theme, primary comparison, differentiation point, and first deliverable before `/re start`.

`/re start` moves the session to `RESEARCH` and runs `MockAgentRunner`. The first round proposes a dangerous dry-run operation so the approval gate can be tested. The harness switches to `APPROVAL_BLOCKED`, reports an `@Sora` approval request, and waits for `/re approve AP-1` or `/re reject AP-1 <reason>`.

If the internal novelty gate has produced a blocking result, `/re start` stays in `PLANNING` and posts an important-channel phase-gate decision request instead of entering `RESEARCH`. Use `/re accept PG-1` to proceed despite the risk, or `/re revise PG-1 <reason>` to keep planning and record the revision reason.

`/re approve` records the approval and continues research rounds until `MAX_ROUNDS`. `/re stop` marks the session `DONE` and posts a journal summary.

## Planning Dialogue And Scout

`/re plan` and normal PLANNING messages let the LLM decide whether to run similar-research search. If Sora says things like "これ調べて" or "類似研究ある？", the planning agent can generate search queries and use the paper-search tool internally. The internal scout strengthens the initial PLANNING phase:

- generates a search query from the research goal, or uses the provided query
- searches for similar research through the configured paper provider
- saves candidates to `papers.jsonl`
- deduplicates papers
- creates source-backed suggestions with paper IDs such as `[P-001]`
- proposes primary comparison, overlap points, differentiation hypotheses, weakness points, required decisions, risks, and questions for Sora
- evaluates a `Novelty Status`: `supported`, `unclear`, `crowded`, or `needs_human_decision`
- blocks `/re start` when the status is `unclear`, `crowded`, or `needs_human_decision`
- writes the result into `research_brief.md`

Use the scout output to decide whether the research should be a reproduction, comparison, implementation prototype, new evaluation, or a narrower investigation. Do not claim novelty without a source-backed comparison; mark uncertain points as `未確認` or `要検証`.

Fake-provider results are useful for tests, but they are not treated as real literature evidence. If only fake papers are available, the novelty gate returns `needs_human_decision`.

## Research Ledger And Reports

The harness keeps `journal.jsonl` as the full audit log. Research decisions are also summarized in `research_ledger.jsonl`, with one row per completed research round:

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

On `/re stop`, the harness writes:

- `artifacts/report.md`
- `run_summary.md`

The report includes sources, novelty status, ledger summary, unverified claims, risks, next steps, and AI provenance. A deterministic report review records warnings in `journal.jsonl` as `report_review_completed`.

If the deterministic review finds warnings, the harness creates a `review` phase gate such as `PG-2`. Experiment/SubAgent execution does not create phase gates; long-running or dangerous operations are still handled by the existing important notice and approval policies.

Default MVP limits:

- `MAX_ROUNDS=3`
- `MAX_TURNS_PER_CONVERSATION=4`
- `CONVERSATION_TIMEOUT_SECONDS=60`
- `CONVERGENCE_PATIENCE=2`
- `FRESH_INTERVAL=2`
- `REPORT_INTERVAL_SECONDS=60`
- `MAX_API_CALLS=0` means no hard API-call limit
- `MAX_TOTAL_TOKENS=0` means no hard estimated-token limit
- `MAX_AGENT_CALLS=0` means no hard real-agent-call limit
- `MAX_COMMAND_SECONDS=300` limits each real sub-agent command
- `PAPER_PROVIDER=arxiv` for normal operation; tests can keep the code default `fake`
- `RESEARCH_ARCHIVE_DIR=research_runs` controls where new `V001.0...` research folders are created

## Real Sub Agent

Set:

```env
SUB_AGENT_COMMAND=codex
```

When this is set, only the `sub` role is replaced with a real CLI call. The harness runs:

```text
codex exec --cd {research_dir} --skip-git-repo-check --sandbox workspace-write --ask-for-approval never -
```

The main/review/fresh roles remain controlled by the harness. This keeps state transitions, approval gates, journal logging, and Discord reports stable while testing one real role at a time.

If a real sub-agent needs a dangerous operation such as `sudo`, `chmod`, `chown`, deletion, external posting, or `git push`, it must not execute it directly. It should report one line:

```text
APPROVAL_REQUIRED: operation=sudo apt install graphviz; reason=required for figure rendering; impact=system package change; dry_run_result=not executed
```

The harness converts that line into an `AP-*` approval request and posts it to the important Discord channel. Sora can then respond from Discord with:

```text
/re approve AP-1
/re reject AP-1 <reason>
```

In the current MVP, approval records permission state and resumes the harness loop; it does not automatically run the dangerous command. This keeps unapproved `sudo` or destructive operations from being executed by accident.

## Approval Gate Policy

Approval is required for:

- file or folder deletion
- overwriting important files
- writing outside the research archive
- `git push`, PR creation, release actions
- external posting
- paid API use
- secret or `.env` transfer
- `sudo`, `chmod`, `chown`
- untrusted network access

Allowed after important-channel notice:

- long-running command candidates
- large or many-file generation candidates

These allowed operations are still reported to the important Discord channel and logged to `journal.jsonl`.

## Research Archive Policy

All files used by a research session should remain in that session's versioned folder. A new `/re new` creates a folder like:

```text
research_runs/V001.0_RS-abc123_research-theme/
```

The folder contains:

- `state.json`
- `journal.jsonl`
- `research_brief.md`
- `papers.jsonl` after literature search
- `artifacts/` for future experiment logs, datasets, figures, diffs, and other research outputs

Do not overwrite an active session with `/re new`. The harness refuses to do so unless the existing session is stopped first. Keep old folders for audit and comparison; start a new research folder for each new topic or major run.

Use `/re runs` to list archived research folders.

## Doctor

Use `/re doctor` to check the local harness setup. It reports:

- project root
- research archive directory
- important/log Discord channel configuration
- paper provider
- Codex CLI availability
- configured `SUB_AGENT_COMMAND`

## Journal

`research_runs/Vxxx.0_.../journal.jsonl` is append-only NDJSON. Each line is one event JSON object. Event types include command receipt, planning questions, brief updates, research round completion, fresh output, approval requests, approval/rejection, Discord reports, errors, and session end. Each line contains the MVP schema fields from the goal prompt, including:

- `timestamp`
- `event_type`
- `session_id`
- `version_label`
- `research_dir`
- `round_id`
- `mode`
- `research_goal`
- `research_brief_snapshot`
- `conversation_sessions`
- `sub_agent_output`
- `review_output`
- `claude_consultation`
- `fresh_agent_output`
- `approval_requests`
- `approvals_received`
- `files_changed`
- `commands_run`
- `errors`

Secret-looking values are masked before writing.

## Literature Search Phase A

The harness now supports lightweight literature search without adding RAG or a vector database. The default provider is `FakePaperSearchProvider`, which keeps tests deterministic. A thin `ArxivPaperSearchProvider` is also available through `PAPER_PROVIDER=arxiv`; it calls the official arXiv Atom API query endpoint and parses returned metadata.

```bash
python main.py --workdir /tmp/research-agent-demo re search "agent evaluation citation"
python main.py --workdir /tmp/research-agent-demo re papers
python main.py --workdir /tmp/research-agent-demo re paper P-001
```

Papers are stored in `research_runs/Vxxx.0_.../papers.jsonl`, one JSON object per paper:

- `paper_id`
- `title`
- `authors`
- `year`
- `venue`
- `url`
- `doi`
- `arxiv_id`
- `abstract`
- `summary`
- `source`
- `retrieved_at`
- `relevance_score`
- `confidence`
- `used_in_rounds`

Search results are deduplicated by DOI, arXiv ID, then normalized title. Research notes produced from papers include citation IDs such as `[P-001]`. If a claim cannot be grounded in a paper, mark it as `未確認` or `要検証`.

## Cost Management

The session records:

- API calls
- estimated tokens
- literature search count

```bash
python main.py --workdir /tmp/research-agent-demo re cost
```

When `MAX_API_CALLS` or `MAX_TOTAL_TOKENS` is greater than zero and the session reaches that limit, the harness moves to `APPROVAL_BLOCKED`, sends a cost-limit message, records the event in `journal.jsonl`, and does not continue automatically.

## Evaluation

`eval/golden_questions.jsonl` contains lightweight golden questions for regression checks. Run:

```bash
python main.py --workdir /tmp/research-agent-demo re eval
```

The current check reports question count, paper availability, required source type hits, forbidden-claim hits, and whether citation IDs are ready.

## Restart From Journal

The active root `state.json` points at the current session. If `research_runs/Vxxx.0_.../state.json` is missing but `journal.jsonl` remains, `SessionStore` can reconstruct the last known session state from the journal and make `/re resume` possible after a crash.

## Research Brief

`research_runs/Vxxx.0_.../research_brief.md` is the current session snapshot. It is separate from the journal: the brief is for reading the current state, while the journal is the append-only audit log.

## Mock Prompts

MockAgentRunner uses fixed templates for the four roles, but it can load role prompts from:

- `prompts/main.md`
- `prompts/sub.md`
- `prompts/review.md`
- `prompts/fresh.md`

If a file is missing, the code default is used.

## Tests

```bash
python -m pytest -q
```

The tests cover:

- `/re new -> /re plan -> /re start -> approval gate -> /re approve -> /re stop`
- internal similar-research planning support
- `/re start` not working before `/re new`
- `research_brief.md` and `journal.jsonl` creation
- fresh-agent and Claude-stub triggers
- literature search to `papers.jsonl`, dedupe, summaries, and citation IDs
- cost command and hard-limit blocking
- journal-based restore
- golden-question eval
- bounded conversation stopping
- pause/resume/status/redirect/idea/reject
- Command DTO entry point shared by CLI and Discord
- secret masking in the journal

## MVP Boundaries

In scope:

- deterministic mock agents
- file-based state
- fake Discord E2E
- dry-run approval gates

Out of scope for this MVP:

- real Claude/Codex/sub-agent process execution
- parallel sessions
- persistent database
- production-grade auth and permissions
- Web UI
- automatic PRs, pushes, or releases
