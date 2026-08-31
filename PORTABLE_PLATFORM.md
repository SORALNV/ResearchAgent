# Portable ResearchAgent + KaggleAgent Platform

This document describes the portable execution platform introduced for the
combined ResearchAgent and KaggleAgent. It is designed for a Windows host with a
Ryzen 7 9700X and 128 GB RAM, while keeping the Edge and Core code portable to
Jetson without source changes.

## Target deployment

```text
Discord phone/PC
       |
       v
Discord Edge container
  - on Windows beside Core, or
  - on Jetson as a low-power always-on gateway
       |
       | authenticated Core API over Docker network / LAN / Tailscale
       v
ResearchAgent Core container
  - durable SQLite state
  - OpenAI Responses runtime
  - Codex CLI through hardened harness
  - WorkSession and thread routing
  - ComputeBroker and JobScheduler
  - Kaggle Gateway and submission approvals
       |
       +--> Kaggle Notebook backend
       +--> owned GPU Worker backend
       +--> paid GPU VM Worker backend
       +--> local CPU smoke-test backend
```

The Bot host does not need a GPU. Long-running compute is represented as a
`Job` and delegated to a `ComputeBackend`.

## Core concepts

- `Project`: long-lived research or Kaggle subject.
- `WorkSession`: one Discord thread/forum post and one bounded objective.
- `Job`: one durable compute unit such as smoke test, CV, training, or inference.
- `JobEvent`: ordered progress/milestone/error/approval event.
- `SteeringEvent`: question, constraint, redirect, hypothesis, pause, or cancel.
- `AgentRuntime`: internal ChatGPT/OpenAI Responses, Codex CLI, or future runtime.
- `ComputeBackend`: Kaggle Notebook, remote GPU, GPU VM, or local CPU.

The source of truth is SQLite. Discord messages are a view and control surface,
not the durable state.

## Internal ChatGPT/OpenAI and Codex use

`AgentRuntimeRouter` selects a runtime by declared capability:

- conversation, reasoning, function tools, and approved computer-use prefer
  `OpenAIResponsesRuntime`;
- coding, file edits, and shell work prefer `CodexCliRuntime`;
- mutating Codex failures do not silently fall back to another model;
- model tool calls are restricted to `HarnessToolRegistry`;
- long-running training is proposed as a durable Job rather than run inside the
  conversation request.

Codex CLI is executed through the existing hardened process harness. This keeps
process-group cancellation, timeouts, environment allowlisting, sandbox flags,
token accounting, checkpoints, and artifact manifests.

OpenAI Responses may call typed tools such as:

- read WorkSession / Job status;
- record a steering instruction;
- propose a durable Job;
- read Kaggle competition/CV/experiment state;
- initialize a Kaggle workspace;
- validate a submission candidate;
- request a human-gated Kaggle submission;
- request a human-gated computer-use session.

Model runtimes never receive the Discord token, Kaggle token, Core token, or
Worker token. Codex receives `OPENAI_API_KEY` only when explicitly included in
`AGENT_ENV_ALLOWLIST`.

## Computer-use fallback

Computer-use is an exception path, not the default Kaggle integration.
Deterministic API/CLI/tools should be preferred. To enable it:

1. build Core with `INSTALL_COMPUTER_USE=true`;
2. set `COMPUTER_USE_ENABLED=true`;
3. set a narrow `COMPUTER_USE_ALLOWED_DOMAINS` allowlist;
4. provide the current official Responses API computer tool definition through
   `OPENAI_COMPUTER_TOOL_JSON`;
5. approve each session and unresolved safety check from Discord.

The browser runs in the Core container with an isolated browser context. Domain
changes outside the allowlist are rejected.

## Discord remote-style UX

Use a Discord Forum channel for `DISCORD_WORK_SESSIONS_CHANNEL_ID`.

```text
/agent new domain:research title:... objective:...
/kg new competition_url:... title:... objective:...
```

Each command creates one WorkSession and one Forum post/thread. Inside it:

- ordinary text is conversation or steering;
- `code: ...` prefers Codex and allows workspace changes through the harness;
- `computer: ...` creates a computer-use approval proposal;
- `/agent status` reads state without waiting for compute;
- `/agent cancel` cancels active Jobs and model runtime processes.

The Edge edits one pinned Live Status message. It posts only milestones,
artifacts, failures, and approval requests as new messages. Fine-grained logs
remain in SQLite and artifact files, keeping the thread readable.

Current status tags expected in the Forum channel are:

- Planning
- Waiting Input
- Queued
- Running
- Review
- Waiting Approval
- Completed
- Failed
- Paused

## Kaggle safety and experiment contract

Kaggle projects use separate durable records:

- `KaggleCompetitionState`
- `CVSpec`
- `ExperimentRecord`
- `SubmissionCandidate`

The generated workspace contains:

```text
competition.json
COMPETITION_POLICY.md
AGENTS.md
EXP_SUMMARY.md
docs/
data/raw/
cv/
src/
experiments/
submissions/
logs/
```

Rules:

- competition rules must be acknowledged before data download or submission;
- a CV specification is locked before scores are compared;
- a new hypothesis creates a child experiment;
- raw data is not overwritten;
- failures and config diffs are retained;
- a submission CSV is validated against sample submission;
- approval is bound to the exact SHA-256 file hash;
- a changed file invalidates approval;
- the model cannot call `kaggle competitions submit` directly.

`KaggleGateway` alone holds Kaggle credentials and performs the final command.

## Compute routing

`ComputeBroker` evaluates a `JobSpec` containing resource requirements rather
than hard-coding a GPU model.

For Kaggle projects the default order is:

```text
Kaggle Notebook -> owned remote GPU -> GPU VM -> local CPU
```

For general research the default order is:

```text
owned remote GPU -> GPU VM -> local CPU
```

Kaggle compute for non-Kaggle research is disabled by default. Paid backends
enter `WAITING_APPROVAL` before submission.

The Kaggle backend packages a script/notebook, pushes it with Kaggle CLI, polls
coarse state, and collects output files. The owned/VM backend sends a verified
source ZIP to a Worker API. The Worker validates paths, symlinks, file count,
size, and bundle hash before execution.

## Secure deployment files

Do not use one shared environment file for production. Use the role-separated
files under `deploy/`:

```text
deploy/.env.core
deploy/.env.edge
deploy/.env.worker
```

Templates are provided as `.example` files.

### Windows 9700X host: Core + Edge

From PowerShell or a Linux shell in the repository:

```bash
cp deploy/.env.core.example deploy/.env.core
cp deploy/.env.edge.example deploy/.env.edge
# fill secrets
docker compose -f deploy/compose.core-edge.yaml up -d --build
```

Docker Desktop with Linux containers is sufficient. An Ubuntu VM is also
supported because the deployment contract is container-based.

### Jetson: Edge only

Build the same source tree for ARM64 and run only the Edge:

```bash
cp deploy/.env.edge.example deploy/.env.edge
# set RESEARCH_AGENT_CORE_URL to the Windows Core's Tailscale/LAN URL
docker compose -f deploy/compose.edge.yaml up -d --build
```

No source change is required. The Edge contains no Codex, OpenAI, Kaggle, or GPU
credentials.

### Owned GPU or GPU VM Worker

```bash
cp deploy/.env.worker.example deploy/.env.worker
# set a unique worker token and advertised resources
docker compose \
  -f deploy/compose.worker.yaml \
  -f deploy/compose.worker-gpu.yaml \
  up -d --build
```

Register the Worker in Core with `REMOTE_COMPUTE_WORKERS_JSON`. Put the Worker
behind Tailscale, WireGuard, a private VPC, or authenticated HTTPS. Do not expose
it directly to the public Internet.

The generic Worker image does not bundle a machine-learning framework. Extend
the `worker` target or install the required framework in the Worker image used
for a particular GPU fleet. The Core/Edge/Job API remains unchanged.

## Ports and security boundaries

- Core API: TCP 8080, bearer `RESEARCH_AGENT_CORE_TOKEN`.
- Worker API: TCP 8090, separate bearer `RESEARCH_WORKER_TOKEN`.
- Discord Edge connects outbound to Discord and Core.
- Core owns OpenAI and Kaggle credentials.
- Worker owns no model, Kaggle, or Discord credentials.
- Model-generated child processes receive environment allowlists.
- Containers use `no-new-privileges` and non-root users.

Bind Core/Worker to localhost unless accessed through a private overlay network
or a reverse proxy with TLS.

## Offline validation scope

The repository includes deterministic tests for:

- SQLite persistence and event ordering;
- thread/session route recovery;
- Codex/OpenAI capability routing;
- OpenAI function tool loops without a live API;
- fail-closed computer-use approval;
- compute routing, paid-backend approval, scheduling, and collection;
- source bundle hash/path/symlink controls;
- Worker child-process secret isolation;
- Kaggle CV/experiment registry;
- submission format and immutable-hash approval;
- authenticated Core API;
- container role separation.

Live OpenAI, Codex authentication, Discord, Kaggle, GPU, Tailscale, and Jetson
execution remain environment-specific smoke tests. CI does not claim those
external services were exercised.
