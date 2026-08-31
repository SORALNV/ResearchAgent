# ResearchAgent + Kaggle Control Plane

## Deployment decision

The primary deployment is a GPU-less Ryzen 7 9700X / 128 GB Windows host running the Linux container through Docker Desktop or a Linux VM. The same source and image build for `linux/arm64` on Jetson.

Jetson can run the complete control plane for light workloads, but the recommended long-term role is an always-on Discord edge/watchdog while the 9700X host owns the SQLite source of truth, Agent execution, scheduling, and artifacts. This split is optional; standalone mode works on either host.

```text
Discord
  -> ResearchAgent Control Plane
       -> internal Codex / OpenAI provider harness
       -> WorkSession / Job / Event registry
       -> ComputeBroker
            -> Kaggle Notebook
            -> owned GPU Worker over VPN
            -> rented GPU VM Worker
            -> bounded local CPU smoke test
```

The Bot host does not require a GPU.

## One codebase on Windows and Jetson

Use the same files:

```text
Dockerfile.control-plane
compose.control-plane.yaml
.env.control-plane.example
```

Windows Docker Desktop:

```env
RESEARCH_AGENT_PLATFORM=linux/amd64
CONTROL_PLANE_MODE=core
```

Jetson:

```env
RESEARCH_AGENT_PLATFORM=linux/arm64
CONTROL_PLANE_MODE=standalone
```

No source change is required. Persist `/runtime` and `/home/research/.codex`; do not bake authentication or state into the image.

```bash
cp .env.control-plane.example .env
docker compose -f compose.control-plane.yaml up -d --build
```

Codex authentication is mounted through `CODEX_HOME`. Agent invocations still receive isolated per-run HOME directories, so Discord and unrelated process secrets are not inherited.

## Shared domain model

Research and Kaggle use the same infrastructure:

```text
Project
  -> WorkSession (one Discord thread)
       -> Job (one execution unit)
            -> JobEvent
            -> artifacts / result
```

Research adds papers, hypotheses, evidence, and novelty gates. Kaggle adds competition rules, CV specifications, child experiments, OOF/results, and hash-bound submission candidates.

`ControlPlaneRegistry` stores Project, WorkSession, Job, and JobEvent in SQLite/WAL. `WorkSessionStore` stores conversation, steering instructions, and the Discord live-status message. `KaggleStore` stores competition-specific state.

## Discord experience

Recommended private channels:

```text
#agent-inbox
#work-sessions      (Forum channel)
#approvals
#agent-ops
```

A message in `#agent-inbox` creates one Forum thread and one WorkSession. The parent inbox receives only the thread link. The thread receives:

- a pinned/editable live-status card
- milestones such as queued, backend selected, started, completed, failed
- user questions and Agent answers
- steering instructions applied at a safe checkpoint
- explicit approval interactions

Progress is projected from structured JobEvent records rather than raw training logs. Fine-grained fold/epoch events are available for owned GPU and GPU VM workers. Kaggle Notebook progress is limited to package, push, kernel status, and output collection.

Free text cannot approve submission, paid GPU startup, or public upload. Those actions require explicit commands/components.

## Compute routing

Default Kaggle order:

```text
kaggle_notebook -> remote_gpu -> gpu_vm -> local_cpu
```

Default research order:

```text
remote_gpu -> gpu_vm -> local_cpu -> kaggle_notebook
```

A JobSpec states capabilities rather than a specific GPU model:

```yaml
resources:
  accelerator: gpu
  min_vram_gb: 16
  ram_gb: 32
  network_required: false
backend_preferences:
  - kaggle_notebook
  - remote_gpu
  - gpu_vm
```

The broker selects only compatible configured backends. If none match, the Job becomes blocked rather than silently running somewhere expensive.

## Kaggle Notebook backend

`KaggleNotebookBackend` creates a deterministic package containing:

```text
kernel-metadata.json
research-agent-job.json
run.py or notebook
source snapshot
```

It then performs:

```text
kaggle kernels push
kaggle kernels status (poll)
kaggle kernels output
```

The Notebook must emit structured outputs such as:

```text
result.json
metrics.json
oof.parquet
submission.csv
```

Credentials are held only by `KaggleCliTransport` / `KaggleSubmissionGateway`, never by Codex or other Agent subprocesses. Local cancellation stops polling; the remote Kaggle run may continue because the public CLI does not provide reliable remote cancellation. This warning is recorded in the Job result.

## Experiment discipline

Kaggle follows the fixed order:

1. confirm rules
2. register/fingerprint data
3. produce a minimal valid submission shape
4. define and approve CV
5. smoke test
6. child experiment
7. full CV/training
8. review results and artifacts
9. prepare submission candidate
10. approve exact file hash
11. submit that unchanged hash

Experiments never overwrite their parent. Failed experiments remain in the registry with their failure reason.

## Submission safety

Submission preparation validates:

- exact columns and order against sample submission
- exact row count
- ID order and uniqueness
- missing and non-finite values
- configured prediction ranges
- regular-file status
- SHA-256

Human approval is tied to candidate ID and SHA-256. If the file changes after validation or approval, submission is rejected. `KaggleSubmissionGateway` is idempotent for already-submitted candidates.

## Remote GPU and GPU VM

`RemoteComputeBackend` uses a typed HTTP contract suitable for Tailscale/WireGuard:

```text
POST /v1/jobs
GET  /v1/jobs/{id}
GET  /v1/jobs/{id}/events?after=N
POST /v1/jobs/{id}/cancel
```

The remote Worker streams structured events and returns artifacts/results. A Worker token is held by the Control Plane, not passed to internal Agents. HTTPS or a private VPN transport is required in production.

## Restart behavior

SQLite/WAL is the source of truth. Jobs found in `preparing`, `running`, or `collecting` after Control Plane restart are marked `interrupted` by default. They are not automatically resubmitted because a Kaggle or remote GPU Job may already be running and a duplicate would waste quota or money.

Set `REQUEUE_INTERRUPTED_JOBS=true` only for idempotent local/Fake jobs. Remote recovery will later use backend-specific reattachment.

## Current explicit limitations

The implementation intentionally does not claim live-service validation for:

- real Kaggle credentials and GPU quota
- real Discord Forum permissions/tags
- real Codex login inside the deployed container
- an owned GPU Worker server and artifact upload
- a rented GPU provider provisioning API
- real OpenAI Computer Use browser execution

The interfaces, state model, security gates, Fake transports, and amd64/arm64 builds are credential-free and CI-testable. Live smoke tests remain deployment tasks, not hidden assumptions.
