# Autonomous Compute Broker and experiment feedback

## Purpose

The routed Discord Edge remains the consultation and decision surface. Long
experiments do not execute in a Discord handler. Once a human accepts a
structured hypothesis, ResearchAgent creates a durable `Job`, selects a compute
backend, runs the experiment, collects artifacts, and turns the result into
reviewable next-hypothesis proposals.

```text
Discord Thread / WorkSession
        |
        | human accepts hypothesis:<id>
        v
Control Plane Job (queued)
        |
        v
Compute Broker
  |-- Kaggle Notebook
  |-- owned/external GPU Worker
  |-- local GPU
  `-- local CPU
        |
        v
materialize/repair -> smoke test -> submit -> poll -> collect
        |
        v
result:<job-id>:<hash> + artifact manifest
        |
        v
AI result review -> hypothesis:<child-id> proposal(s)
        |
        | human interprets result and selects a child hypothesis
        v
next child Job
```

The feedback loop is automatic up to proposal generation. It deliberately does
not auto-approve its own interpretation or next hypothesis.

## Human/Agent boundary

Humans retain the three research-direction decisions:

1. select the hypothesis to run after consulting the AI;
2. confirm the interpretation of an experiment result;
3. approve a Kaggle submission, or decide whether Research results should be
   developed into a paper.

The Agent owns implementation, code repair, smoke tests, backend selection,
experiment execution, progress tracking, collection, result normalization,
comparison, and generation of next-hypothesis candidates.

Operational safety is separate. A backend marked `paid=true` enters
`waiting_approval` before it is used. Destructive operations, credential
changes, publication, and Computer Use remain subject to their existing gates.

## JobSpec contract

The existing Control Plane `JobSpec` is the source of truth. Important fields:

- `domain`: `research` or `kaggle`;
- `project_id` and `work_session_id`;
- `resources`: CPU, RAM, GPU count/memory, storage, network, labels;
- `backend_preferences`: optional per-job priority order;
- `max_runtime_seconds`;
- `parent_job_id` and `experiment_id`;
- `payload`: executable and experiment metadata.

Supported payload keys:

```json
{
  "title": "child experiment",
  "hypothesis": "one falsifiable claim",
  "implementation_prompt": "requirements for the workspace-write provider",
  "source_dir": "/optional/existing/source",
  "entrypoint": ["python", "run.py"],
  "smoke_command": ["python", "smoke.py"],
  "outputs": ["result.json", "metrics.json", "progress.json"],
  "hypothesis_subject_ref": "hypothesis:H1",
  "parent_result_ref": "result:JOB-...:0123456789abcdef"
}
```

String commands are parsed with `shlex`; argv arrays are preferred. The
experiment subprocess is never invoked through a shell.

## Materialization and smoke testing

`ProviderExperimentMaterializer` prepares a private job workspace.

1. Optionally copy `payload.source_dir` without following symlinks.
2. Persist `JOB_SPEC.json`.
3. Use the existing workspace-write provider, normally Codex CLI, when
   `implementation_prompt` is present.
4. Resolve an executable entrypoint.
5. Run the explicit `smoke_command`, generated `smoke.py`, or at minimum
   `py_compile` for a Python script.
6. Refuse a GPU job that has no meaningful smoke path.

The provider is instructed to write:

- `progress.json` with `progress` in `[0, 1]` and a `stage`;
- `metrics.json`;
- `result.json` containing a summary, metrics, primary metric, risks,
  reproducibility information, and optionally structured `next_hypotheses`.

Materialization uses the existing provider sandbox and does not receive Discord,
Kaggle, Worker, or OpenAI credentials through the experiment environment.

## Backend routing

A Job's `backend_preferences` are tried first. The default order is:

```text
Kaggle:
  kaggle_notebook -> remote worker(s) -> local_gpu -> local_cpu

Research:
  remote worker(s) -> local_gpu -> local_cpu
```

The broker checks, in order:

- backend health/availability;
- domain support;
- accelerator type;
- GPU count and GPU memory;
- CPU and RAM requirements;
- ephemeral storage;
- network requirement;
- required capability labels.

The first compatible backend wins. Rejections are persisted in the
`compute.backend.selected` event. If no backend satisfies the JobSpec, the Job
is paused instead of silently running on an incompatible host.

### LocalProcessBackend

The local backend supports CPU and NVIDIA GPU modes. GPU discovery uses
`nvidia-smi`, or explicit `LOCAL_GPU_COUNT` / `LOCAL_GPU_MEMORY_MB` values.

The long command runs through `harness.compute_process` in a dedicated process
group. It writes `.compute_exit.json`, allowing a restarted Core to determine
whether a process completed even when the original in-memory `Popen` object no
longer exists.

Only an environment allowlist is forwarded. In particular, `DISCORD_*`,
`OPENAI_*`, `KAGGLE_*`, and Worker bearer tokens are not copied into the
experiment process.

For a Core container with direct NVIDIA access:

```bash
docker compose -f compose.yaml -f compose.local-gpu.yaml up -d
```

The normal GPU-less Core deployment continues to use only `compose.yaml`.

### KaggleNotebookBackend

The Kaggle backend:

1. validates or creates `kernel-metadata.json`;
2. runs `kaggle kernels push`;
3. polls `kaggle kernels status`;
4. downloads output with `kaggle kernels output`;
5. passes collected files through the same artifact and feedback pipeline.

Kaggle credentials exist only in Core's CLI environment. They are not written
to `JobSpec`, source bundles, Worker state, or Agent prompts. The backend never
submits a competition prediction file; final submission remains a separate
SHA-256-bound human gate.

### RemoteGpuBackend

An owned GPU PC and a rented GPU VM use the same Worker API. The Core sends:

- the immutable Job record;
- a bounded ZIP source bundle;
- bundle and content hashes.

The Worker receives no Discord, OpenAI, Codex, or Kaggle credentials. Artifact
downloads are checked against the advertised byte length and SHA-256 digest.

One Worker can be configured with:

```env
REMOTE_GPU_WORKER_NAME=remote_gpu
REMOTE_GPU_WORKER_URL=http://100.x.y.z:8090
REMOTE_GPU_WORKER_TOKEN=replace-with-a-distinct-token
REMOTE_GPU_WORKER_PAID=false
REMOTE_GPU_WORKER_GPU_COUNT=1
REMOTE_GPU_WORKER_GPU_MEMORY_MB=24576
```

Multiple Workers use `COMPUTE_REMOTE_WORKERS_JSON`. Use `token_env` to reference
a separate environment variable rather than placing the token in JSON.

## Portable Worker

Run the Worker directly:

```bash
WORKER_TOKEN='distinct-secret' \
WORKER_GPU_COUNT=1 \
WORKER_GPU_MEMORY_MB=24576 \
python -m harness.compute_worker --host 0.0.0.0 --port 8090 \
  --data-dir ./worker-runtime
```

Or use the container deployment:

```bash
docker compose \
  -f deploy/compose.worker.yaml \
  -f deploy/compose.worker-gpu.yaml \
  up -d
```

The base Worker compose file is CPU-capable. The GPU overlay adds NVIDIA device
access. The default published address is `127.0.0.1`; expose it through
Tailscale, a private network, or TLS rather than publishing the bearer-token API
directly to the Internet.

Worker endpoints:

```text
GET  /health
POST /v1/jobs
GET  /v1/jobs/{job_id}
POST /v1/jobs/{job_id}/cancel
GET  /v1/jobs/{job_id}/artifacts
GET  /v1/jobs/{job_id}/artifacts/{relative_path}
```

All endpoints except `/health` require `Authorization: Bearer <WORKER_TOKEN>`.
Paths, symlinks, file counts, byte counts, archive hashes, artifact sizes, and
artifact hashes are validated.

## Scheduler lifecycle and recovery

Control Plane states:

```text
queued
  |-- compatible free/owned backend --> running
  |-- paid/explicit-approval backend -> waiting_approval -> queued
  `-- no compatible backend ----------> paused

running
  |-- backend success -> collect -> feedback -> succeeded
  |-- backend failure -----------------> failed
  |-- cancel/steering -----------------> cancelled
  `-- repeatedly unknown --------------> paused
```

For each active Job, `ComputeRuntimeStore` persists:

- selected backend;
- backend job ID/handle;
- workspace and artifact directory;
- approval status;
- unknown-status poll count;
- materialized effective JobSpec;
- collection status and `result_ref`.

At Core startup, queued and running Jobs are re-enqueued. Remote and Kaggle jobs
are polled from their durable backend IDs. Local jobs are recovered from PID and
`.compute_exit.json`. Scheduler shutdown does not cancel active backend work by
default.

## Artifact collection

Every backend returns the same `CollectedResult`. The scheduler then:

1. copies only declared/default outputs into a job artifact directory;
2. rejects symlinks and unsafe relative paths;
3. builds a SHA-256 artifact manifest with file/byte limits;
4. stores hash-qualified artifact references on the Control Plane Job;
5. normalizes `result.json` and `metrics.json`;
6. writes a stable `result:<job-id>:<hash>` reference.

Full training logs remain artifacts. Discord receives status/milestone summaries,
not unbounded log streams.

## Automatic feedback and next hypotheses

`ResultFeedbackEngine` writes immutable events:

```text
experiment.result.collected
experiment.hypothesis.proposed
experiment.feedback.generated
compute.job.completed
```

It first consumes structured `next_hypotheses` emitted by the experiment. If
none exist, it asks the configured planning provider to generate up to the
configured limit. If the provider is unavailable, a conservative single-factor
reproduction proposal is created so the result is not dropped from the loop.

Each child proposal contains:

- stable `hypothesis:<id>`;
- parent Job and `result_ref`;
- falsifiable claim;
- implementation prompt;
- entrypoint/smoke contract;
- resources and backend preferences;
- output files and success/failure conditions.

The proposal is visible to subsequent Discord consultation. It is not executed
until:

1. a human records `result_interpretation=accept` for the exact parent
   `result_ref`; and
2. a human records `hypothesis=accept` for the child `hypothesis:<id>`.

The accepted proposal is converted into a deterministic child Job, preserving
the failed/successful parent and preventing accidental duplicate Jobs from
Discord retries.

## Event types

The scheduler emits bounded status/control events including:

- `compute.job.enqueued`;
- `compute.backend.selected`;
- `compute.approval.required` / `compute.approval.accepted`;
- `compute.smoke.passed`;
- `compute.job.submitted`;
- `compute.job.progress` at stage/10-percent boundaries;
- `compute.steering.applied`;
- `compute.job.completed`, `compute.job.failed`, `compute.job.paused`, or
  `compute.job.cancelled`.

These events are idempotent and suitable for the Discord Live Status renderer.

## Known real-environment validation boundary

Credential-free CI verifies state transitions, routing, local subprocesses,
source-bundle safety, injected Kaggle CLI behavior, injected remote transports,
Worker execution, artifact collection, and the parent-result/child-hypothesis
gates.

The following still require explicit real-environment checks before relying on
them for expensive work:

- a real Kaggle Notebook push/status/output cycle;
- the intended owned GPU PC over Tailscale or another private transport;
- NVIDIA Container Toolkit and GPU passthrough on each target OS;
- a rented paid Worker approval flow;
- restart recovery during an actual multi-hour training run.
