# Control-plane state model

## Purpose

`harness.control_plane` is the durable state boundary shared by the Research
and Kaggle domains. It models the user-facing Discord work thread separately
from an individual research run or compute attempt.

```text
Project
  └─ WorkSession  <-> Discord thread / forum post / another remote conversation
       ├─ Job
       │    └─ child Job(s)
       ├─ Event (ordered, immutable)
       └─ Steering (claimable user input)
```

The existing `ResearchSession` and its journal remain the active legacy
research-run implementation until the Discord thread router is connected. This
module is additive: it does not silently migrate, rewrite, or delete existing
research archives.

## Public API

Import the stable facade rather than the implementation modules:

```python
from pathlib import Path

from harness.control_plane import (
    ControlPlaneStore,
    Domain,
    EventLane,
    JobSpec,
    SteeringKind,
)

store = ControlPlaneStore(Path("runtime/control_plane"))
project = store.create_project("protein-folding", Domain.RESEARCH)
session = store.create_work_session(
    project.project_id,
    "Discord: compare two folding pipelines",
    origin="discord",
    external_ref={
        "guild_id": "123",
        "parent_channel_id": "456",
        "thread_id": "789",
    },
)
job = store.create_job(
    JobSpec(
        project_id=project.project_id,
        work_session_id=session.work_session_id,
        domain=Domain.RESEARCH,
        kind="research_run",
        backend_preferences=("remote_gpu", "local_cpu"),
        max_runtime_seconds=3600,
    )
)
store.append_event(
    event_type="job.queued",
    lane=EventLane.STATUS,
    project_id=project.project_id,
    work_session_id=session.work_session_id,
    job_id=job.job_id,
    idempotency_key="discord:message:111",
)
store.enqueue_steering(
    project_id=project.project_id,
    work_session_id=session.work_session_id,
    job_id=job.job_id,
    kind=SteeringKind.SUPPLEMENT,
    text="Add a leakage check at the next checkpoint",
    idempotency_key="discord:message:112:steering",
)
```

All payloads and metadata must be JSON-serializable. Credentials, API tokens,
raw Kaggle secrets, and unrestricted environment snapshots must not be stored
in these records.

## Entities

### Project

A long-lived Research, Kaggle, or hybrid project. A project can contain many
work sessions and survives Discord thread closure.

Important fields:

- `domain`: `research`, `kaggle`, or `hybrid`
- `root_ref`: an optional repository, workspace, or project path reference
- `metadata`: JSON-only domain metadata
- `status`: `active`, `paused`, or `archived`

A project cannot be archived while a work session remains open. An archived
project cannot be reopened.

### WorkSession

One interactive conversation and its control state. The primary Discord model
is one work session per thread or forum post.

Important fields:

- `origin`: for example `discord`, `cli`, or `api`
- `external_ref`: provider-specific identifiers
- `live_status_message_id`: the Discord message edited for live status
- `current_job_id`: the currently focused non-terminal job
- `status`: `open`, `active`, `paused`, `blocked`, or `closed`

External conversation identifiers are unique for the same origin when one of
these stable identity fields is present:

- `thread_id`
- `forum_post_id`
- `conversation_id`
- `session_key`

This prevents two work sessions from consuming the same Discord thread. Use
`find_work_session_by_external_ref()` in the future thread router.

A work session cannot close while it owns a queued, running,
waiting-for-approval, or paused job. When the focused job finishes, the store
selects another active job instead of incorrectly clearing `current_job_id`.

### JobSpec and Job

`JobSpec` is the backend-neutral request. `Job` adds runtime state.

`JobSpec` includes:

- `domain`, `project_id`, and `work_session_id`
- `kind` and JSON `payload`
- CPU, memory, GPU, accelerator, storage, network, and label requirements
- ordered `backend_preferences`
- runtime limit and priority
- optional parent job and experiment identifiers
- whether an external approval is required

The initial job state is `queued`. Valid state changes are deliberately
restricted:

```text
queued -> running | paused | failed | cancelled
running -> waiting_approval | paused | succeeded | failed | cancelled
waiting_approval -> queued | running | failed | cancelled
paused -> queued | running | failed | cancelled
terminal -> no different state
```

Optimistic concurrency is available through `expected_revision`. A stale
worker receives `ConflictError` rather than overwriting a newer state. Runtime
fields include backend identity, lease information, checkpoint reference,
artifact references, attempt count, error, and start/finish timestamps.

A child job must stay inside its parent's project and work session. A
non-hybrid project rejects jobs from another domain.

### Event

Events are immutable, globally ordered control-plane facts. They are not the
place for full training logs; large logs belong in artifacts.

Lanes:

- `control`: commands, approvals, cancellation, and routing decisions
- `status`: milestones used by live status
- `data`: structured domain observations
- `audit`: security and lifecycle evidence

`list_events(after_sequence=...)` is a forward cursor API. `latest_events()`
returns the newest matching tail in chronological order and is intended for
Live Status rendering. `snapshot()` also uses the tail rather than the oldest
events.

### Steering

Steering is user input that can arrive while work continues. It has an
explicit lifecycle (`pending`, `claimed`, then a terminal resolution) so two
workers cannot silently process the same input.

Default application policy:

| Input kind | Policy | Meaning |
|---|---|---|
| `question` | `read_only` | answer from state without changing the job |
| `supplement` | `next_checkpoint` | apply at the next safe checkpoint |
| `change` | `next_checkpoint` | revise execution at a safe boundary |
| `new_hypothesis` | `child_job` | preserve current work and create a child job |
| `cancel` | `immediate` | route through the control lane immediately |

Workers claim pending steering with a consumer identity and resolve it as
`applied`, `rejected`, or `superseded`. A claimed item cannot be resolved by a
different consumer.

## Storage and crash behavior

Default layout:

```text
control_plane/
├─ projects/<project-id>.json
├─ work_sessions/<work-session-id>.json
├─ jobs/<job-id>.json
├─ events/<zero-padded-sequence>-<event-id>.json
├─ steering/<steering-id>.json
├─ idempotency/
│  ├─ events/<sha256-key>.json
│  └─ steering/<sha256-key>.json
├─ index.json
└─ .control.lock
```

Properties:

- entity files use write, `fsync`, and atomic replace;
- mutations are serialized by an in-process re-entrant lock and a portable
  cross-process lock (`fcntl` on Linux, `msvcrt` on Windows);
- an event sequence is reserved before the event is published, so a crash can
  create a gap but cannot allow another writer to reuse the sequence;
- event files are immutable individual records, avoiding a partially appended
  JSONL line;
- hashed idempotency markers contain the canonical event or steering record;
  if the entity file is missing after a crash, a retry restores it;
- deleting or corrupting `index.json` causes its sequence counter to be rebuilt
  from event filenames at store initialization;
- identifiers are validated before they are used as paths.

Atomicity is per entity or event, not a transaction across every file in a
multi-entity operation. The lock prevents concurrent writers from observing a
normal in-process half-update, and all critical derived fields are recoverable,
but this is not a distributed database. A future multi-host scheduler should
put the same public models behind a transactional backend rather than sharing
this directory over an unsafe network filesystem.

## Idempotency rules

Discord delivery and worker retries must provide stable, namespaced keys, for
example:

```text
discord:<guild-id>:<message-id>:event
discord:<guild-id>:<message-id>:steering
worker:<job-id>:<checkpoint-id>:status
```

Reusing a key in another project, work session, or job raises `ConflictError`.
Within the same scope, the first canonical record is returned. Do not include
secrets in an idempotency key.

## Next integration boundary

The Discord Thread Router should be built above this API:

1. resolve or create a `Project`;
2. find a `WorkSession` by Discord `thread_id`, or create and bind one;
3. append the incoming Discord message as a control event with an idempotency
   key;
4. convert mid-run input to `Steering`;
5. render `snapshot()` into the single editable Live Status message;
6. create `JobSpec` records and hand them to the Compute Broker;
7. preserve the existing approval, checkpoint, cancellation, review, and
   artifact-promotion mechanisms while bridging legacy `ResearchSession` runs.

No live Discord, Kaggle, OpenAI, Codex-login, remote-GPU, or cloud credential is
required by this model layer or its tests.
