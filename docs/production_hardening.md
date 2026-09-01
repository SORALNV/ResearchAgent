# Production hardening

The active Discord product applies this layer after the natural-channel and
MethodBook integrations. It does not add a new execution mode.

## Discord authorization

State-changing operations are fail-closed by default. Configure at least one
trusted Discord user before starting the bot:

```env
DISCORD_ACCESS_CONTROL_REQUIRED=true
DISCORD_ALLOWED_USER_IDS=123456789012345678
DISCORD_ADMIN_USER_IDS=
```

`/agent setup` may be run only by a globally allowed user. The setup actor is
stored as `ChannelSessionConfig.created_by` and becomes the channel-session
owner. The owner, globally allowed users, and admin users may send Agent
instructions, steer or interrupt Codex, resolve Codex approvals, approve/cancel
Compute, record decisions, and archive the channel. Read-only status remains
available to users who can already view the Discord channel.

Set `DISCORD_ACCESS_CONTROL_REQUIRED=false` only for a deliberately open, trusted
Discord server. Bot/non-snowflake actor IDs are rejected.

## Codex App Server wire contract

Environment-created production runtimes normalize legacy repository input
objects to the current generated App Server v2 field:

```json
{"type":"text","text":"...","text_elements":[]}
```

The obsolete `textElements` field is not emitted by the active runtime. Existing
low-level direct-constructor fixtures remain isolated compatibility fixtures;
`get_shared_codex_app_server()` always uses environment-created official
settings.

## Codex state and container home

The default host authentication directory is excluded from both Git and Docker
build contexts:

```text
codex-home/
**/.codex/
```

The Core tmpfs home is created for UID/GID 10001 with mode 0700. The hardening
does not use `seccomp=unconfined`.

## Compute approval binding

The active Compute stack uses `BackendBoundApprovalScheduler`. A paid-backend
approval is stored with the exact selected backend. If availability changes and
the broker selects another backend, the prior approval is invalidated and the
Job returns to `waiting_approval` for the new backend.

## Remote Worker artifact transport

Default urllib Worker traffic is same-origin across redirects and response-size
bounded. Artifact collection additionally requires:

- a safe, unique relative path;
- an absolute or relative URL that resolves to the configured Worker origin;
- a non-negative advertised size;
- a SHA-256 value that normalizes to 64 lowercase hex;
- per-file, total-byte, and file-count limits;
- streaming download limits before files enter the final artifact directory;
- exact size and SHA-256 verification in a staging directory.

A cross-origin artifact URL or redirect is rejected before an Authorization
header can be forwarded. The default limits are configurable:

```env
REMOTE_WORKER_MAX_RESPONSE_BYTES=8388608
REMOTE_ARTIFACT_MAX_FILES=2000
REMOTE_ARTIFACT_MAX_FILE_BYTES=2147483648
REMOTE_ARTIFACT_MAX_TOTAL_BYTES=4294967296
```

## Validation boundary

Credential-free CI covers policy, wiring, source-tree exclusions, same-origin
checks, byte limits, artifact verification, and backend-bound approval selection.
Live Discord, authenticated Codex, Kaggle, and external Worker operations remain
explicit deployment checks.
