# Codex App Server Discord frontend

## Scope

The routed Discord bot uses one long-lived official `codex app-server` process
instead of starting `codex exec` for each agent invocation. ResearchAgent remains
the outer application: its WorkSession, approval, checkpoint, review, Compute,
Kaggle, paper, and artifact policies are not replaced by this transport change.

```text
Discord channel / thread
  -> ResearchAgent WorkSession
  -> durable Codex thread binding
  -> codex app-server over JSONL stdio
  -> Codex Core / Harness
```

Each routed Discord WorkSession uses the binding key
`discord:<work_session_id>`. The corresponding Codex `thread.id` is stored under
`CODEX_APP_SERVER_STATE_DIR/thread_bindings.json`. A later message in the same
Discord thread resumes that Codex thread with `thread/resume` before starting a
new turn.

## Official protocol methods used

Client lifecycle:

- `initialize`
- `initialized`

Conversation lifecycle:

- `thread/start`
- `thread/resume`
- `turn/start`
- `turn/steer`
- `turn/interrupt`

Execution and progress are read from App Server notifications including
`turn/started`, `turn/completed`, `turn/plan/updated`, `item/started`,
`item/completed`, `item/mcpToolCall/progress`, and
`thread/tokenUsage/updated`.

Command and file-change approvals use the server-initiated requests:

- `item/commandExecution/requestApproval`
- `item/fileChange/requestApproval`

The Discord response is sent to the same JSON-RPC request ID as
`{"decision": ...}`. Supported decisions are `accept`, `acceptForSession`,
`decline`, and `cancel`.

The client does not send the deprecated `multiAgentMode` parameter. Internal
Codex Harness activity is left to Codex. `collabAgentToolCall`,
`subAgentActivity`, and child `thread/started` events are observed and associated
with the parent WorkSession; ResearchAgent does not implement a replacement
subagent protocol.

## Discord behavior

An idle WorkSession message starts a normal routed consultation turn. If the same
WorkSession already has an active Discord Codex turn, another normal message is
sent using `turn/steer` with the active `turnId` as `expectedTurnId`.

Available controls:

```text
/agent codex_status
/agent steer instruction:<text>
/agent interrupt
/agent codex_approvals
/agent codex_approval approval_ref:<CAP-...> decision:<decision>
```

Existing Research/Kaggle commands remain available, including hypothesis,
interpretation, submission, paper, Compute approval, and Compute cancellation.

App Server events are persisted as scoped Control Plane events. Discord receives
milestones rather than every output delta:

- turn start/completion;
- plan updates;
- command, file-change, MCP, and multi-agent item lifecycle;
- approval requests;
- App Server errors.

Full model prompts, assistant deltas, command output, and reasoning text are not
copied into the Discord progress event payload. The final assistant message is
still delivered through the existing consultation response path.

## Configuration

```env
CODEX_HOME=/data/codex
CODEX_APP_SERVER_COMMAND=codex app-server --listen stdio://
CODEX_APP_SERVER_STATE_DIR=codex_app_server
CODEX_APP_SERVER_REQUEST_TIMEOUT_SECONDS=30
CODEX_APP_SERVER_TURN_TIMEOUT_SECONDS=
CODEX_APP_SERVER_APPROVAL_POLICY=on-request
CODEX_APP_SERVER_APPROVALS_REVIEWER=user
CODEX_APP_SERVER_NETWORK_ACCESS=false

AGENT_RUNTIME_ORDER=codex_app_server,openai_responses
MAIN_AGENT_RUNTIME_ORDER=codex_app_server,openai_responses
SUB_AGENT_RUNTIME_ORDER=codex_app_server
REVIEW_AGENT_RUNTIME_ORDER=openai_responses,codex_app_server
PLANNING_AGENT_RUNTIME_ORDER=codex_app_server,openai_responses
```

`codex_cli` remains accepted as a configuration alias for existing deployments,
but the provider router maps it to App Server. Direct Codex commands are rejected
by the generic subprocess sandbox, so the alias cannot re-enable `codex exec`.

The App Server process receives a small environment allowlist needed for Codex
authentication, TLS, proxies, and `CODEX_HOME`. Discord, Kaggle, Worker, and other
application credentials are not inherited by it.

## Remaining protocol boundaries

This integration handles the stable command-execution and file-change approval
flows required for the current Discord bot. Other server-initiated interaction
families, such as experimental `item/tool/requestUserInput`, MCP elicitation
forms, permission-profile grants, dynamic tool callbacks, external attestation,
and external clock callbacks, are not exposed as Discord forms. They fail closed
as unsupported server requests rather than being answered with an invented
payload.

The automated test suite uses an injected deterministic App Server process. It
verifies JSONL request/response correlation, persistent thread resume, steer,
interrupt, approvals, child-thread association, Control Plane event persistence,
and removal of the `codex exec` path. Live Codex login and live Discord delivery
remain deployment-time checks because repository CI has no credentials.
