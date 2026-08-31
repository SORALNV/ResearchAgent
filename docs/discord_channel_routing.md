# Discord channel domain routing

## Purpose

The Discord Edge selects `research` or `kaggle` from configured channel IDs.
Users do not switch modes with a command, and a model cannot choose its own
domain.

The runtime hierarchy is:

```text
configured Discord channel
  -> dedicated Discord Thread / Forum post
    -> one WorkSession
      -> domain-specific AI consultation handler
      -> Job / Event / Steering / human decision gates
```

An exact channel or thread mapping wins over its parent mapping. Otherwise a
Thread inherits its parent's domain. An unconfigured channel is ignored by the
real Discord Edge and rejected by the router API. It never falls back to another
domain.

## Configuration

Use either the explicit map, split lists, or both:

```env
# Compact form
DISCORD_CHANNEL_DOMAIN_MAP=111111111111111111=research,222222222222222222=kaggle

# Equivalent split form
DISCORD_RESEARCH_CHANNEL_IDS=111111111111111111
DISCORD_KAGGLE_CHANNEL_IDS=222222222222222222

# Durable Project / WorkSession / Job / Event / Steering state
CONTROL_PLANE_DIR=control_plane

# A message in a mapped parent text channel creates a dedicated Thread.
DISCORD_CREATE_THREADS=true
```

JSON is also accepted:

```env
DISCORD_CHANNEL_DOMAIN_MAP={"111111111111111111":"research","222222222222222222":"kaggle"}
```

Configuration rules:

- IDs must be numeric Discord snowflakes.
- A channel cannot be mapped to both domains.
- `hybrid` is deliberately rejected at the Discord boundary.
- Unknown channels fail closed.
- `DISCORD_CHANNEL_ID` alone keeps the legacy single-channel Research bot.
- Setting one of the new domain-routing variables selects the routed Discord
  Edge when `python main.py bot` starts.

## Runtime behavior

### Parent channel

With `DISCORD_CREATE_THREADS=true`, a normal message in a mapped parent text
channel creates a dedicated Thread. The original message becomes the first
message of the new WorkSession, and the AI response is posted in that Thread.

Forum posts already arrive as Discord Threads and are used directly.

### Thread

Each Thread or direct mapped conversation has one durable WorkSession. The
router stores:

- the selected domain;
- guild, parent-channel, channel, and thread identifiers;
- the three human direction decisions for that domain;
- the AI-owned execution responsibilities;
- idempotent incoming and outgoing Discord events.

A repeated Discord delivery reuses the original incoming event and cached
assistant response rather than invoking the model twice.

### AI consultation

The routed Edge uses the existing provider policy:

```env
PLANNING_AGENT_RUNTIME_ORDER=openai_responses,codex_cli
```

The Research and Kaggle handlers receive different system constraints.

Research mode emphasizes prior work, falsifiability, evidence, and
reproducibility. Kaggle mode emphasizes competition rules, locked validation,
leakage control, reproducibility, and candidate validation. Neither handler may
perform a final human-only decision.

This handler is a read-only conversation and planning layer. Long-running
experiments must become durable Jobs and be delegated to a Compute Backend;
they are not executed inside the Discord response path.

## Human responsibility boundary

There are three research-direction decisions per domain.

| Domain | Human-only decisions |
|---|---|
| Research | hypothesis selection; final result interpretation; whether to start preparing a paper |
| Kaggle | hypothesis selection; final result interpretation; whether to submit an exact candidate |

The AI owns the remaining routine work:

- implementation and code changes;
- debugging and retry design;
- public-information and literature investigation;
- experiment execution through a Compute Backend;
- logs, artifacts, comparisons, and reproducibility evidence;
- review assistance and interpretation candidates.

Operational safety approvals remain separate. Paid compute, credential changes,
destructive operations, external publication, and other high-impact actions may
still require approval even though they are not research-direction decisions.

## Discord commands

The routed Edge registers one common `/agent` group.

```text
/agent mode
/agent status

/agent hypothesis subject_ref:<id> verdict:<accept|reject|defer> note:<text>
/agent interpret result_ref:<id> verdict:<accept|reject|defer> interpretation:<text>

/agent submit sha256:<64 hex> verdict:<accept|reject|defer> note:<text>
/agent paper result_ref:<id> verdict:<accept|reject|defer> note:<text>

/agent gate action:<action> subject_ref:<id>
```

`approve`, `approved`, and `yes` are accepted aliases for `accept`; `deny` and
`rejected` are aliases for `reject`.

Domain restrictions are enforced:

- `/agent submit` is valid only in a Kaggle WorkSession.
- `/agent paper` is valid only in a Research WorkSession.
- a model or bot account cannot record a human decision.
- submission approval is normalized to `sha256:<64 lowercase hex>`.
- a changed submission file hash invalidates the previous approval.
- later decisions for the same subject override earlier decisions.

Controlled action names for `/agent gate` are:

```text
start_experiment
continue_from_result
submit_kaggle
start_paper_draft
```

The gate API is also callable by future Compute Broker, Kaggle submission, and
paper-generation integrations. These integrations must check the gate
immediately before the controlled action.

## Failure behavior

The routed path is deliberately fail closed:

- no domain mapping: no routed processing;
- no handler for the selected domain: no fallback to the other domain;
- model/runtime failure: the incoming message remains recorded and an explicit
  diagnostic response is stored;
- missing human decision: controlled action remains blocked;
- wrong domain or wrong subject/hash: controlled action remains blocked;
- duplicate Discord delivery: no duplicate model call or decision event.

Credentials are not stored in Project, WorkSession, Event, or Steering records.
