# Portable ResearchAgent runtime

## Deployment target

The primary deployment is a Linux container on the Ryzen 7 9700X / 128 GB
Windows host through Docker Desktop or another Linux-container runtime. The
same `Dockerfile` and `compose.yaml` are intentionally architecture-neutral and
can be rebuilt on Jetson Linux without source changes.

```text
Windows host (linux/amd64)          Jetson (linux/arm64)
          \                              /
           same Dockerfile + compose.yaml
                         |
                ResearchAgent Core
                         |
     Codex CLI / OpenAI Responses / Computer bridge
```

GPU compute is not coupled to this container. Kaggle Notebook, a remote GPU
worker, or a rented GPU VM can be added later as compute backends while the
control plane remains unchanged.

## Runtime provider policy

The harness chooses providers from a comma-separated chain:

```env
AGENT_RUNTIME_ORDER=codex_cli,openai_responses
```

Supported values:

- `codex_cli`: local Codex CLI, including its workspace harness and sandbox.
- `cli`: another explicitly configured local command.
- `openai_responses`: OpenAI Responses API for planning, synthesis, and review.
- `openai_computer`: optional GUI automation through an external bridge.

Role-specific chains override the global chain:

```env
MAIN_AGENT_RUNTIME_ORDER=codex_cli,openai_responses
SUB_AGENT_RUNTIME_ORDER=codex_cli
REVIEW_AGENT_RUNTIME_ORDER=openai_responses,codex_cli
FRESH_AGENT_RUNTIME_ORDER=openai_responses,codex_cli
PLANNING_AGENT_RUNTIME_ORDER=openai_responses,codex_cli
```

`workspace-write` is restricted to local CLI providers. Plain Responses API
calls can reason and review, but they are not treated as a substitute for the
Codex workspace harness. This prevents a text-only fallback from falsely
claiming that local files were modified.

Every attempt is written to:

```text
<research-run>/artifacts/runtime_events.jsonl
```

The existing checkpoint, review, approval, artifact-hash, and convergence
layers remain above the provider router.

## First start on Windows

1. Install Docker Desktop and enable Linux containers.
2. Clone the repository.
3. Create the persistent directories and configuration:

   ```powershell
   Copy-Item .env.example .env
   New-Item -ItemType Directory -Force runtime,research_runs,codex-home
   ```

4. Build the image:

   ```powershell
   docker compose build
   ```

5. Authenticate Codex into the persistent `CODEX_HOME` volume:

   ```powershell
   docker compose run --rm core codex login
   ```

6. Set Discord channel IDs and token in `.env`, then start:

   ```powershell
   docker compose up -d
   docker compose logs -f core
   ```

7. Check the container without starting Discord:

   ```powershell
   docker compose run --rm core python -m harness.container_health
   ```

The Codex login state lives in `RA_CODEX_HOME_DIR`; rebuilding the image does
not erase it.

## OpenAI Responses fallback

Install/runtime image support is already included. Set an explicit model that
is available to the account:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=...
OPENAI_REASONING_EFFORT=
```

The API key is consumed by the parent provider and is not forwarded to child
CLI processes unless it is deliberately added to `AGENT_ENV_ALLOWLIST`.

API fallback is suitable for:

- PLANNING dialogue
- main plan and integration
- independent review
- fresh hypotheses

It is not used for local workspace writes.

## Optional Computer Use

Computer Use is off by default and is not part of the default provider chain.
It requires all of the following:

```env
OPENAI_COMPUTER_ENABLED=true
OPENAI_COMPUTER_MODEL=...
OPENAI_COMPUTER_BRIDGE_URL=https://bridge.example
OPENAI_COMPUTER_ALLOWED_STAGES=planning_dialogue
OPENAI_COMPUTER_REQUIRE_APPROVAL=true
```

The bridge must implement:

```text
POST /v1/actions
```

Request fields include the research session, stage, call ID, and requested
computer action. The response must contain a screenshot data URL:

```json
{
  "screenshot_data_url": "data:image/png;base64,..."
}
```

Before a GUI session starts, the provider emits an `APPROVAL_REQUIRED` event.
Pending safety checks returned by the model also require a separate harness
approval. Disabling this requirement is not recommended for an unattended
bot.

The bridge is deliberately separate from the core container. A Windows browser
bridge can be used while the same ARM64 core runs on Jetson, so migration does
not require source changes.

## Jetson migration

On Jetson Linux:

```bash
git clone https://github.com/SORALNV/ResearchAgent.git
cd ResearchAgent
cp .env.example .env
mkdir -p runtime research_runs codex-home
sudo chown -R 10001:10001 runtime research_runs codex-home
docker compose build
docker compose run --rm core python -m harness.container_health
docker compose up -d
```

The official Codex installer provides a Linux ARM64 binary. `TARGETARCH` is
available during the build, but no architecture-specific filename is hardcoded
in this repository.

Jetson can therefore run the complete control plane. The 9700X host remains the
recommended primary system for concurrent Agent calls, large archives, and
multiple projects; Jetson is a valid low-power deployment or standby target.

## Multi-architecture build verification

CI builds both target architectures without downloading Codex:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --build-arg INSTALL_CODEX=false \
  .
```

The production build keeps `INSTALL_CODEX=true` unless the deployment is API
only.

## Security boundaries

- Discord credentials remain in the parent service and are hard-denied to child
  Agent processes.
- The OpenAI API key is not in the default child-process allowlist.
- The root filesystem is read-only in Compose; only runtime, research archive,
  Codex state, `/tmp`, and the container home are writable.
- Linux capabilities are dropped and `no-new-privileges` is enabled.
- Generic CLI execution remains denied unless a compatible sandbox policy is
  explicitly enabled.
- Computer Use requires a separate bridge, stage allowlist, and approval gate.
