# External GPU Worker deployment

The Worker must not receive the Core `.env`. It needs only its own bearer token,
resource inventory, and optional NVIDIA runtime variables.

```bash
cp deploy/.env.worker.example deploy/.env.worker
# edit WORKER_TOKEN and the actual GPU/CPU/memory values

docker compose \
  --env-file deploy/.env.worker \
  -f deploy/compose.worker.yaml \
  -f deploy/compose.worker-gpu.yaml \
  up -d --build
```

The default bind address is `127.0.0.1:8090`. Reach it through Tailscale, SSH
port forwarding, or a private TLS reverse proxy. Do not expose a bearer-token
Worker directly to the public Internet.

Configure Core with a different `.env`:

```env
REMOTE_GPU_WORKER_NAME=remote_gpu
REMOTE_GPU_WORKER_URL=http://<private-worker-address>:8090
REMOTE_GPU_WORKER_TOKEN=<same-worker-token>
REMOTE_GPU_WORKER_PAID=false
REMOTE_GPU_WORKER_GPU_COUNT=1
REMOTE_GPU_WORKER_GPU_MEMORY_MB=24576
```

`REMOTE_GPU_WORKER_PAID=true` causes Core to stop the selected Job in
`waiting_approval` before sending source code to that Worker.

Worker source bundles and downloaded artifacts are bounded and SHA-256 checked.
The experiment subprocess receives an environment allowlist and cannot see the
Worker bearer token.
