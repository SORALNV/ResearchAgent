# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG TARGETARCH
ARG INSTALL_CODEX=true

LABEL org.opencontainers.image.title="ResearchAgent"
LABEL org.opencontainers.image.description="Portable Discord research/Kaggle control plane and compute worker"
LABEL org.opencontainers.image.source="https://github.com/SORALNV/ResearchAgent"
LABEL io.researchagent.targetarch="${TARGETARCH}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap \
        ca-certificates \
        curl \
        git \
        tini \
    && rm -rf /var/lib/apt/lists/*

# The official Codex installer selects the Linux amd64 or arm64 standalone
# binary. Set INSTALL_CODEX=false for API-only, Worker, or CI images.
RUN if [ "${INSTALL_CODEX}" = "true" ]; then \
        curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/install-codex.sh; \
        HOME=/root sh /tmp/install-codex.sh; \
        CODEX_BIN="$(command -v codex)"; \
        test -n "${CODEX_BIN}" && test -x "${CODEX_BIN}"; \
        CODEX_REAL_BIN="$(readlink -f "${CODEX_BIN}")"; \
        CODE_MODE_HOST_BIN="$(dirname "${CODEX_REAL_BIN}")/codex-code-mode-host"; \
        test -x "${CODE_MODE_HOST_BIN}"; \
        if [ "${CODEX_BIN}" != "/usr/local/bin/codex" ]; then \
            install -m 0755 "${CODEX_BIN}" /usr/local/bin/codex; \
        fi; \
        install -m 0755 "${CODE_MODE_HOST_BIN}" /usr/local/bin/codex-code-mode-host; \
        test -x /usr/local/bin/codex-code-mode-host; \
        rm -f /tmp/install-codex.sh; \
    fi

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -e '.[runtime]' \
    && kaggle --help >/dev/null

RUN useradd --create-home --uid 10001 --shell /bin/sh researchagent \
    && mkdir -p /data/runtime /data/research_runs /data/codex /data/worker \
    && chown -R researchagent:researchagent /app /data \
    && install -m 0755 /app/deploy/entrypoint.sh /usr/local/bin/research-agent-entrypoint

ENV HOME=/home/researchagent \
    PROJECT_ROOT=/data/runtime \
    RESEARCH_ARCHIVE_DIR=/data/research_runs \
    CODEX_HOME=/data/codex \
    AGENT_SANDBOX_BACKEND=none \
    AGENT_ALLOW_UNSANDBOXED_GENERIC=false

VOLUME ["/data/runtime", "/data/research_runs", "/data/codex", "/data/worker"]
USER researchagent

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-m", "harness.container_health"]

STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/research-agent-entrypoint"]
CMD ["python", "main.py", "--workdir", "/data/runtime", "--research-archive-dir", "/data/research_runs", "bot"]
