# syntax=docker/dockerfile:1.7

FROM node:22-bookworm-slim AS codex
ARG CODEX_NPM_PACKAGE=@openai/codex
RUN npm install --global "${CODEX_NPM_PACKAGE}"

FROM python:3.12-slim-bookworm AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      tini \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-platform.txt /tmp/requirements-platform.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-platform.txt
COPY . /app
RUN python -m pip install --no-deps -e . \
    && useradd --create-home --uid 10001 --shell /bin/bash agent \
    && mkdir -p /var/lib/research-agent \
    && chown -R agent:agent /var/lib/research-agent /app
ENTRYPOINT ["/usr/bin/tini", "--"]

FROM python-base AS edge
USER agent
CMD ["python", "-m", "harness.platform.discord_edge"]

FROM python-base AS worker
USER agent
ENV RESEARCH_WORKER_DATA_DIR=/var/lib/research-agent/worker
VOLUME ["/var/lib/research-agent"]
EXPOSE 8090
CMD ["python", "-m", "harness.compute.worker_api"]

FROM python-base AS core
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      bubblewrap \
    && python -m pip install "kaggle>=1.7,<2" \
    && rm -rf /var/lib/apt/lists/*
COPY --from=codex /usr/local/bin/node /usr/local/bin/node
COPY --from=codex /usr/local/bin/npm /usr/local/bin/npm
COPY --from=codex /usr/local/bin/npx /usr/local/bin/npx
COPY --from=codex /usr/local/bin/codex /usr/local/bin/codex
COPY --from=codex /usr/local/lib/node_modules /usr/local/lib/node_modules
ARG INSTALL_COMPUTER_USE=false
RUN if [ "${INSTALL_COMPUTER_USE}" = "true" ]; then \
      python -m pip install "playwright>=1.50,<2" && \
      python -m playwright install --with-deps chromium; \
    fi
USER agent
ENV RESEARCH_AGENT_DATA_DIR=/var/lib/research-agent/core \
    AGENT_ENV_ALLOWLIST=OPENAI_API_KEY \
    AGENT_SANDBOX_BACKEND=auto \
    AGENT_ALLOW_UNSANDBOXED_GENERIC=false
VOLUME ["/var/lib/research-agent"]
EXPOSE 8080
CMD ["uvicorn", "harness.platform.asgi_portable:app", "--host", "0.0.0.0", "--port", "8080"]
