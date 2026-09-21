# Upstream ORAgentBench docker/base/Dockerfile at c9eb952435a4352f33daa2a35efe0f8c76d31b28, verbatim,
# plus one layer that installs tmux (required by Gym's Terminus 2 scaffold; upstream's own
# prebuilt-agent images likewise add scaffold tooling on top of this base).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV ORCLAW_SOLVE_TIME_LIMIT_SECONDS=300
ENV ORCLAW_SCIP_GAP=0.0005

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        networkx \
        numpy \
        pandas \
        pyomo \
        pyscipopt \
        scipy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends tmux procps && rm -rf /var/lib/apt/lists/*
