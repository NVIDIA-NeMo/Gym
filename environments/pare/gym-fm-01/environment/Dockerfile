FROM docker:27.4.0-dind AS docker-tools
FROM ghcr.io/astral-sh/uv:0.12.23 AS uv-tools
FROM ubuntu:24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl git python3 python3-venv python3-pip python-is-python3 iptables \
    && rm -rf /var/lib/apt/lists/*
COPY --from=docker-tools /usr/local/bin/ /usr/local/bin/
COPY --from=uv-tools /uv /uvx /usr/local/bin/
COPY --chmod=755 start-docker.sh /usr/local/bin/start-docker.sh
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/start-docker.sh"]
