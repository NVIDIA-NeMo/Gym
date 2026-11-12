ARG UBUNTU_VERSION=24.04
ARG NVM_VERSION=0.40.2
ARG NODE_MAJOR_VERSION=22
ARG OPENCODE_VERSION=latest

FROM ubuntu:${UBUNTU_VERSION}

ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1 PYTHONPATH=""

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETARCH} \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked,id=apt-lists-${TARGETARCH} \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        asciinema \
        curl \
        git \
        python3 \
        python3-venv \
        python3-pip \
        r-base \
        r-base-dev \
        tmux \
        wget

ARG NVM_VERSION
ARG NODE_MAJOR_VERSION
ENV NVM_DIR=/root/.nvm
RUN --mount=type=cache,target=/root/.nvm/.cache,sharing=locked,id=nvm-cache-${TARGETARCH} \
    --mount=type=cache,target=/root/.npm,sharing=locked,id=npm-cache-${TARGETARCH} \
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v${NVM_VERSION}/install.sh | bash && \
    source "${NVM_DIR}/nvm.sh" && \
    nvm install "${NODE_MAJOR_VERSION}" && \
    ln -sf "$(command -v node)" /usr/local/bin/node && \
    ln -sf "$(command -v npm)" /usr/local/bin/npm && \
    ln -sf "$(command -v npx)" /usr/local/bin/npx

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV MAMBA_ROOT_PREFIX=/root/.mamba
COPY --from=mambaorg/micromamba:latest /usr/bin/micromamba /bin/
RUN micromamba config set repodata_use_zst true  ## Enable faster downloads

ARG OPENCODE_VERSION
RUN --mount=type=cache,target=/root/.npm,sharing=locked,id=npm-cache-${TARGETARCH} \
    source "${NVM_DIR}/nvm.sh" && \
    npm i -g "opencode-ai@${OPENCODE_VERSION}"

WORKDIR /app
VOLUME [ "/app" ]
