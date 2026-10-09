ARG UBUNTU_VERSION=24.04
ARG PY_VERSION=3.12
ARG NVM_VERSION=0.40.2
ARG NODE_MAJOR_VERSION=22
ARG OPENCODE_VERSION=1.18.29
ARG KERNEL_ENV_PATH=/opt/envs/bmb

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

## What every trial would otherwise install into its container's tmpfs: the venv Harbor's bootstrap
## makes for its exec server, and the plugin package OpenCode installs next to its config, pinned
## to OpenCode's own version as OpenCode pins it.
RUN --mount=type=cache,target=/root/.npm,sharing=locked,id=npm-cache-${TARGETARCH} \
    source "${NVM_DIR}/nvm.sh" && \
    python3 -m venv /opt/harbor-server && \
    /opt/harbor-server/bin/pip install --no-cache-dir uvicorn fastapi && \
    mkdir -p /root/.config/opencode && \
    npm install --prefix /root/.config/opencode --save-exact --no-audit --no-fund \
        "@opencode-ai/plugin@$(opencode --version)"

ARG KERNEL_ENV_PATH
ARG PY_VERSION
##
# See https://huggingface.co/datasets/Anthropic/BioMysteryBench-full/discussions/5#6a2874c38bd704ff625d0781 for dependencies.
#
# Generate/update lockfile using:
#  uvx conda-lock lock --micromamba -p linux-64 -p linux-aarch64 \
#   -f bio.yaml -f graphics.yaml -f py.yaml -f r.yaml -f runtime.yaml \
#   --lockfile env.lock --without-cuda
#
RUN --mount=type=cache,target=/root/.mamba/pkgs,sharing=locked,id=mamba-cache-${TARGETARCH} \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked,id=uv-cache-${TARGETARCH} \
    --mount=type=bind,source=conda-envs,target=/conda-envs \
    micromamba create -p "${KERNEL_ENV_PATH}" "python=${PY_VERSION}" pip -c conda-forge -y && \
    uvx conda-lock install --micromamba -p "${KERNEL_ENV_PATH}" /conda-envs/env.lock && \
    micromamba clean -ay

## Setup PATH to environment bin
ENV PATH="${KERNEL_ENV_PATH}/bin:${PATH}" LD_LIBRARY_PATH="${KERNEL_ENV_PATH}/lib:${LD_LIBRARY_PATH}"

WORKDIR /app
VOLUME ["/app"]
