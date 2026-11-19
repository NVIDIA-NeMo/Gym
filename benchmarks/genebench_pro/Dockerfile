ARG UBUNTU_VERSION=24.04

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
        ca-certificates \
        curl \
        git \
        python3 \
        python3-venv \
        python3-pip \
        r-base \
        r-base-dev \
        ripgrep \
        tmux \
        wget

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked,id=uv-cache-${TARGETARCH} \
    uv venv "${VIRTUAL_ENV}" --python /usr/bin/python3 && \
    uv pip install numpy pandas scipy statsmodels scikit-learn matplotlib

ENV MAMBA_ROOT_PREFIX=/root/.mamba
COPY --from=mambaorg/micromamba:latest /usr/bin/micromamba /bin/
RUN micromamba config set repodata_use_zst true

WORKDIR /app
VOLUME [ "/app" ]
