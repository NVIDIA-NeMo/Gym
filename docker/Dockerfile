# syntax=docker/dockerfile:1
# Usage:
#   Self-contained build (default: builds from main):
#     docker buildx build -f docker/Dockerfile --tag <registry>/nemo-gym:latest --push .
#
#   Self-contained build (specific git ref):
#     docker buildx build -f docker/Dockerfile --build-arg GYM_GIT_REF=v0.4.0 \
#       --tag <registry>/nemo-gym:v0.4.0 --push .
#
#   Local source override:
#     docker buildx build --build-context nemo-gym=. -f docker/Dockerfile \
#       --tag <registry>/nemo-gym:latest --push .
#
#   Non-root runtime override (the image still defaults to root):
#     docker run --rm --user 65532:65532 \
#       --env HOME=/home/nemo-runtime \
#       --env XDG_CACHE_HOME=/opt/nemo-gym/cache/xdg \
#       <registry>/nemo-gym:latest --help
#

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04

# Source stage: clones Gym from GitHub. Override with --build-context nemo-gym=.
# to inject a local checkout instead.
FROM scratch AS nemo-gym
ARG GYM_GIT_REF=main
ADD --keep-git-dir=true https://github.com/NVIDIA-NeMo/Gym.git#${GYM_GIT_REF} /

FROM ${BASE_IMAGE} AS base
ENV NEMO_GYM_CONTAINER=1
USER root

ARG ENROOT_VERSION=3.5.0
RUN <<"EOF" bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y --no-install-recommends \
    bash \
    jq \
    curl \
    git \
    wget \
    less \
    vim

apt-get purge -y \
    ffmpeg \
    libavcodec* \
    libavformat* \
    libavutil* \
    libswscale* \
    libswresample* \
    libx264* \
    libx265* \
    libfdk-aac* \
    libmp3lame* \
    2>/dev/null || true

apt-get autoremove -y

# Install enroot for the EnrootProvider sandbox backend.
# GitHub releases provide signed .debs for amd64 and arm64.
arch=$(dpkg --print-architecture)
curl -fSsL -o /tmp/enroot.deb \
    "https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/enroot_${ENROOT_VERSION}-1_${arch}.deb"
apt-get install -y /tmp/enroot.deb
rm /tmp/enroot.deb

apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

ARG UV_VERSION=0.11.29
ARG PYTHON_VERSION=3.13.14
ARG TARGETARCH
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache
RUN case "${TARGETARCH}" in \
        amd64) \
            uv_arch="x86_64"; \
# pragma: allowlist nextline secret
            uv_sha256="04f8b82f5d47f0512dcd32c67a4a6f16a0ea27c81537c338fd0ad6b23cebe829"; \
            ;; \
        arm64) \
            uv_arch="aarch64"; \
# pragma: allowlist nextline secret
            uv_sha256="94500fb064ae3c971a873cba64d94694c50677e0a4dbf78735c80509e7429919"; \
            ;; \
        *) \
            echo "Unsupported TARGETARCH for uv: ${TARGETARCH}" >&2; \
            exit 1; \
            ;; \
    esac && \
    uv_archive="uv-${uv_arch}-unknown-linux-gnu.tar.gz" && \
    curl -fLSs -o "/tmp/${uv_archive}" \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${uv_archive}" && \
    echo "${uv_sha256}  /tmp/${uv_archive}" | sha256sum -c - && \
    tar -xzf "/tmp/${uv_archive}" -C /tmp && \
    install -m 0755 "/tmp/uv-${uv_arch}-unknown-linux-gnu/uv" /usr/local/bin/uv && \
    install -m 0755 "/tmp/uv-${uv_arch}-unknown-linux-gnu/uvx" /usr/local/bin/uvx && \
    rm -rf "/tmp/${uv_archive}" "/tmp/uv-${uv_arch}-unknown-linux-gnu" && \
    /usr/local/bin/uv python install ${PYTHON_VERSION}
ENV PATH="/usr/local/bin:$PATH"

ARG RUNTIME_UID=65532
ARG RUNTIME_GID=65532
RUN if ! getent group "${RUNTIME_GID}" >/dev/null; then \
        groupadd --gid "${RUNTIME_GID}" nemo-runtime; \
    fi && \
    if ! getent passwd "${RUNTIME_UID}" >/dev/null; then \
        useradd --no-log-init --uid "${RUNTIME_UID}" --gid "${RUNTIME_GID}" \
            --create-home --home-dir /home/nemo-runtime --shell /bin/bash nemo-runtime; \
    fi && \
    install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" \
        /home/nemo-runtime \
        /opt/nemo-gym \
        /opt/nemo-gym/cache \
        /opt/nemo-gym/results \
        /opt/uv/cache

ENV RAY_USAGE_STATS_ENABLED=0


FROM base AS hermetic

WORKDIR /opt/nemo-gym

ARG BASE_IMAGE
ARG UV_VERSION
ARG RUNTIME_UID
ARG RUNTIME_GID

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_gym_venv
ENV UV_LINK_MODE=copy

# Copy only dependency metadata first for layer caching.
COPY --from=nemo-gym --chown=${RUNTIME_UID}:${RUNTIME_GID} pyproject.toml uv.lock ./
COPY --from=nemo-gym --chown=${RUNTIME_UID}:${RUNTIME_GID} \
    nemo_gym/__init__.py nemo_gym/package_info.py ./nemo_gym/

# The telemetry extra is included even though telemetry is off by default. It is a few MB
# of pure-Python packages, and without it in the image nobody can switch tracing on with
# an environment variable — they would have to rebuild. Off remains the default: the code
# is inert until NEMO_LENS_ENABLED=1.
RUN uv venv --seed && \
    uv sync --link-mode symlink --locked --extra vllm --extra telemetry --no-install-project

ENV PATH="/opt/nemo_gym_venv/bin:$PATH"


FROM hermetic AS release

ARG NEMO_GYM_PREFETCH_CONFIGS=""
ARG NEMO_GYM_COMMIT
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ARG RC_DATE=00.00
ARG TARGETARCH
ENV NEMO_GYM_COMMIT=${NEMO_GYM_COMMIT:-<unknown>}
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

# Copy full source as root while the package build introspects the preserved
# Git checkout. Exclude pyproject.toml and uv.lock since they are already
# present from the hermetic stage. Runtime ownership is applied after install.
COPY --from=nemo-gym --exclude=pyproject.toml --exclude=uv.lock . /opt/nemo-gym

# Install the nemo-gym project itself along with the vllm and telemetry extras. The
# worktree was created for the runtime UID in the base stage, so restore build ownership
# before setuptools invokes Git.
RUN chown root:root /opt/nemo-gym && \
    UV_LINK_MODE=symlink uv sync --locked --extra vllm --extra telemetry

# Keep runtime uv and XDG caches separate from the root-owned build cache that
# backs the symlinked project environment.
ENV UV_CACHE_DIR=/opt/nemo-gym/cache/uv
RUN install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" \
        "${UV_CACHE_DIR}" \
        /opt/nemo-gym/cache/xdg

# Optional: pre-warm per-server venvs at build time so they are ready at runtime.
# Pass --build-arg NEMO_GYM_PREFETCH_CONFIGS="path/to/config1.yaml,path/to/config2.yaml"
# Uses 'gym env prefetch' which accepts the same config format as 'gym env start',
# installs each server's venv serially, and exits without starting any server process.
RUN <<"EOF" bash -exu
if [[ -n "${NEMO_GYM_PREFETCH_CONFIGS:-}" ]]; then
    gym env prefetch "+config_paths=[${NEMO_GYM_PREFETCH_CONFIGS}]"
fi

# Keep ownership changes for generated caches and server venvs in the layer
# that creates them so later layers do not duplicate their contents.
chown -R "${RUNTIME_UID}:${RUNTIME_GID}" /opt/nemo-gym/cache
find /opt/nemo-gym -type d -name .venv -prune \
    -exec chown -R "${RUNTIME_UID}:${RUNTIME_GID}" {} +
EOF

# Own source and project-environment directories so the runtime identity can
# mutate their contents without copying prefetched artifacts into this layer.
RUN venv_site_packages="$(/opt/nemo_gym_venv/bin/python -c \
        'import site; print(site.getsitepackages()[0])')" && \
    touch "${venv_site_packages}/.nonroot-uninstall-probe" && \
    find /opt/nemo-gym \
        -path /opt/nemo-gym/cache -prune -o \
        -type d -name .venv -prune -o \
        -type d -exec chown "${RUNTIME_UID}:${RUNTIME_GID}" {} + && \
    find /opt/nemo_gym_venv -type d \
        -exec chown "${RUNTIME_UID}:${RUNTIME_GID}" {} +

USER ${RUNTIME_UID}:${RUNTIME_GID}

# Runtime callers select this UID and set HOME/cache variables explicitly; the
# published image remains root by default. Prove that the CLI, uv toolchain,
# project environment, source tree, and runtime paths support that contract.
RUN export HOME=/home/nemo-runtime \
        XDG_CACHE_HOME=/opt/nemo-gym/cache/xdg && \
    test "$(id -u)" = "${RUNTIME_UID}" && \
    test -w "${HOME}" && \
    test -w "${XDG_CACHE_HOME}" && \
    test -w "${UV_CACHE_DIR}" && \
    test -w /opt/nemo-gym/cache && \
    test -w /opt/nemo-gym/results && \
    test -x /usr/local/bin/uv && \
    test -x /usr/local/bin/uvx && \
    test -x /opt/nemo_gym_venv/bin/python && \
    venv_site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')" && \
    rm "${venv_site_packages}/.nonroot-uninstall-probe" && \
    touch "${HOME}/.nonroot-write-probe" \
        "${XDG_CACHE_HOME}/.nonroot-write-probe" \
        "${UV_CACHE_DIR}/.nonroot-write-probe" \
        /opt/nemo-gym/.nonroot-write-probe \
        /opt/nemo-gym/cache/.nonroot-write-probe \
        /opt/nemo-gym/results/.nonroot-write-probe \
        "${venv_site_packages}/.nonroot-write-probe" && \
    gym --help >/dev/null && \
    uv venv /opt/nemo-gym/cache/container-smoke && \
    /opt/nemo-gym/cache/container-smoke/bin/python -c \
        "import sys; assert sys.version_info[:2] == (3, 13)" && \
    rm -r /opt/nemo-gym/cache/container-smoke \
        "${HOME}/.nonroot-write-probe" \
        "${XDG_CACHE_HOME}/.nonroot-write-probe" \
        "${UV_CACHE_DIR}/.nonroot-write-probe" \
        /opt/nemo-gym/.nonroot-write-probe \
        /opt/nemo-gym/cache/.nonroot-write-probe \
        /opt/nemo-gym/results/.nonroot-write-probe \
        "${venv_site_packages}/.nonroot-write-probe"

USER root
ENTRYPOINT ["gym"]
CMD ["--help"]
