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
#   Test image from a specific git commit (runs as a non-root user):
#     docker buildx build -f docker/Dockerfile --build-arg GYM_GIT_REF=<commit> \
#       --target test --load --tag nemo-gym:test .
#     docker run --rm --entrypoint bash nemo-gym:test -lc \
#       'PYTEST_ADDOPTS="-m \"not sandbox\" --cov-report= --cov-fail-under=0" ng_dev_test'
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

ARG NEMO_GYM_UID=10001
ARG NEMO_GYM_GID=10001

RUN <<"EOF" bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y --no-install-recommends \
    jq \
    curl \
    git \
    rsync \
    wget \
    less \
    vim

groupadd --gid "${NEMO_GYM_GID}" nemo-gym
useradd --uid "${NEMO_GYM_UID}" --gid "${NEMO_GYM_GID}" --create-home --shell /bin/bash nemo-gym

apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

ARG UV_VERSION=0.11.29
ARG PYTHON_VERSION=3.13.14
ENV HOME=/home/nemo-gym \
    UV_CACHE_DIR=/opt/uv/cache \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

ENV RAY_USAGE_STATS_ENABLED=0


FROM base AS hermetic

WORKDIR /opt/nemo-gym

ARG BASE_IMAGE
ARG UV_VERSION

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_gym_venv
ENV UV_LINK_MODE=copy

# Copy only dependency metadata first for layer caching.
COPY --from=nemo-gym pyproject.toml uv.lock ./
COPY --from=nemo-gym nemo_gym/__init__.py nemo_gym/package_info.py ./nemo_gym/

RUN uv venv --seed && \
    uv sync --link-mode symlink --locked --extra vllm --no-install-project

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

# Copy full source. Exclude pyproject.toml and uv.lock since they are
# already present from the hermetic stage.
COPY --from=nemo-gym --exclude=pyproject.toml --exclude=uv.lock . /opt/nemo-gym

# Install the nemo-gym project itself along with the vllm extra.
RUN UV_LINK_MODE=symlink uv sync --locked --extra vllm

# Optional: pre-warm per-server venvs at build time so they are ready at runtime.
# Pass --build-arg NEMO_GYM_PREFETCH_CONFIGS="path/to/config1.yaml,path/to/config2.yaml"
# Uses 'gym env prefetch' which accepts the same config format as 'gym env start',
# installs each server's venv serially, and exits without starting any server process.
RUN <<"EOF" bash -exu
if [[ -n "${NEMO_GYM_PREFETCH_CONFIGS:-}" ]]; then
    gym env prefetch "+config_paths=[${NEMO_GYM_PREFETCH_CONFIGS}]"
fi
EOF

RUN chown -R nemo-gym:nemo-gym /home/nemo-gym /opt/nemo-gym /opt/nemo_gym_venv /opt/uv

USER nemo-gym
ENTRYPOINT ["gym"]
CMD ["--help"]


FROM release AS test

USER root
RUN uv sync --link-mode copy --locked --extra dev --extra sandbox --no-extra vllm && \
    uv cache clean && \
    chown -R nemo-gym:nemo-gym /home/nemo-gym /opt/nemo-gym /opt/nemo_gym_venv

USER nemo-gym
RUN <<"EOF" bash -euxo pipefail
test "$(id -u)" -ne 0
PYTEST_ADDOPTS='-m "not sandbox" --cov-report= --cov-fail-under=0' ng_dev_test
pytest --cov=. --cov-append --cov-report=term-missing --durations=10 -m sandbox
EOF

ENTRYPOINT []
CMD ["bash"]


# Keep the runtime image as the default output when no target is specified.
FROM release AS final
