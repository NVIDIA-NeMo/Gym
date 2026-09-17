# syntax=docker/dockerfile:1.7
# OpenCode + STEM runtime for OpenSandbox; linux/amd64. Inputs are pinned by version and SHA-256.
FROM ghcr.io/astral-sh/uv:0.9.7@sha256:68d029b921923c7234ef0178e669d6d253c8356153f1b54316df8cc3e15f5c01 AS uv
FROM python:3.10-bookworm@sha256:d1112dbfeb8ebaea80d9e7d914de5c8e9f21b5a565829fb667c07ee6437e1759

ARG DEBIAN_SNAPSHOT=20260912T000000Z
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s/ bookworm main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s/ bookworm-security main\n' "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      bash git curl ripgrep gfortran pkg-config libgmp-dev libmpfr-dev libmpc-dev \
      libgl1 libglib2.0-0 libsndfile1 libenchant-2-2 libgeos-dev libglpk-dev \
      libsnappy-dev poppler-utils tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 MPLBACKEND=Agg \
    OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_SHARE=1 OPENCODE_DISABLE_MODELS_FETCH=true
RUN mkdir -p /opt/image-provenance && \
    curl -fL --retry 3 https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/bcf059af55c20a89f797724598f9908d126153e6/requirements/sandbox.lock -o /opt/image-provenance/requirements.lock && \
    echo 'aa25fd3c0ce7f6c6bdd73da8002ceaaa3b0a411e211dac74566f67ed1d8e6e82  /opt/image-provenance/requirements.lock' | sha256sum -c -
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /opt/image-provenance/requirements.lock \
      --extra-index-url https://download.pytorch.org/whl/cpu

RUN curl -fL --retry 3 https://github.com/anomalyco/opencode/releases/download/v1.17.11/opencode-linux-x64.tar.gz -o /tmp/opencode.tar.gz && \
    tar -xzf /tmp/opencode.tar.gz -C /usr/local/bin opencode && rm /tmp/opencode.tar.gz
RUN echo '0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded  /usr/local/bin/opencode' | sha256sum -c - && \
    chmod 0755 /usr/local/bin/opencode
RUN uv pip freeze --system > /opt/image-provenance/python-installed.txt && \
    dpkg-query -W > /opt/image-provenance/debian-installed.txt

USER root
WORKDIR /workspace
CMD ["sleep", "infinity"]
