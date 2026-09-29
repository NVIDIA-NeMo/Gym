# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Kimina Lean Server on Lean + Mathlib, for CombiBench. linux/amd64.
#
# Upstream CombiBench verifies through Kimina Lean Server, so this image carries the same
# three things its harness talks to: a Lean toolchain, a built Mathlib, and the REPL that
# Kimina pools. One image per Lean version; build args select it and versions.py holds the
# pins. See README.md.
#
# This is NOT upstream Kimina's Dockerfile. That one elan-installs Lean and clones Mathlib
# at a branch name with nothing verified, which is not enough to say what a score was
# measured against. The structure here follows lean_proof/lean_image/Dockerfile: checksum
# the toolchain, pin Mathlib by commit, assert the commit's own lean-toolchain matches, and
# do the last build offline so an incomplete cache fails the build instead of a rollout.
FROM debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251 AS base

ARG DEBIAN_SNAPSHOT=20260916T000000Z
# http, not https: this image ships no CA bundle and ca-certificates is installed below.
# apt's GPG signature check is what protects integrity.
#
# APT::Sandbox::User=root: apt drops privileges to `_apt` to fetch, which needs a user
# namespace. Builders without one (buildah under chroot isolation) fail with "Could not
# switch group".
#
# libatomic1 is not incidental: `prisma generate` downloads a prebuilt node, which links
# against it. Upstream Kimina gets it transitively from build-essential; this image pulls
# the one library rather than a compiler toolchain it does not otherwise need.
RUN printf 'APT::Sandbox::User "root";\n' > /etc/apt/apt.conf.d/00-no-sandbox && \
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/%s/ bookworm main\ndeb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/%s/ bookworm-security main\n' "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      bash git curl ca-certificates zstd libgmp-dev jq \
      python3 python3-venv procps libatomic1 && \
    rm -rf /var/lib/apt/lists/*

FROM base AS lean
# LEAN_VERSION must be the toolchain the Mathlib commit's own lean-toolchain names; the
# build asserts that below rather than trusting it. Same for the REPL.
ARG LEAN_VERSION=v4.24.0
ARG LEAN_SHA256
ARG MATHLIB_COMMIT
ARG REPL_COMMIT

RUN test -n "$LEAN_SHA256" || { echo "LEAN_SHA256 is required; see README.md" >&2; exit 1; } && \
    test -n "$MATHLIB_COMMIT" || { echo "MATHLIB_COMMIT is required; see README.md" >&2; exit 1; } && \
    test -n "$REPL_COMMIT" || { echo "REPL_COMMIT is required; see README.md" >&2; exit 1; }

RUN curl -fL --retry 3 "https://github.com/leanprover/lean4/releases/download/${LEAN_VERSION}/lean-${LEAN_VERSION#v}-linux.tar.zst" -o /tmp/lean.tar.zst && \
    echo "${LEAN_SHA256}  /tmp/lean.tar.zst" | sha256sum -c - && \
    mkdir -p /opt/lean && tar --zstd -xf /tmp/lean.tar.zst -C /opt/lean --strip-components=1 && \
    rm /tmp/lean.tar.zst
ENV PATH=/opt/lean/bin:$PATH

RUN git init /opt/mathlib && cd /opt/mathlib && \
    git remote add origin https://github.com/leanprover-community/mathlib4.git && \
    git fetch --depth 1 origin "$MATHLIB_COMMIT" && git checkout --detach FETCH_HEAD && \
    test "$(cat lean-toolchain)" = "leanprover/lean4:${LEAN_VERSION}" && \
    lake exe cache get && \
    mkdir -p /opt/image-provenance && \
    git rev-parse HEAD > /opt/image-provenance/mathlib-commit.txt && \
    cp lean-toolchain lake-manifest.json /opt/image-provenance/ && \
    rm -rf /root/.cache/mathlib

# Upstream Kimina's setup.sh does exactly this rewrite, and the REPL needs it: it runs as
# `lake env` inside the project, and lake re-resolves the manifest's git dependencies over
# the network unless they are declared as already-unpacked paths. Without it the REPL's
# first start fails in a container that has no network, or silently refetches in one that
# does.
RUN cd /opt/mathlib && \
    jq '.packages |= map(.type="path"|del(.url)|.dir=".lake/packages/"+.name)' \
       lake-manifest.json > lake-manifest.json.tmp && \
    mv lake-manifest.json.tmp lake-manifest.json

# The REPL is the process Kimina actually pools. It must be built with the same toolchain
# as Mathlib or it cannot load Mathlib's oleans.
RUN git init /opt/repl && cd /opt/repl && \
    git remote add origin https://github.com/leanprover-community/repl.git && \
    git fetch --depth 1 origin "$REPL_COMMIT" && git checkout --detach FETCH_HEAD && \
    test "$(cat lean-toolchain)" = "leanprover/lean4:${LEAN_VERSION}" && \
    git rev-parse HEAD > /opt/image-provenance/repl-commit.txt && \
    lake build

# Offline: an incomplete Mathlib cache fails here, not mid-rollout.
RUN --network=none cd /opt/mathlib && lake build Mathlib

FROM base AS server
ARG KIMINA_COMMIT
RUN test -n "$KIMINA_COMMIT" || { echo "KIMINA_COMMIT is required; see README.md" >&2; exit 1; }

# Debian's python is externally managed (PEP 668), so the server gets its own venv rather
# than --break-system-packages.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN git init /opt/kimina && cd /opt/kimina && \
    git remote add origin https://github.com/project-numina/kimina-lean-server.git && \
    git fetch --depth 1 origin "$KIMINA_COMMIT" && git checkout --detach FETCH_HEAD && \
    mkdir -p /opt/image-provenance && git rev-parse HEAD > /opt/image-provenance/kimina-commit.txt

# Upstream resolves the server extra through uv and installs the pinned set; doing the same
# keeps this image's Python dependencies the ones the project tests against.
# `prisma generate` is not optional: server/prisma_client.py imports the generated client
# unconditionally, even though the proof log it backs is disabled here.
#
# --no-hashes: `uv export` writes hashes, which puts pip into --require-hashes mode, and
# that mode requires *every* requirement to be `==`-pinned. One transitive dependency
# (google-api-core[grpc], via google-cloud-appengine-logging) is specified as a range, so
# the install aborts. Versions stay pinned exactly as uv.lock resolved them either way;
# only per-file hash verification is given up.
RUN pip install --no-cache-dir uv && \
    cd /opt/kimina && \
    uv export --extra server --no-dev --no-emit-project --no-hashes > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e . && \
    prisma generate

FROM base AS runtime
# ARGs do not cross stages; this one is only used by the probe at the end.
ARG LEAN_VERSION=v4.24.0

# This container compiles untrusted model output, and `native_decide` is allowed by design,
# so that output can run compiled code in here. Doing it as uid 0 is the one avoidable part
# of that. The user is created before the COPYs so the two trees Lake writes into can be
# copied with --chown, which costs nothing: `chown -R` afterwards would duplicate every
# file of a multi-gigabyte Mathlib in a new layer.
RUN groupadd --gid 1000 lean && useradd --uid 1000 --gid 1000 --create-home lean

COPY --from=lean /opt/lean /opt/lean
# Writable by `lean`: the REPL runs as `lake env` with its cwd here, and Lake writes under
# .lake. /opt/lean, /opt/venv and the provenance files are read-and-execute only.
COPY --from=lean --chown=lean:lean /opt/mathlib /opt/mathlib
COPY --from=lean --chown=lean:lean /opt/repl /opt/repl
COPY --from=lean /opt/image-provenance/ /opt/image-provenance/
COPY --from=server /opt/venv /opt/venv
COPY --from=server --chown=lean:lean /opt/kimina /opt/kimina
COPY --from=server /opt/image-provenance/ /opt/image-provenance/

ENV HOME=/home/lean
ENV PATH=/opt/venv/bin:/opt/lean/bin:$PATH
ENV LEAN_SERVER_HOST=0.0.0.0 \
    LEAN_SERVER_PORT=8000 \
    LEAN_SERVER_REPL_PATH=/opt/repl/.lake/build/bin/repl \
    LEAN_SERVER_PROJECT_DIR=/opt/mathlib \
    LEAN_SERVER_MAX_REPLS=8 \
    LEAN_SERVER_MAX_REPL_USES=-1 \
    LEAN_SERVER_MAX_REPL_MEM=12G \
    LEAN_SERVER_MAX_WAIT=60 \
    LEAN_SERVER_INIT_REPLS={} \
    LEAN_SERVER_DATABASE_URL=

# ENV PATH is image config, which does not survive conversion to squashfs for enroot -- the
# files arrive, the config does not. Symlinks are filesystem, so they do. The LEAN_SERVER_*
# values above are lost the same way, which is why the Slurm scripts pass them explicitly;
# the symlinks are what keep `lake` findable for the REPL subprocess regardless.
RUN ln -sf /opt/lean/bin/lake /usr/local/bin/lake && \
    ln -sf /opt/lean/bin/lean /usr/local/bin/lean && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python

# Everything below this line runs as the unprivileged user, so the build-time probe is also
# the check that the server's own work — starting a REPL, loading Mathlib, importing the
# generated prisma client — still succeeds without root. A build that passes is the
# validation; there is no separate claim to take on trust.
USER lean

# Start a REPL the way the server does -- `lake env <repl>` with cwd at the project -- and
# require it to load Mathlib and report the Lean version. This is the check that turns a
# broken image into a failed build: the server answers /health long before it can start a
# REPL, so without this the first symptom is every rollout returning "Failed to start REPL".
RUN cd /opt/mathlib && \
    printf '{"cmd": "import Mathlib\\n#eval Lean.versionString"}\n\n' | \
      lake env /opt/repl/.lake/build/bin/repl | tee /tmp/probe.json && \
    grep -q "${LEAN_VERSION#v}" /tmp/probe.json && \
    rm /tmp/probe.json

# The server imports the generated prisma client unconditionally, so an import failure here
# is a build failure rather than a container that answers /health and nothing else.
RUN cd /opt/kimina && python -c "import server; print('server imports')"

WORKDIR /opt/kimina
CMD ["python", "-m", "server"]
