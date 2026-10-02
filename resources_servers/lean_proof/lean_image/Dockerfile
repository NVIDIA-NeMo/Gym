# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lean + Mathlib for the Gym Lean benchmarks, linux/amd64.
#
# One image per Mathlib version; build args select it, and versions.py holds the pins.
# See README.md.
FROM debian:bookworm-slim@sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251 AS base

ARG DEBIAN_SNAPSHOT=20260916T000000Z
# http, not https: this image ships no CA bundle and ca-certificates is installed below.
# apt's GPG signature check is what protects integrity.
#
# APT::Sandbox::User=root: apt drops privileges to `_apt` to fetch, which needs a user
# namespace. Builders without one (buildah under chroot isolation) fail with "Could not
# switch group".
RUN printf 'APT::Sandbox::User "root";\n' > /etc/apt/apt.conf.d/00-no-sandbox && \
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/%s/ bookworm main\ndeb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/%s/ bookworm-security main\n' "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      bash git curl ca-certificates zstd libgmp-dev && \
    rm -rf /var/lib/apt/lists/*

FROM base AS lean
# LEAN_VERSION must be the toolchain the Mathlib commit's own lean-toolchain names; the
# build asserts that below rather than trusting it.
ARG LEAN_VERSION=v4.19.0
ARG LEAN_SHA256
ARG MATHLIB_COMMIT

RUN test -n "$LEAN_SHA256" || { echo "LEAN_SHA256 is required; see README.md" >&2; exit 1; } && \
    test -n "$MATHLIB_COMMIT" || { echo "MATHLIB_COMMIT is required; see README.md" >&2; exit 1; }

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

# Offline: an incomplete cache fails here, not mid-rollout.
RUN --network=none cd /opt/mathlib && lake build Mathlib

FROM base AS runtime
COPY --from=lean /opt/lean /opt/lean
COPY --from=lean /opt/mathlib /opt/mathlib
COPY --from=lean /opt/image-provenance/ /opt/image-provenance/
ENV PATH=/opt/lean/bin:$PATH

# ENV PATH is image config, which does not survive conversion to squashfs for enroot --
# the files arrive, the config does not. Symlinks are filesystem, so they do.
RUN ln -sf /opt/lean/bin/lake /usr/local/bin/lake && \
    ln -sf /opt/lean/bin/lean /usr/local/bin/lean

# Compiles run as this user (see LeanSandbox.compile), so where the provider can switch uid a
# submission's `#eval` can neither rewrite the toolchain nor write outside /tmp. The build cannot
# test that by running as `lean`: its namespace has one uid, so `su` here keeps uid 0 (as under
# enroot). It asserts the permissions that make it true instead.
RUN useradd --system --create-home --shell /bin/sh lean && \
    test -z "$(find /opt/lean /opt/mathlib ! -type l -perm /022)" && \
    test -z "$(find /opt/lean /opt/mathlib ! -user root)"

# `lake env` asks git about each package. Git refuses root-owned repos for another user ("dubious
# ownership"), so lake concludes the package URL changed and tries to delete and re-clone it -- which
# the read-only toolchain rightly refuses, failing every compile. The repos are ours: trust them.
RUN git config --system --add safe.directory '*'

# Compile as the verifier will: `lake env lean` in the project, as `lean`, under `env -i` with
# no PATH. Setting PATH here would hide the failure this exists to catch.
RUN printf 'import Mathlib\n#eval Lean.versionString\n' > /tmp/smoke.lean && \
    su -s /bin/sh lean -c 'cd /opt/mathlib && env -i sh -c "lake env lean /tmp/smoke.lean"' && \
    rm /tmp/smoke.lean

WORKDIR /opt/mathlib

# Ignored under enroot, which drops image config; LeanSandbox.compile names the user anyway.
USER lean

CMD ["sleep", "infinity"]
