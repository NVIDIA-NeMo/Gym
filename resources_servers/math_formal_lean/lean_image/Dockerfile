# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lean + Mathlib for the Gym Lean benchmarks, linux/amd64.
#
# One image per Mathlib version: compiled oleans do not carry across versions, and the
# benchmarks disagree (minif2f/proofnet/putnam_bench/mobench v4.12.0, leancat v4.19.0).
# Build args select the version; see README.md for the per-version values.
#
# Pinning follows responses_api_agents/opencode_sandboxed_agent/offline_science_image:
# the Lean tarball by sha256, Mathlib by commit, and the final `lake build` runs with
# --network=none so an incomplete cache fails the build instead of the benchmark.
FROM debian:bookworm-slim@sha256:40b107342c492725bc7aacbe93a49945445191ae58e9b2ef7fdb95d7a5b3ea3e AS base

ARG DEBIAN_SNAPSHOT=20260916T000000Z
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/%s/ bookworm main\ndeb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/%s/ bookworm-security main\n' "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      bash git curl ca-certificates zstd libgmp-dev && \
    rm -rf /var/lib/apt/lists/*

FROM base AS lean
# Defaults are leancat's pin. `LEAN_VERSION` must be the toolchain named by the Mathlib
# commit's own lean-toolchain file; the build asserts they agree rather than trusting it.
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

# Offline: proves the cache is complete at build time rather than at rollout time.
RUN --network=none cd /opt/mathlib && lake build Mathlib

FROM base AS runtime
COPY --from=lean /opt/lean /opt/lean
COPY --from=lean /opt/mathlib /opt/mathlib
COPY --from=lean /opt/image-provenance/ /opt/image-provenance/
ENV PATH=/opt/lean/bin:$PATH

# The verifier execs `lake env lean` through a non-login shell, which reads no profile, so
# the binaries have to resolve without one. Asserted here, not discovered mid-run.
RUN env -i /opt/lean/bin/lake --version && \
    env -i /opt/lean/bin/lean --version

WORKDIR /opt/mathlib
CMD ["sleep", "infinity"]
