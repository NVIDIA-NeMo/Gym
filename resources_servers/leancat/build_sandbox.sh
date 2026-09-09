#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build a NeMo-Skills Lean 4 sandbox pinned to Mathlib v4.19.0, which is what LeanCat
# statements are written against.
#
# The stock NeMo-Skills sandbox pins v4.12.0 in two places (the elan toolchain install and
# the mathlib `require` line). LeanCat's `CategoryTheory` statements do not compile against
# it, and the failures look like ordinary "unknown identifier" compile errors rather than
# like a misconfiguration -- so a run against the stock image produces a plausible, wrong,
# near-zero score. This script rewrites those pins rather than vendoring a copy of their
# Dockerfile, so upstream fixes keep flowing through.
#
# Expect hours, not minutes: `lake exe cache get` pulls prebuilt Mathlib oleans when the tag
# matches, and falls back to compiling Mathlib from source when it does not.
#
# Usage:
#   ./build_sandbox.sh [output_dir]
#
# Then point the server at it:
#   export NEMO_SKILLS_SANDBOX_HOST=<node> NEMO_SKILLS_SANDBOX_PORT=6000

set -euo pipefail

LEAN_VERSION="v4.19.0"
IMAGE_TAG="leancat-sandbox:mathlib-${LEAN_VERSION}"
OUTPUT_DIR="${1:-$(pwd)}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> Cloning NeMo-Skills into $WORKDIR"
git clone --depth 1 https://github.com/NVIDIA-NeMo/Skills.git "$WORKDIR/Skills"
cd "$WORKDIR/Skills"

DOCKERFILE="dockerfiles/Dockerfile.sandbox"
test -f "$DOCKERFILE" || { echo "ERROR: $DOCKERFILE not found; upstream layout changed." >&2; exit 1; }

# Fail loudly if the pin we expect to rewrite is not there, rather than silently building
# a sandbox on whatever version upstream moved to.
if ! grep -q 'v4\.12\.0' "$DOCKERFILE"; then
    echo "ERROR: expected 'v4.12.0' pins in $DOCKERFILE and found none." >&2
    echo "       Upstream changed its pin; check what it is now before editing this script." >&2
    exit 1
fi

echo "==> Repinning Lean and Mathlib to ${LEAN_VERSION}"
sed -i "s/v4\.12\.0/${LEAN_VERSION}/g" "$DOCKERFILE"
grep -n "${LEAN_VERSION}" "$DOCKERFILE"

echo "==> Building ${IMAGE_TAG} (this takes hours if the Mathlib cache misses)"
docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" .

SQSH="${OUTPUT_DIR}/leancat-sandbox-mathlib-${LEAN_VERSION}.sqsh"
echo "==> Exporting to ${SQSH} for pyxis/enroot"
enroot import -o "$SQSH" "dockerd://${IMAGE_TAG}"

echo
echo "Done: $SQSH"
echo
echo "Launch it into an existing allocation alongside the eval, then set"
echo "NEMO_SKILLS_SANDBOX_HOST to that node:"
echo
echo "  srun --overlap --container-image=$SQSH --container-name=leancat-sandbox \\"
echo "       --nodes=1 --ntasks=1 /start-with-nginx.sh &"
echo
echo "Verify before trusting any score -- all 100 reference statements must compile with"
echo "only a 'declaration uses sorry' warning:"
echo "  python resources_servers/leancat/check_sandbox.py"
