#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install Lean 4 + Mathlib v4.19.0 into a plain directory, no container build required.
#
# Why not build an image: LeanCat needs Mathlib v4.19.0, and there is no published image at
# that pin (leanprover-community/mathlib ships only `latest`/`gitpod`/`debian`), while the
# NeMo-Skills sandbox pins v4.12.0 -- on which LeanCat's CategoryTheory statements fail with
# ordinary-looking "unknown identifier" errors, i.e. a plausible near-zero score that is not
# a model result. Building a correct image needs Docker, which HPC login nodes typically do
# not have.
#
# elan installs entirely in user space, so none of that is necessary. Bind-mount the
# resulting directory into whatever sandbox image you already have.
#
# `lake exe cache get` downloads prebuilt Mathlib oleans because v4.19.0 is a tagged
# release, so expect a large download rather than a multi-hour source build.
#
# Usage:
#   ./setup_lean.sh /lustre/<...>/lean4-mathlib-v4.19.0

set -euo pipefail

LEAN_VERSION="v4.19.0"
PREFIX="${1:?usage: setup_lean.sh <install-dir>}"

mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"

export ELAN_HOME="$PREFIX/elan"
export PATH="$ELAN_HOME/bin:$PATH"

if ! command -v elan >/dev/null 2>&1; then
    echo "==> Installing elan into $ELAN_HOME"
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
        | sh -s -- -y --no-modify-path --default-toolchain "leanprover/lean4:${LEAN_VERSION}"
fi

elan toolchain install "leanprover/lean4:${LEAN_VERSION}"

PROJECT="$PREFIX/my_project"
if [ ! -d "$PROJECT" ]; then
    echo "==> Creating Lean project at $PROJECT"
    cd "$PREFIX"
    lake +"leanprover/lean4:${LEAN_VERSION}" new my_project
fi

cd "$PROJECT"
echo "leanprover/lean4:${LEAN_VERSION}" > lean-toolchain
if ! grep -q "mathlib" lakefile.lean 2>/dev/null; then
    echo "require mathlib from git \"https://github.com/leanprover-community/mathlib4\" @ \"${LEAN_VERSION}\"" >> lakefile.lean
fi

echo "==> Fetching prebuilt Mathlib ${LEAN_VERSION} (download, not a source build)"
lake update
lake exe cache get
lake build

echo
echo "Done. Lean project: $PROJECT"
echo
echo "Point the sandbox at it -- mount the whole prefix and set lean_project_dir:"
echo
echo "  provider_options:"
echo "    mounts: [\"$PREFIX:/lean4\"]"
echo "  env:"
echo "    PATH: /lean4/elan/bin:/usr/local/bin:/usr/bin:/bin"
echo "  # resources_servers.leancat.lean_project_dir: /lean4/my_project"
echo
echo "Then verify before spending anything on inference:"
echo "  python resources_servers/leancat/check_sandbox.py"
