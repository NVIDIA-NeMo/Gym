#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is derived from utopia-group/VeriSoftBench `scripts/build_repo.sh`
# (https://github.com/utopia-group/VeriSoftBench) at commit
# d066523f2b087f28b793c05eacb29bf2e42c0504 and remains under its original MIT
# terms (reproduced below). Modifications Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES and contributors, licensed under the Apache License 2.0 (SPDX header
# above). NVIDIA modifications: the "Pre-seed lake packages" step below, which puts
# every Lake dependency — newly cloned or already present — at its pinned
# lake-manifest revision before `lake build`; the optional per-repo
# `nemo-gym-build.json` overrides (fallback and extra build targets a repository
# needs beyond upstream's protocol); and the post-build revision verification,
# which fails the build when any package ends off its pinned revision.
# Everything else is upstream's, unchanged.
#
# MIT License
#
# Copyright (c) 2026 VeriSoftBench Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# build_repo.sh — Build a single Lean repository for VeriSoftBench evaluation.
#
# Usage (local source):
#   ./build_repo.sh <repo_name> <source_dir> <config_dir> <output_dir> [--skip-if-built]
#
# Usage (clone from GitHub):
#   ./build_repo.sh <repo_name> --clone <config_dir> <output_dir> [--skip-if-built]
#
# Arguments:
#   repo_name   — Name of the repository (e.g., ArkLib)
#   source_dir  — Directory containing repo source code (BuildScripts/LeanSrc)
#                  OR --clone to clone from GitHub using repos.json
#   config_dir  — Directory containing build configs (BuildScripts/build_config)
#   output_dir  — Target directory for built repos (e.g., data/lean_repos)
#   --skip-if-built  — Skip if .lake/ directory already exists in output
#
# This script:
#   1. Copies repo source to output_dir/<repo_name>/ (or clones from GitHub)
#   2. Overlays build config files (lean-toolchain, lakefile, lake-manifest.json)
#   3. Installs the required Lean toolchain via elan
#   4. Runs `lake exe cache get` for mathlib-dependent repos
#   5. Runs `lake build`
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOS_JSON="${SCRIPT_DIR}/../repos.json"

# --- Argument parsing ---
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <repo_name> <source_dir|--clone> <config_dir> <output_dir> [--skip-if-built]"
    exit 1
fi

REPO_NAME="$1"
SOURCE_DIR="$2"
CONFIG_DIR="$3"
OUTPUT_DIR="$4"
SKIP_IF_BUILT=false
CLONE_MODE=false

if [ "${SOURCE_DIR}" = "--clone" ]; then
    CLONE_MODE=true
fi

shift 4
while [ "$#" -gt 0 ]; do
    case "$1" in
        --skip-if-built) SKIP_IF_BUILT=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

REPO_CFG="${CONFIG_DIR}/${REPO_NAME}"
REPO_OUT="${OUTPUT_DIR}/${REPO_NAME}"

echo "========================================"
echo "Building: ${REPO_NAME}"
echo "========================================"

# --- Validate inputs ---
if [ "${CLONE_MODE}" = false ]; then
    REPO_SRC="${SOURCE_DIR}/${REPO_NAME}"
    if [ ! -d "${REPO_SRC}" ]; then
        echo "ERROR: Source directory not found: ${REPO_SRC}"
        exit 1
    fi
fi

if [ ! -d "${REPO_CFG}" ]; then
    echo "ERROR: Config directory not found: ${REPO_CFG}"
    exit 1
fi

# --- Skip if already built ---
if [ "${SKIP_IF_BUILT}" = true ] && [ -d "${REPO_OUT}/.lake" ]; then
    echo "SKIP: ${REPO_NAME} already built (${REPO_OUT}/.lake exists)"
    exit 0
fi

# --- Get source (clone or copy) ---
if [ "${CLONE_MODE}" = true ]; then
    # Read URL and commit from repos.json
    if [ ! -f "${REPOS_JSON}" ]; then
        echo "ERROR: repos.json not found at ${REPOS_JSON}"
        exit 1
    fi
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: python3 is required for --clone mode (JSON parsing)"
        exit 1
    fi

    REPO_URL=$(python3 -c "
import json
with open('${REPOS_JSON}') as f:
    data = json.load(f)
repo = data['repos'].get('${REPO_NAME}')
if not repo:
    raise SystemExit('ERROR: ${REPO_NAME} not found in repos.json')
print(repo['url'])
")
    REPO_COMMIT=$(python3 -c "
import json
with open('${REPOS_JSON}') as f:
    data = json.load(f)
print(data['repos']['${REPO_NAME}']['commit'])
")

    echo "Cloning ${REPO_URL} at commit ${REPO_COMMIT}..."
    rm -rf "${REPO_OUT}"
    git clone --quiet "${REPO_URL}" "${REPO_OUT}"
    cd "${REPO_OUT}"
    git checkout --quiet "${REPO_COMMIT}"
    # Remove .git to save space (matches Docker build behavior)
    rm -rf "${REPO_OUT}/.git"
    cd - > /dev/null
else
    echo "Copying source from ${REPO_SRC} to ${REPO_OUT}..."
    mkdir -p "${REPO_OUT}"
    # Use rsync if available, fall back to cp
    if command -v rsync &> /dev/null; then
        rsync -a --exclude='.git' --exclude='.lake' "${REPO_SRC}/" "${REPO_OUT}/"
    else
        cp -r "${REPO_SRC}/." "${REPO_OUT}/"
        rm -rf "${REPO_OUT}/.git" "${REPO_OUT}/.lake"
    fi
fi

# --- Overlay build config files ---
echo "Overlaying build config from ${REPO_CFG}..."
for f in "${REPO_CFG}"/*; do
    if [ -f "$f" ]; then
        # nemo-gym-build.json is a build override for this script, not a repo file.
        if [ "$(basename "$f")" = "nemo-gym-build.json" ]; then
            continue
        fi
        cp "$f" "${REPO_OUT}/$(basename "$f")"
        echo "  Copied: $(basename "$f")"
    fi
done

# --- Per-repo build overrides (NeMo Gym addition) ---
# An optional ${REPO_CFG}/nemo-gym-build.json documents repairs a repository needs
# beyond upstream's protocol: `try_cache_fallback` retries a failed `lake build`
# with `--try-cache`, and `extra_targets` lists targets `lake build` does not reach
# on its own but verification requires. A `reason` field records why.
BUILD_OVERRIDES="${REPO_CFG}/nemo-gym-build.json"
TRY_CACHE_FALLBACK=false
EXTRA_TARGETS=""
if [ -f "${BUILD_OVERRIDES}" ]; then
    TRY_CACHE_FALLBACK=$(python3 -c "
import json
with open('${BUILD_OVERRIDES}') as f:
    cfg = json.load(f)
print('true' if cfg.get('try_cache_fallback') else 'false')
")
    EXTRA_TARGETS=$(python3 -c "
import json
with open('${BUILD_OVERRIDES}') as f:
    cfg = json.load(f)
print(' '.join(cfg.get('extra_targets', [])))
")
    echo "Build overrides: try_cache_fallback=${TRY_CACHE_FALLBACK}, extra_targets=[${EXTRA_TARGETS}]"
fi

# --- Resolve lakefile conflicts ---
# Lake prefers lakefile.lean over lakefile.toml; remove the other
# to ensure the overlaid config takes effect.
if [ -f "${REPO_CFG}/lakefile.toml" ] && [ -f "${REPO_OUT}/lakefile.lean" ]; then
    rm -f "${REPO_OUT}/lakefile.lean"
    echo "  Removed conflicting lakefile.lean (using lakefile.toml from config)"
elif [ -f "${REPO_CFG}/lakefile.lean" ] && [ -f "${REPO_OUT}/lakefile.toml" ]; then
    rm -f "${REPO_OUT}/lakefile.toml"
    echo "  Removed conflicting lakefile.toml (using lakefile.lean from config)"
fi

# --- Read lean-toolchain ---
if [ ! -f "${REPO_OUT}/lean-toolchain" ]; then
    echo "ERROR: lean-toolchain not found in ${REPO_OUT}"
    exit 1
fi
TOOLCHAIN=$(cat "${REPO_OUT}/lean-toolchain" | tr -d '[:space:]')
echo "Lean toolchain: ${TOOLCHAIN}"

# --- Install toolchain via elan ---
if command -v elan &> /dev/null; then
    echo "Installing toolchain: ${TOOLCHAIN}"
    elan toolchain install "${TOOLCHAIN}" || {
        echo "WARNING: elan install failed, toolchain may already be installed"
    }
else
    echo "WARNING: elan not found. Assuming Lean toolchain is already installed."
fi

# --- Determine if mathlib repo (check for mathlib in lake-manifest.json) ---
HAS_MATHLIB=false
if [ -f "${REPO_OUT}/lake-manifest.json" ]; then
    if grep -q '"mathlib"' "${REPO_OUT}/lake-manifest.json" 2>/dev/null; then
        HAS_MATHLIB=true
    fi
fi

# --- Build ---
cd "${REPO_OUT}"

# --- Pre-seed lake packages at their pinned revisions (NeMo Gym addition) ---
# Lake clones each dependency and runs `git checkout <rev>`. When upstream's pinned
# revision is no longer reachable from any branch of the dependency (observed for
# quote4 9371548b, required by iris-lean: the `stable` branch was rewritten), that
# checkout fails with git exit 128 and the whole image build dies. Fetching the
# revision by SHA first keeps the pin exact and lets Lake find the package in place.
# A package directory that already exists (e.g. left by an earlier cache or build
# attempt) is re-pinned the same way rather than trusted (observed for Cli in
# lean-mlir, left on the wrong revision before anything compiled). This runs before
# `lake exe cache get` so that cache resolution starts from the pinned revisions.
if [ -f "${REPO_OUT}/lake-manifest.json" ]; then
    echo "Pre-seeding lake packages from lake-manifest.json..."
    python3 - "${REPO_OUT}" <<'PYSEED'
import json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "lake-manifest.json").read_text())
pkgs_dir = root / manifest.get("packagesDir", ".lake/packages")
for pkg in manifest.get("packages", []):
    if pkg.get("type") != "git":
        continue
    rev = pkg["rev"]
    dest = pkgs_dir / pkg["name"]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  seeding {pkg['name']} @ {rev[:12]} from {pkg['url']}", flush=True)
        subprocess.run(["git", "clone", "--quiet", pkg["url"], str(dest)], check=True)
    elif not (dest / ".git").exists():
        raise SystemExit(f"ERROR: {dest} exists without .git; cannot pin {pkg['name']} to {rev}")
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != rev:
        print(f"  pinning existing {pkg['name']}: {head[:12]} -> {rev[:12]}", flush=True)
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "origin", rev], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", rev], check=True)
PYSEED
fi

# For mathlib repos, fetch cached oleans first
if [ "${HAS_MATHLIB}" = true ]; then
    echo "Fetching mathlib cache..."
    lake exe cache get || {
        echo "WARNING: lake exe cache get failed (may not be available for this mathlib version)"
    }
fi

echo "Running: lake build"
if ! lake build; then
    if [ "${TRY_CACHE_FALLBACK}" = true ]; then
        echo "lake build failed; retrying with --try-cache (declared in nemo-gym-build.json)"
        lake build --try-cache
    else
        echo "ERROR: lake build failed for ${REPO_NAME}"
        exit 1
    fi
fi

if [ -n "${EXTRA_TARGETS}" ]; then
    echo "Building extra targets: ${EXTRA_TARGETS}"
    # shellcheck disable=SC2086  # EXTRA_TARGETS is a space-separated target list
    lake build ${EXTRA_TARGETS}
fi

# --- Make every manifest revision checkout-able for `lake env` (NeMo Gym addition) ---
# `lake build` may leave a transitive package at a revision other than the one the
# top-level manifest pins (observed for Cli in lean-mlir). At verification time
# `lake env lean` then tries `git checkout <rev>` and dies with git exit 128 because
# the revision was never fetched. Fetch and check out each pinned revision, then
# verify every package really ends on it; a package left off its pin fails the build
# instead of being reported as success.
if [ -f "${REPO_OUT}/lake-manifest.json" ]; then
    python3 - "${REPO_OUT}" <<'PYFETCH'
import json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "lake-manifest.json").read_text())
pkgs_dir = root / manifest.get("packagesDir", ".lake/packages")

def head_of(dest):
    return subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

checked = 0
failed = []
for pkg in manifest.get("packages", []):
    rev = pkg.get("rev")
    dest = pkgs_dir / pkg["name"]
    if pkg.get("type") != "git" or not rev or not (dest / ".git").exists():
        continue
    checked += 1
    if head_of(dest) != rev:
        print(f"  {pkg['name']}: at {head_of(dest)[:8]}, manifest wants {rev[:8]}; fetching and checking out", flush=True)
        fetch = subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "origin", rev])
        checkout = subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", rev])
        if fetch.returncode != 0 or checkout.returncode != 0 or head_of(dest) != rev:
            failed.append(pkg["name"])
if failed:
    raise SystemExit(f"ERROR: packages left off their manifest revisions: {', '.join(failed)}")
print(f"  verified {checked} packages at their manifest revisions", flush=True)
PYFETCH
fi

echo "SUCCESS: ${REPO_NAME} built successfully"
echo ""
