#!/usr/bin/env bash
#
# Derived from utopia-group/VeriSoftBench scripts/build_repo.sh at commit
# d066523f2b087f28b793c05eacb29bf2e42c0504 (MIT License, Copyright (c) 2026 VeriSoftBench
# Authors). NVIDIA modifications (Apache-2.0): the "Pre-seed lake packages" step below,
# which fetches every Lake dependency at its pinned revision by SHA before `lake build`.
# Everything else is upstream's, unchanged.
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
        cp "$f" "${REPO_OUT}/$(basename "$f")"
        echo "  Copied: $(basename "$f")"
    fi
done

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

# For mathlib repos, fetch cached oleans first
if [ "${HAS_MATHLIB}" = true ]; then
    echo "Fetching mathlib cache..."
    lake exe cache get || {
        echo "WARNING: lake exe cache get failed (may not be available for this mathlib version)"
    }
fi

# --- Pre-seed lake packages at their pinned revisions (NeMo Gym addition) ---
# Lake clones each dependency and runs `git checkout <rev>`. When upstream's pinned
# revision is no longer reachable from any branch of the dependency (observed for
# quote4 9371548b, required by iris-lean: the `stable` branch was rewritten), that
# checkout fails with git exit 128 and the whole image build dies. Fetching the
# revision by SHA first keeps the pin exact and lets Lake find the package in place.
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
    dest = pkgs_dir / pkg["name"]
    if dest.exists():
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  seeding {pkg['name']} @ {pkg['rev'][:12]} from {pkg['url']}", flush=True)
    subprocess.run(["git", "clone", "--quiet", pkg["url"], str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "origin", pkg["rev"]], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", pkg["rev"]], check=True)
PYSEED
fi

echo "Running: lake build"
lake build

echo "SUCCESS: ${REPO_NAME} built successfully"
echo ""
