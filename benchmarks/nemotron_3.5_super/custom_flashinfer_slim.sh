#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install a same-release FlashInfer branch and invalidate selected native modules.
# Keep the base image's cubin and JIT-cache packages; no previous build manifest
# is needed. Select every native module affected by your changes; vLLM compiles
# those modules when they are first used.
# Requires git and Python build dependencies; vLLM warmup needs the CUDA toolkit.
#
# SLURM_ACCOUNT=... BASE_IMAGE=/lustre/base.sqsh FLASHINFER_BRANCH=... \
#   FLASHINFER_INVALIDATE_MODULES='fused_moe_trtllm_sm100,checkpointing_ssu_*,replayssm_materialize_*' \
#   bash custom_flashinfer_slim.sh /lustre/output.sqsh
# Accepts comma-separated module names/globs (default: fused_moe_trtllm_sm100).
# FLASHINFER_REBUILD_MODULES is accepted as a compatibility alias.
# No kernels are compiled here; use SAVE_WARMED_CONTAINER in the vLLM launcher
# to bake startup compilation results. Runtime dependencies remain installed.
set -Eeuo pipefail

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/bxyu-nvidia/flashinfer.git}"
export FLASHINFER_INVALIDATE_MODULES="${FLASHINFER_INVALIDATE_MODULES:-${FLASHINFER_REBUILD_MODULES:-fused_moe_trtllm_sm100}}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.0a}"

if [[ "${1:-}" == __inside_build ]]; then
    : "${FLASHINFER_HEAD_SHA:?}" "${BASE_IMAGE:?}"
    command -v git >/dev/null || { echo "Missing build tool: git" >&2; exit 1; }
    BUILD_DIR=$(mktemp -d /tmp/flashinfer-slim.XXXXXX)
    export BUILD_DIR
    trap 'cd /; rm -rf "$BUILD_DIR"' EXIT
    unset FLASHINFER_DISABLE_JIT FLASHINFER_DISABLE_VERSION_CHECK FLASHINFER_CUBIN_DIR
    git init "$BUILD_DIR/source"
    git -C "$BUILD_DIR/source" remote add origin "$FLASHINFER_REPO"
    git -C "$BUILD_DIR/source" fetch --depth=1 origin "$FLASHINFER_HEAD_SHA"
    git -C "$BUILD_DIR/source" checkout --detach FETCH_HEAD
    git -C "$BUILD_DIR/source" submodule update --init --recursive --depth=1

    # Python 1: check that the branch and installed Python/cubin packages share
    # a release. Save the version suffix, back up existing AOT overrides, and
    # record binary-package versions for comparison after installation.
    # Run outside the checkout so imports refer to the base image's package.
    cd /
    python3 - <<'PY'
import importlib.metadata as metadata
import json
import os
import shutil
from pathlib import Path

import flashinfer
import flashinfer_cubin
from flashinfer.jit import env

root = Path(os.environ["BUILD_DIR"])
release = (root / "source/version.txt").read_text().strip()
version = flashinfer.__version__
if version != flashinfer_cubin.__version__ or version.partition("+")[0] != release:
    raise RuntimeError(f"Slim builds require matching releases: branch={release}, Python={version}, "
                       f"cubins={flashinfer_cubin.__version__}")
(root / "local-version").write_text(version.partition("+")[2])
if env.FLASHINFER_AOT_DIR.exists():
    shutil.copytree(env.FLASHINFER_AOT_DIR, root / "aot")
packages = {dist.metadata["Name"]: dist.version for dist in metadata.distributions()
            if dist.metadata["Name"].lower().replace("_", "-").startswith("flashinfer-")
            and dist.metadata["Name"].lower().replace("_", "-") != "flashinfer-python"}
(root / "binary-packages.json").write_text(json.dumps(packages))
(root / "old-cache-root").write_text(str(env.FLASHINFER_CACHE_DIR))
PY
    export FLASHINFER_LOCAL_VERSION="$(cat "$BUILD_DIR/local-version")"
    unset FLASHINFER_DEV_RELEASE_SUFFIX
    export FLASHINFER_BUILD_NO_PIP=1
    export BUILD_NVEP=0 BUILD_NIXL_EP=0 BUILD_NCCL_EP=0
    # Python 2 (pip): replace only flashinfer-python with this branch. Use the
    # image's build dependencies (--no-build-isolation) and leave runtime
    # dependencies, including cubins/providers, installed as-is (--no-deps).
    python3 -m pip install --no-deps --no-build-isolation --force-reinstall "$BUILD_DIR/source"

    # Python 3: restore unrelated AOT overrides, remove matching module families
    # from package/provider/JIT caches, and update provider manifests. No spec
    # generators, shape lists, CUDA compilation, or kernel launches are needed.
    python3 - <<'PY'
import fnmatch
import importlib.metadata as metadata
import json
import os
import shutil
from pathlib import Path

from flashinfer._build_meta import __git_commit__
from flashinfer.jit import env

root = Path(os.environ["BUILD_DIR"])
if __git_commit__ != os.environ["FLASHINFER_HEAD_SHA"]:
    raise RuntimeError("Installed FlashInfer source commit differs from requested commit")
packages = json.loads((root / "binary-packages.json").read_text())
for package, version in packages.items():
    if metadata.version(package) != version:
        raise RuntimeError(f"Unexpected binary package change: {package}")
if (root / "aot").exists():
    shutil.copytree(root / "aot", env.FLASHINFER_AOT_DIR, dirs_exist_ok=True)
patterns = os.environ["FLASHINFER_INVALIDATE_MODULES"].split(",")


def selected(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


removed = []
# Scan all architectures, including cached specializations not used at startup.
module_roots = set(env.FLASHINFER_AOT_DIRS)
cache_roots = {Path((root / "old-cache-root").read_text()), env.FLASHINFER_CACHE_DIR,
               Path("/opt/super-vl-evals/flashinfer-warmup-workspace")}
for cache_root in cache_roots:
    module_roots.update(cache_root.glob("**/cached_ops"))
for module_root in sorted(module_roots):
    for directory in sorted(module_root.glob("*")):
        if selected(directory.name):
            if directory.is_symlink():
                directory.unlink()
            elif directory.is_dir():
                shutil.rmtree(directory)
            else:
                continue
            removed.append(str(directory))
            print(f"Invalidated {directory}", flush=True)
# Remove provider registry entries as well as binaries so missing entries fall
# through to JIT rather than advertising a prebuilt specialization.
for provider in env.FLASHINFER_AOT_PROVIDERS:
    manifest_path = provider.jit_cache_dir.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    remaining = [name for name in manifest["modules"] if not selected(name)]
    if remaining != manifest["modules"]:
        manifest["modules"] = remaining
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
record = {key: os.environ[key] for key in
          ("BASE_IMAGE", "FLASHINFER_HEAD_SHA", "FLASHINFER_CUDA_ARCH_LIST", "FLASHINFER_INVALIDATE_MODULES")}
record["invalidated_paths"] = removed
record["retained_packages"] = packages
(root / "flashinfer-slim.json").write_text(json.dumps(record, indent=2) + "\n")
print(f"Invalidated {len(removed)} native cache entries; selected modules will JIT on first use", flush=True)
PY
    mkdir -p /opt/super-vl-evals
    cp "$BUILD_DIR/flashinfer-slim.json" /opt/super-vl-evals/flashinfer-slim.json
    # These derived caches/plans describe the previous installation. Recreate
    # them during warmup instead of validating stale manifests or reusing graphs.
    rm -rf /opt/super-vl-evals/flashinfer-autotune-cache \
        /opt/super-vl-evals/vllm-cache /opt/super-vl-evals/prims-cute-cache
    # A previous full-build verifier describes the old Python sources/modules.
    # Keep it for reference, but don't let the launcher apply it to this build.
    if [[ -f /opt/super-vl-evals/flashinfer-build.env ]]; then
        mv /opt/super-vl-evals/flashinfer-build.env /opt/super-vl-evals/flashinfer-build.before-slim.env
    fi
    exit 0
fi

if [[ $# -ne 1 || -z "${SLURM_ACCOUNT:-}" || -z "${BASE_IMAGE:-}" || -z "${FLASHINFER_BRANCH:-}" ]]; then
    echo "Usage: SLURM_ACCOUNT=... BASE_IMAGE=... FLASHINFER_BRANCH=... $0 OUTPUT.sqsh" >&2
    exit 2
fi
# Keep patterns to module-name characters; never interpret them as file paths.
if [[ "$FLASHINFER_INVALIDATE_MODULES" == *, ]]; then
    echo "Empty module pattern" >&2; exit 2
fi
IFS=',' read -r -a modules <<< "$FLASHINFER_INVALIDATE_MODULES"
for module in "${modules[@]}"; do
    if [[ ! "$module" =~ ^[A-Za-z0-9_*?-]+$ ]]; then
        echo "Invalid module name/glob: $module" >&2; exit 2
    fi
done
export BASE_IMAGE FLASHINFER_BRANCH
mkdir -p "$(dirname "$1")"
OUT_SQSH="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
[[ ! -e "$OUT_SQSH" && ! -L "$OUT_SQSH" ]] || { echo "Output already exists: $OUT_SQSH" >&2; exit 2; }
FLASHINFER_HEAD_SHA=$(git ls-remote --exit-code "$FLASHINFER_REPO" "refs/heads/$FLASHINFER_BRANCH" | cut -f1)
[[ "$FLASHINFER_HEAD_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "Could not resolve branch" >&2; exit 2; }
if [[ -n "${EXPECTED_FLASHINFER_SHA:-}" && "$FLASHINFER_HEAD_SHA" != "$EXPECTED_FLASHINFER_SHA" ]]; then
    echo "Branch moved: expected $EXPECTED_FLASHINFER_SHA, got $FLASHINFER_HEAD_SHA" >&2
    exit 2
fi
export FLASHINFER_HEAD_SHA
SNAP=$(mktemp "$(dirname "$OUT_SQSH")/.flashinfer-slim.XXXXXX")
trap 'rm -f "$SNAP"' EXIT
cp "$0" "$SNAP"
MOUNTS="$(dirname "$SNAP"):$(dirname "$SNAP")"
[[ ! -d /lustre ]] || MOUNTS="$MOUNTS,/lustre:/lustre"
echo "Building $FLASHINFER_BRANCH @ $FLASHINFER_HEAD_SHA; invalidate: $FLASHINFER_INVALIDATE_MODULES"
if ! srun \
    --account="$SLURM_ACCOUNT" --partition="${SLURM_PARTITION:-batch}" --qos="${SLURM_QOS:-interactive}" \
    --job-name=flashinfer-slim --nodes=1 --ntasks=1 --segment=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" --gpus-per-node=4 --mem=0 --time="${SLURM_TIME:-01:00:00}" \
    --container-image="$BASE_IMAGE" --container-mounts="$MOUNTS" --no-container-mount-home \
    --container-save="$OUT_SQSH" --export=ALL bash "$SNAP" __inside_build; then
    rm -f "$OUT_SQSH"
    echo "Build failed; removed incomplete output" >&2
    exit 1
fi
sha256sum "$OUT_SQSH" | tee "$OUT_SQSH.sha256"
