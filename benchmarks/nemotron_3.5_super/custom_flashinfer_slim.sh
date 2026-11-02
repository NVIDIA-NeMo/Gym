#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install a same-release FlashInfer branch and rebuild selected host dispatchers.
# Keep the base image's cubin and JIT-cache packages; no previous build manifest
# is needed. The caller must select every native module affected by their changes.
# Requires the base image's CUDA toolkit, git, and Python build dependencies.
#
# SLURM_ACCOUNT=... BASE_IMAGE=/lustre/base.sqsh FLASHINFER_BRANCH=... \
#   FLASHINFER_REBUILD_MODULES=fused_moe_trtllm_sm100,fmha_gen \
#   bash custom_flashinfer_slim.sh /lustre/output.sqsh
# Supported modules: fused_moe_trtllm_sm100 (default), fmha_gen.
# Prims CuTe-DSL kernels still compile on first use; this script builds native
# dispatchers only. No runtime dependencies are installed or upgraded.
set -Eeuo pipefail

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/bxyu-nvidia/flashinfer.git}"
export FLASHINFER_REBUILD_MODULES="${FLASHINFER_REBUILD_MODULES:-fused_moe_trtllm_sm100}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.0a}"
export MAX_JOBS="${MAX_JOBS:-64}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"

if [[ "${1:-}" == __inside_build ]]; then
    : "${FLASHINFER_HEAD_SHA:?}" "${BASE_IMAGE:?}"
    for executable in git nvcc c++ ninja; do
        command -v "$executable" >/dev/null || { echo "Missing build tool: $executable" >&2; exit 1; }
    done
    BUILD_DIR=$(mktemp -d /tmp/flashinfer-slim.XXXXXX)
    export BUILD_DIR
    trap 'cd /; rm -rf "$BUILD_DIR"' EXIT
    export FLASHINFER_WORKSPACE_BASE="$BUILD_DIR/workspace"
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
PY
    export FLASHINFER_LOCAL_VERSION="$(cat "$BUILD_DIR/local-version")"
    unset FLASHINFER_DEV_RELEASE_SUFFIX
    export FLASHINFER_BUILD_NO_PIP=1
    export BUILD_NVEP=0 BUILD_NIXL_EP=0 BUILD_NCCL_EP=0
    # Python 2 (pip): replace only flashinfer-python with this branch. Use the
    # image's build dependencies (--no-build-isolation) and leave runtime
    # dependencies, including cubins/providers, installed as-is (--no-deps).
    python3 -m pip install --no-deps --no-build-isolation --force-reinstall "$BUILD_DIR/source"

    # Write a helper for the two Python calls below; this heredoc does not run it.
    # Both modes check the source commit, retained package versions, and expected
    # cubin manifest. Build mode compiles; verify mode only loads saved binaries.
    cat > "$BUILD_DIR/modules.py" <<'PY'
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import shutil
import sys
from pathlib import Path

from flashinfer._build_meta import __git_commit__
from flashinfer.artifacts import ArtifactPath, CheckSumHash
from flashinfer.jit import env
from flashinfer.jit.core import build_jit_specs

# Extend this registry with zero-argument JIT spec generators as needed.
generators = {
    "fused_moe_trtllm_sm100": ("flashinfer.jit.fused_moe", "gen_trtllm_gen_fused_moe_sm100_module", "TRTLLM_GEN_BMM"),
    "fmha_gen": ("flashinfer.jit.attention.modules", "gen_trtllm_gen_fmha_module", "TRTLLM_GEN_FMHA"),
}
root = Path(os.environ["BUILD_DIR"])
verify = sys.argv[1] == "verify"
if __git_commit__ != os.environ["FLASHINFER_HEAD_SHA"]:
    raise RuntimeError("Installed FlashInfer source commit differs from requested commit")
for package, version in json.loads((root / "binary-packages.json").read_text()).items():
    if metadata.version(package) != version:
        raise RuntimeError(f"Unexpected binary package change: {package}")
if not verify and (root / "aot").exists():
    shutil.copytree(root / "aot", env.FLASHINFER_AOT_DIR, dirs_exist_ok=True)
records = json.loads((root / "modules.json").read_text()) if verify else {}
for name in dict.fromkeys(os.environ["FLASHINFER_REBUILD_MODULES"].split(",")):
    name = name.strip()
    module, function, artifact_name = generators[name]
    artifact = getattr(ArtifactPath, artifact_name)
    manifest = env.FLASHINFER_CUBIN_DIR / artifact / "checksums.txt"
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != getattr(CheckSumHash, artifact_name):
        raise RuntimeError(f"Installed cubins do not match {artifact_name} expected by this branch: {manifest}. "
                           "Install compatible cubins or use the full build script.")
    spec = getattr(importlib.import_module(module), function)()
    if spec.name != name:
        raise RuntimeError(f"Generator returned {spec.name}, expected {name}")
    destination = env.FLASHINFER_AOT_DIR / name / f"{name}.so"
    if verify:
        if spec.aot_path != destination or hashlib.sha256(destination.read_bytes()).hexdigest() != records[name]:
            raise RuntimeError(f"Rebuilt module is missing, changed, or shadowed: {name}")
        spec.build_and_load()
        print(f"Loaded rebuilt module with JIT disabled: {destination}", flush=True)
    else:
        build_jit_specs([spec], verbose=True, skip_prebuilt=False)
        compiled = env.FLASHINFER_JIT_DIR / name / f"{name}.so"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compiled, destination)
        records[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
if not verify:
    (root / "modules.json").write_text(json.dumps(records, indent=2) + "\n")
PY
    # Python 3: restore AOT overrides backed up before pip, compile only selected
    # modules even if prebuilt versions exist, and install their .so files as AOT
    # overrides. Save their SHA256 hashes for the subsequent verification call.
    python3 "$BUILD_DIR/modules.py" build
    rm -rf "$FLASHINFER_WORKSPACE_BASE"
    # Python 4: in a fresh process with compilation disabled and its workspace
    # deleted, check that module resolution selects the saved .so files, verify
    # their hashes, and load them. This is a load check, not a GPU correctness test.
    FLASHINFER_DISABLE_JIT=1 python3 "$BUILD_DIR/modules.py" verify
    mkdir -p /opt/super-vl-evals
    # Python 5: write flashinfer-slim.json into the saved image, recording the
    # base image, source SHA, target architecture, rebuilt module hashes, and
    # retained binary-package versions. This call performs no compilation.
    python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BUILD_DIR"])
record = {key: os.environ[key] for key in
          ("BASE_IMAGE", "FLASHINFER_HEAD_SHA", "FLASHINFER_CUDA_ARCH_LIST", "FLASHINFER_REBUILD_MODULES")}
record["modules"] = json.loads((root / "modules.json").read_text())
record["retained_packages"] = json.loads((root / "binary-packages.json").read_text())
Path("/opt/super-vl-evals/flashinfer-slim.json").write_text(json.dumps(record, indent=2) + "\n")
PY
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
IFS=',' read -r -a modules <<< "$FLASHINFER_REBUILD_MODULES"
for module in "${modules[@]}"; do
    case "$module" in
        fused_moe_trtllm_sm100|fmha_gen) ;;
        *) echo "Unsupported module: $module. Choose fused_moe_trtllm_sm100 and/or fmha_gen." >&2; exit 2 ;;
    esac
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
echo "Building $FLASHINFER_BRANCH @ $FLASHINFER_HEAD_SHA; modules: $FLASHINFER_REBUILD_MODULES"
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
