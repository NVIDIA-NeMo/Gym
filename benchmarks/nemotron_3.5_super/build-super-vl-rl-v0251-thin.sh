#!/usr/bin/env bash
# Canonical Super VL RL v0.25.1 SQSH build for GB200/B200.
#
# Usage:
#   SLURM_ACCOUNT=<account> ./build-super-vl-rl-v0251-thin.sh /path/image.sqsh
#
# The build always resolves the latest heads of the team release branches,
# verifies the published TRTLLM-GEN cubin manifest, and rebuilds the small
# fmha_gen host dispatcher. vLLM must remain a non-editable install because its
# precompiled extensions rely on site-packages-relative RPATHs.
set -Eeuo pipefail

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.25.1}"
VLLM_REPO="${VLLM_REPO:-https://github.com/TomerBN-Nvidia/vllm.git}"
VLLM_BRANCH="${VLLM_BRANCH:-super_vl_rl_v0.25.1}"
FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/TomerBN-Nvidia/flashinfer.git}"
FLASHINFER_BRANCH="${FLASHINFER_BRANCH:-super_vl_rl_v0.6.13}"

export FMHA_ARTIFACT_PATH="${FMHA_ARTIFACT_PATH:-e3a8eba02eb19f4485652f84f5095524350246b5/fmha/trtllm-gen/}"
export FMHA_MANIFEST_SHA256="${FMHA_MANIFEST_SHA256:-03e0f29f970de40b0fd3c6025a16a39fb6c9af2a6549a63da73cf3da8494e658}"
export FMHA_MANIFEST_ENTRIES="${FMHA_MANIFEST_ENTRIES:-19227}"

VLLM_VERSION=0.25.1         # setuptools-scm cannot infer a version from a branch clone
CUDA_ARCH=10.0a             # Blackwell (GB200/B200); 9.0a for H100
BUILD_ROOT=/opt/super-vl-rl

###############################################################################
# Inside the container (this script re-execs itself here).
###############################################################################
if [[ "${1:-}" == __inside_build ]]; then
    echo "=== ${SLURM_JOB_ID:-N/A} on $(hostname) — $(date) ==="

    # The base image has python3 but no `python`, and no git at all.
    # FlashInfer's JIT needs cublasLt.h from the CUDA dev packages.
    command -v python &>/dev/null || ln -sf "$(which python3)" /usr/local/bin/python
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends git curl ca-certificates ninja-build \
        nvidia-cuda-dev libcublas-dev-13-0 2>&1 | tail -5
    pip install uv 2>&1 | tail -3

    echo ""; echo ">>> vLLM @ ${VLLM_BRANCH}"
    mkdir -p "${BUILD_ROOT}"
    git clone --depth=1 -b "${VLLM_BRANCH}" "${VLLM_REPO}" "${BUILD_ROOT}/vllm" 2>&1 | tail -3
    cd "${BUILD_ROOT}/vllm"
    vllm_sha=$(git rev-parse HEAD)
    test -z "${VLLM_HEAD_SHA:-}" || test "${vllm_sha}" = "${VLLM_HEAD_SHA}"

    # Precompiled: downloads upstream's prebuilt binaries, installs only the
    # Python layer. NOTE: this SILENTLY DISCARDS any C++/CUDA change on the
    # branch — drop it for a full source build if the branch touches csrc/.
    export VLLM_USE_PRECOMPILED=1
    export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"
    uv pip install --system . --prerelease=allow --torch-backend=auto \
        --index-strategy unsafe-best-match 2>&1

    echo ""; echo ">>> FlashInfer @ ${FLASHINFER_BRANCH}"
    # flashinfer-python builds in seconds (JIT). flashinfer-cubin is the slow
    # one: its build backend downloads ~16k cubins, which is what bakes them in.
    pip uninstall -y flashinfer-cubin flashinfer-python 2>/dev/null || true
    git clone --depth=1 -b "${FLASHINFER_BRANCH}" "${FLASHINFER_REPO}" "${BUILD_ROOT}/flashinfer" 2>&1 | tail -3
    cd "${BUILD_ROOT}/flashinfer"
    flashinfer_sha=$(git rev-parse HEAD)
    test -z "${FLASHINFER_HEAD_SHA:-}" || test "${flashinfer_sha}" = "${FLASHINFER_HEAD_SHA}"
    git submodule update --init --recursive --depth=1 2>&1 | tail -3

    export FLASHINFER_CUDA_ARCH_LIST="${CUDA_ARCH}"
    # ".[cu13]" — the bare install pulls the cu12 nvidia-cutlass-dsl flavour.
    uv pip install --system --prerelease=allow --no-build-isolation -e ".[cu13]" 2>&1
    uv pip install --system --no-build-isolation -e ./flashinfer-cubin 2>&1

    echo ""; echo ">>> Verify the TRTLLM-GEN cubins"
    cubin_root=$(python3 -c 'import flashinfer_cubin; print(flashinfer_cubin.get_cubin_dir())')
    manifest="${cubin_root}/${FMHA_ARTIFACT_PATH}/checksums.txt"
    test -f "${manifest}"
    test "$(sha256sum "${manifest}" | cut -d' ' -f1)" = "${FMHA_MANIFEST_SHA256}"
    test "$(wc -l < "${manifest}")" -eq "${FMHA_MANIFEST_ENTRIES}"
    python3 -c 'import os
from flashinfer.artifacts import ArtifactPath, CheckSumHash
assert ArtifactPath.TRTLLM_GEN_FMHA == os.environ["FMHA_ARTIFACT_PATH"]
assert CheckSumHash.TRTLLM_GEN_FMHA == os.environ["FMHA_MANIFEST_SHA256"]'

    echo ""; echo ">>> Rebuild the fmha_gen host dispatcher for the new cubins"
    python3 - <<'PY'
import shutil
from pathlib import Path

import flashinfer_jit_cache
from flashinfer.jit import env as jit_env
from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module
from flashinfer.jit.core import build_jit_specs

root = Path("/opt/super-vl-rl/flashinfer")
jit_env.FLASHINFER_CSRC_DIR = root / "csrc"
jit_env.FLASHINFER_INCLUDE_DIR = root / "include"
jit_env.CUTLASS_INCLUDE_DIRS = [
    root / "3rdparty/cutlass/include",
    root / "3rdparty/cutlass/tools/util/include",
]
jit_env.SPDLOG_INCLUDE_DIR = root / "3rdparty/spdlog/include"
jit_env.CCCL_INCLUDE_DIRS = [
    root / "3rdparty/cccl/cub",
    root / "3rdparty/cccl/libcudacxx/include",
    root / "3rdparty/cccl/thrust",
]

spec = gen_trtllm_gen_fmha_module()
build_dir = jit_env.FLASHINFER_JIT_DIR / spec.name
shutil.rmtree(build_dir, ignore_errors=True)
build_jit_specs([spec], verbose=False, skip_prebuilt=False)

compiled = build_dir / f"{spec.name}.so"
destination = Path(flashinfer_jit_cache.get_jit_cache_dir()) / spec.name / compiled.name
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(compiled, destination)
Path("/opt/super-vl-rl/fmha_dispatcher.path").write_text(str(destination))
PY
    fmha_dispatcher=$(cat /opt/super-vl-rl/fmha_dispatcher.path)
    grep -aFq "${FMHA_ARTIFACT_PATH}" "${fmha_dispatcher}"
    if grep -aFq '158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen/' "${fmha_dispatcher}"; then
        echo "ERROR: fmha_gen.so still references the old cubins" >&2
        exit 1
    fi
    fmha_dispatcher_sha256=$(sha256sum "${fmha_dispatcher}" | cut -d' ' -f1)

    echo ""; echo ">>> Runtime extras"
    uv pip install --system instanttensor "ray[default]>=2.54" 2>&1 | tail -5

    echo ""; echo ">>> Cleanup"
    rm -rf /opt/uv/cache /root/.cache/uv /root/.cache/pip /var/lib/apt/lists/* 2>/dev/null || true
    apt-get clean || true
    rm -rf "${BUILD_ROOT}/flashinfer/.git" 2>/dev/null || true
    rm -rf "${BUILD_ROOT}/vllm" 2>/dev/null || true
    find /usr/lib/aarch64-linux-gnu /usr/local/cuda-13.0 -name '*_static*.a' -delete 2>/dev/null || true
    python3 -c 'import flashinfer, instanttensor, ray, torch, vllm
from vllm.vllm_flash_attn import flash_attn_varlen_func'

    # Branch HEADs move, so this file is the only record of what is in the image.
    cat > "${BUILD_ROOT}/build.env" <<EOF
built_on=$(date -Is)
build_script=thin
base_image=${BASE_IMAGE}
vllm=${VLLM_BRANCH} @ ${vllm_sha}
flashinfer=${FLASHINFER_BRANCH} @ ${flashinfer_sha}
fmha_artifact=${FMHA_ARTIFACT_PATH}
fmha_manifest_sha256=${FMHA_MANIFEST_SHA256}
fmha_dispatcher_sha256=${fmha_dispatcher_sha256}
cuda_arch=${CUDA_ARCH}
omitted=cudnn-cu12,modelopt,mamba-ssm,causal-conv1d,static-archives
EOF
    echo ""; cat "${BUILD_ROOT}/build.env"
    exit 0
fi

###############################################################################
# Login node.
###############################################################################
OUT_SQSH="${1:-}"
if [[ -z "${OUT_SQSH}" || -z "${SLURM_ACCOUNT:-}" ]]; then
    echo "Usage: SLURM_ACCOUNT=<account> $0 <OUTPUT.sqsh>" >&2
    exit 2
fi

mkdir -p "$(dirname "${OUT_SQSH}")"
OUT_SQSH="$(cd "$(dirname "${OUT_SQSH}")" && pwd)/$(basename "${OUT_SQSH}")"
[[ -e "${OUT_SQSH}" ]] && { echo "ERROR: ${OUT_SQSH} already exists." >&2; exit 1; }

# Run an immutable snapshot: bash reads a script incrementally by byte offset,
# so editing this file while a build is running corrupts that run.
SNAP="$(dirname "${OUT_SQSH}")/.snapshot-$(basename "${OUT_SQSH}" .sqsh).sh"
cp "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" "${SNAP}"
trap 'rm -f "${SNAP}"' EXIT

MOUNTS="$(dirname "${SNAP}"):$(dirname "${SNAP}")"
[[ -d /lustre ]] && MOUNTS="${MOUNTS},/lustre:/lustre"

# Resolve branch HEADs before allocating a node, so a typo costs seconds.
VLLM_HEAD_SHA=$(git ls-remote "${VLLM_REPO}" "refs/heads/${VLLM_BRANCH}" | cut -f1)
FLASHINFER_HEAD_SHA=$(git ls-remote "${FLASHINFER_REPO}" "refs/heads/${FLASHINFER_BRANCH}" | cut -f1)
[[ -n "${VLLM_HEAD_SHA}" && -n "${FLASHINFER_HEAD_SHA}" ]] || {
    echo "ERROR: source branch not found." >&2; exit 1;
}
export VLLM_HEAD_SHA FLASHINFER_HEAD_SHA

echo "Base       ${BASE_IMAGE}"
echo "vLLM       ${VLLM_BRANCH} @ ${VLLM_HEAD_SHA:0:12}"
echo "FlashInfer ${FLASHINFER_BRANCH} @ ${FLASHINFER_HEAD_SHA:0:12}"
echo "Output     ${OUT_SQSH}"
echo "Building — ~22 min, dominated by the cubin download."

if ! srun \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION:-batch}" \
    --job-name=super-vl-rl-v0251-thin \
    --nodes=1 --ntasks=1 \
    --gpus-per-node=4 \
    --time="${SLURM_TIME:-02:00:00}" \
    --container-image="${BASE_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-save="${OUT_SQSH}" \
    --export=ALL \
    bash "${SNAP}" __inside_build
then
    # pyxis writes the .sqsh even when the build fails or times out, and a
    # broken image left on disk looks usable.
    echo "ERROR: build failed — removing ${OUT_SQSH}" >&2
    rm -f "${OUT_SQSH}"
    exit 1
fi

ls -lh "${OUT_SQSH}"
echo "Contents:  srun --container-image=${OUT_SQSH} cat /opt/super-vl-rl/build.env"
echo "VALIDATE:  SLURM_ACCOUNT=${SLURM_ACCOUNT} ./smoke-gsm8k-sqsh.sh ${OUT_SQSH}"
