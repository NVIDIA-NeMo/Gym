#!/usr/bin/env bash
set -Eeuo pipefail

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:nightly-2a02f6efe319c885e3ccbcecde402e0028f9ec1e}"
VLLM_REPO="${VLLM_REPO:-https://github.com/bxyu-nvidia/vllm.git}"
VLLM_BRANCH="${VLLM_BRANCH:-bxyu/mtp-fix-try02}"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-2a02f6efe319c885e3ccbcecde402e0028f9ec1e}"
BUILD_ROOT=/opt/super-vl-evals

###############################################################################
# Inside the container (this script re-execs itself here).
###############################################################################
if [[ "${1:-}" == __inside_build ]]; then
    echo "=== ${SLURM_JOB_ID:-N/A} on $(hostname) — $(date) ==="

    command -v python &>/dev/null || ln -sf "$(which python3)" /usr/local/bin/python
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends git ca-certificates 2>&1 | tail -5
    pip install uv 2>&1 | tail -3

    echo ""; echo ">>> vLLM @ ${VLLM_BRANCH}"
    mkdir -p "${BUILD_ROOT}"
    git clone --depth=1 -b "${VLLM_BRANCH}" "${VLLM_REPO}" "${BUILD_ROOT}/vllm" \
        2>&1 | tail -3
    cd "${BUILD_ROOT}/vllm"
    vllm_sha=$(git rev-parse HEAD)
    test -z "${VLLM_HEAD_SHA:-}" || test "${vllm_sha}" = "${VLLM_HEAD_SHA}"

    # The compatibility branch must remain Python-only so the exact v0.27.1
    # native extensions can be reused safely.
    git fetch --depth=1 https://github.com/vllm-project/vllm.git \
        "${VLLM_PRECOMPILED_WHEEL_COMMIT}" 2>&1 | tail -3
    non_python_changes=$(
        git diff --name-only "${VLLM_PRECOMPILED_WHEEL_COMMIT}" HEAD -- \
            | grep -Ev '\.py$' || true
    )
    if [[ -n "${non_python_changes}" ]]; then
        echo "ERROR: precompiled build cannot consume non-Python changes:" >&2
        echo "${non_python_changes}" >&2
        exit 1
    fi

    export VLLM_USE_PRECOMPILED=1
    export VLLM_PRECOMPILED_WHEEL_COMMIT
    export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"
    uv pip install --system . --prerelease=allow --torch-backend=auto \
        --index-strategy unsafe-best-match 2>&1

    echo ""; echo ">>> Verify"
    python3 -c 'import torch, vllm
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.model_executor.models.registry import ModelRegistry
assert "NemotronH_Omni_Reasoning_V3" in ModelRegistry.get_supported_archs()
print(f"vLLM {vllm.__version__}; torch {torch.__version__}")'

    echo ""; echo ">>> Cleanup"
    # Leave the source tree before removing it: Python's import machinery can
    # call getcwd(), which fails if the process is still inside that directory.
    cd /
    rm -rf /opt/uv/cache /root/.cache/uv /root/.cache/pip \
        /var/lib/apt/lists/* 2>/dev/null || true
    apt-get clean || true
    rm -rf "${BUILD_ROOT}/vllm" 2>/dev/null || true
    find /usr/lib/aarch64-linux-gnu /usr/local/cuda-13.0 \
        -name '*_static*.a' -delete 2>/dev/null || true

    python3 -c 'import torch, vllm
from vllm.vllm_flash_attn import flash_attn_varlen_func'

    cat > "${BUILD_ROOT}/build.env" <<EOF
built_on=$(date -Is)
build_script=super-vl-evals-v0271-thin
base_image=${BASE_IMAGE}
vllm=${VLLM_BRANCH} @ ${vllm_sha}
vllm_precompiled_wheel=${VLLM_PRECOMPILED_WHEEL_COMMIT}
base_image_flashinfer_and_cubins=unchanged
omitted=custom-flashinfer,custom-cubins,cubin-download,cubin-rebuild
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
[[ -e "${OUT_SQSH}" ]] && {
    echo "ERROR: ${OUT_SQSH} already exists." >&2
    exit 1
}

# Execute an immutable snapshot so later edits cannot corrupt an active build.
SNAP="$(dirname "${OUT_SQSH}")/.snapshot-$(basename "${OUT_SQSH}" .sqsh).sh"
cp "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" "${SNAP}"
trap 'rm -f "${SNAP}"' EXIT

MOUNTS="$(dirname "${SNAP}"):$(dirname "${SNAP}")"
[[ -d /lustre ]] && MOUNTS="${MOUNTS},/lustre:/lustre"

VLLM_HEAD_SHA=$(git ls-remote "${VLLM_REPO}" "refs/heads/${VLLM_BRANCH}" | cut -f1)
[[ -n "${VLLM_HEAD_SHA}" ]] || {
    echo "ERROR: vLLM branch not found: ${VLLM_BRANCH}" >&2
    exit 1
}
if [[ -n "${EXPECTED_VLLM_SHA:-}" && "${VLLM_HEAD_SHA}" != "${EXPECTED_VLLM_SHA}" ]]; then
    echo "ERROR: vLLM branch moved." >&2
    echo "expected ${EXPECTED_VLLM_SHA}" >&2
    echo "actual   ${VLLM_HEAD_SHA}" >&2
    exit 1
fi
export VLLM_HEAD_SHA

echo "Base   ${BASE_IMAGE}"
echo "vLLM   ${VLLM_BRANCH} @ ${VLLM_HEAD_SHA}"
echo "Output ${OUT_SQSH}"

if ! srun \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION:-batch}" \
    --job-name=super-vl-evals-v0271 \
    --nodes=1 --ntasks=1 --segment=1 \
    --gpus-per-node=4 --mem=0 \
    --time="${SLURM_TIME:-01:00:00}" \
    --container-image="${BASE_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --qos=${SLURM_QOS:-interactive} \
    --container-save="${OUT_SQSH}" \
    --export=ALL \
    bash "${SNAP}" __inside_build
then
    echo "ERROR: build failed — removing ${OUT_SQSH}" >&2
    rm -f "${OUT_SQSH}"
    exit 1
fi

ls -lh "${OUT_SQSH}"
sha256sum "${OUT_SQSH}" | tee "${OUT_SQSH}.sha256"
echo "Contents: srun --container-image=${OUT_SQSH} cat ${BUILD_ROOT}/build.env"
