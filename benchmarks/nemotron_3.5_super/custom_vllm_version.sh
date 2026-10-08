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
    apt-get install -y --no-install-recommends git ca-certificates curl build-essential pkg-config perl \
        2>&1 | tail -5
    pip install uv 2>&1 | tail -3

    echo ""; echo ">>> vLLM @ ${VLLM_BRANCH}"
    mkdir -p "${BUILD_ROOT}"
    git clone --depth=1 -b "${VLLM_BRANCH}" "${VLLM_REPO}" "${BUILD_ROOT}/vllm" \
        2>&1 | tail -3
    cd "${BUILD_ROOT}/vllm"
    vllm_sha=$(git rev-parse HEAD)
    test -z "${VLLM_HEAD_SHA:-}" || test "${vllm_sha}" = "${VLLM_HEAD_SHA}"

    # Allow Python, text, and YAML changes when reusing the wheel's native code.
    # Other file types still require review before using precompiled artifacts.
    git fetch --depth=1 https://github.com/vllm-project/vllm.git \
        "${VLLM_PRECOMPILED_WHEEL_COMMIT}" 2>&1 | tail -3
    # An explicitly fetched tag may exist only as FETCH_HEAD in a shallow clone.
    precompiled_base_sha=$(git rev-parse 'FETCH_HEAD^{commit}')
    changed_files=$(git diff --name-only "${precompiled_base_sha}" HEAD --)
    unsupported_changes=$(
        printf '%s\n' "${changed_files}" | grep -Ev \
            '(^$|\.(py|txt|yaml)$)' || true
    )
    if [[ -n "${unsupported_changes}" ]]; then
        echo "ERROR: precompiled build cannot consume changes outside .py, .txt, and .yaml files:" >&2
        echo "${unsupported_changes}" >&2
        exit 1
    fi

    export VLLM_USE_PRECOMPILED=1
    export VLLM_PRECOMPILED_WHEEL_COMMIT
    export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"
    uv pip install --system . --prerelease=allow --torch-backend=auto \
        --index-strategy unsafe-best-match 2>&1

    # VLLM_USE_PRECOMPILED also copies Rust artifacts from the wheel. Python-only
    # launcher changes can still break their CLI/protocol, so rebuild Rust from
    # this checkout AFTER installing the wheel (which would overwrite the build).
    uv pip install --system 'setuptools>=77.0.3,<81' 'setuptools-scm>=9.2.0' 'setuptools-rust>=1.9.0' wheel
    bash tools/build_rust.sh
    installed_vllm_dir=$(cd / && python3 -c 'import vllm; from pathlib import Path; print(Path(vllm.__file__).parent)')
    install -m 755 vllm/vllm-rs "${installed_vllm_dir}/vllm-rs"
    for rust_extension in vllm/_rust_*.so; do
        [[ -f "$rust_extension" ]] || continue
        install -m 755 "$rust_extension" "${installed_vllm_dir}/"
    done

    # Check the installed executable, not a source-tree copy. This catches the
    # --input-address vs --input-listener-fd failure before publishing an image.
    python3 - "${installed_vllm_dir}" <<'PY'
import ast
import subprocess
import sys
from pathlib import Path

package = Path(sys.argv[1])
tree = ast.parse((package / "v1/utils.py").read_text())
manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RustFrontendProcessManager")
flags = {node.value for node in ast.walk(manager)
         if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("--")}
help_text = subprocess.check_output([str(package / "vllm-rs"), "frontend", "--help"], text=True)
missing = flags - set(help_text.split())
if missing:
    raise SystemExit(f"Installed Rust frontend does not support Python launcher flags: {sorted(missing)}")
print("Rust frontend accepts the installed Python launcher's flags")
PY

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
vllm_rust_source=${vllm_sha}
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
