#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Layer a FlashInfer source build onto a vLLM image using Slurm/Pyxis.
# Example (run on a login node; the output directory must be shared):
#   SLURM_ACCOUNT=my_account BASE_IMAGE=/lustre/images/custom-vllm.sqsh \
#     FLASHINFER_BRANCH=my-branch bash custom_flashinfer_version.sh /lustre/images/custom-flashinfer.sqsh
# Optional: FLASHINFER_REPO, EXPECTED_FLASHINFER_SHA, SLURM_PARTITION,
# SLURM_QOS, SLURM_TIME. Keep the CUDA toolkit in the image for runtime JIT.
set -Eeuo pipefail

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/bxyu-nvidia/flashinfer.git}"
BUILD_ROOT=/opt/super-vl-evals

###############################################################################
# Inside the container (this script re-execs itself here).
###############################################################################
if [[ "${1:-}" == __inside_build ]]; then
    : "${BASE_IMAGE:?}" "${FLASHINFER_BRANCH:?}" "${FLASHINFER_HEAD_SHA:?}"
    echo ">>> FlashInfer ${FLASHINFER_BRANCH} @ ${FLASHINFER_HEAD_SHA} on $(hostname)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends git ca-certificates build-essential
    python3 -m pip install uv
    command -v nvcc >/dev/null || {
        echo "ERROR: BASE_IMAGE needs the CUDA toolkit (nvcc) for FlashInfer JIT." >&2
        exit 1
    }

    mkdir -p "${BUILD_ROOT}"
    BUILD_DIR=$(mktemp -d "${BUILD_ROOT}/flashinfer-build.XXXXXX")
    trap 'rm -rf "${BUILD_DIR}"' EXIT
    git init "${BUILD_DIR}/source"
    cd "${BUILD_DIR}/source"
    git remote add origin "${FLASHINFER_REPO}"
    # Fetch the resolved commit, even if the branch moves while the job queues.
    git fetch --depth=1 origin "${FLASHINFER_HEAD_SHA}"
    git checkout --detach FETCH_HEAD
    test "$(git rev-parse HEAD)" = "${FLASHINFER_HEAD_SHA}"
    git submodule update --init --recursive --depth=1

    # Prevent dependency resolution from replacing the image's CUDA/PyTorch stack.
    # Other FlashInfer dependencies can be installed/upgraded as the branch requires.
    python3 - "${BUILD_DIR}" <<'PY'
import importlib.metadata as metadata
import re
import sys
import tomllib
from pathlib import Path

build_dir = Path(sys.argv[1])
constraints = []
remove = []
preserve = {"torch", "torchvision", "torchaudio", "vllm", "triton", "numpy", "cuda-python", "cuda-bindings"}
for dist in metadata.distributions():
    name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
    if name.startswith("flashinfer-"):
        remove.append(dist.metadata["Name"])
    elif name in preserve or name.startswith("nvidia-"):
        constraints.append(f"{dist.metadata['Name']}=={dist.version}")
(build_dir / "constraints.txt").write_text("\n".join(constraints) + "\n")
(build_dir / "uninstall.txt").write_text("".join(f"{name}\n" for name in remove))
requirements = {"wheel"}
for project in (Path("."), Path("flashinfer-cubin")):
    requirements.update(tomllib.loads((project / "pyproject.toml").read_text())["build-system"]["requires"])
(build_dir / "build-requirements.txt").write_text("\n".join(sorted(requirements)) + "\n")
PY

    # Remove both the old cubins and every JIT-cache provider. An old precompiled
    # kernel can otherwise bypass the CUDA sources we are trying to change.
    if [[ -s "${BUILD_DIR}/uninstall.txt" ]]; then
        uv pip uninstall --system -r "${BUILD_DIR}/uninstall.txt"
    fi
    uv pip install --system --constraint "${BUILD_DIR}/constraints.txt" \
        -r "${BUILD_DIR}/build-requirements.txt" -r requirements.txt

    # A commit-specific version also isolates runtime JIT caches from older builds.
    export FLASHINFER_LOCAL_VERSION="g${FLASHINFER_HEAD_SHA}"
    unset FLASHINFER_DEV_RELEASE_SUFFIX FLASHINFER_DISABLE_JIT FLASHINFER_DISABLE_VERSION_CHECK
    unset FLASHINFER_CUBIN_DIR
    # Keep build hooks from silently replacing runtime dependencies outside uv's
    # constrained resolution above (supported by newer FlashInfer branches).
    export FLASHINFER_BUILD_NO_PIP=1
    uv pip install --system --no-deps --no-build-isolation .
    # This packages the branch's downloaded cubin artifacts; it does not rebuild
    # proprietary precompiled kernels. Editable CUDA sources use runtime JIT.
    uv pip install --system --no-deps --no-build-isolation ./flashinfer-cubin

    # Verify the installed packages outside the checkout, using a fresh JIT cache.
    cd /
    FLASHINFER_WORKSPACE_BASE="${BUILD_DIR}/smoke-cache" python3 - <<'PY'
import os

import flashinfer
import flashinfer_cubin
import torch
import vllm
from flashinfer._build_meta import __git_commit__

assert __git_commit__ == os.environ["FLASHINFER_HEAD_SHA"], __git_commit__
assert flashinfer.__version__ == flashinfer_cubin.__version__
assert torch.cuda.is_available(), "The build job needs a GPU for its JIT smoke check"
x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
actual = flashinfer.rmsnorm(x, weight, eps=1e-6)
expected = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
torch.testing.assert_close(actual, expected.to(x.dtype), rtol=1e-2, atol=1e-2)
torch.cuda.synchronize()
print(f"FlashInfer {flashinfer.__version__} from {flashinfer.__file__}")
print(f"vLLM {vllm.__version__}; torch {torch.__version__}")
print("Installed FlashInfer GPU JIT smoke check passed")
PY

    # Leave the vLLM build manifest intact when layering onto a custom image.
    {
        printf 'BASE_IMAGE=%s\nFLASHINFER_REPO=%s\nFLASHINFER_BRANCH=%s\nFLASHINFER_SHA=%s\n' \
            "${BASE_IMAGE}" "${FLASHINFER_REPO}" "${FLASHINFER_BRANCH}" "${FLASHINFER_HEAD_SHA}"
        python3 -c 'import flashinfer; print(f"FLASHINFER_VERSION={flashinfer.__version__}")'
    } > "${BUILD_ROOT}/flashinfer-build.env"
    cat "${BUILD_ROOT}/flashinfer-build.env"
    uv cache clean
    python3 -m pip cache purge || true
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    exit 0
fi

###############################################################################
# Login node.
###############################################################################
if [[ $# -ne 1 || -z "${SLURM_ACCOUNT:-}" || -z "${BASE_IMAGE:-}" || -z "${FLASHINFER_BRANCH:-}" ]]; then
    echo "Usage: SLURM_ACCOUNT=<account> BASE_IMAGE=<vllm image or .sqsh> FLASHINFER_BRANCH=<branch> $0 <OUTPUT.sqsh>" >&2
    exit 2
fi
export BASE_IMAGE FLASHINFER_BRANCH
OUT_SQSH="$1"
mkdir -p "$(dirname "${OUT_SQSH}")"
OUT_SQSH="$(cd "$(dirname "${OUT_SQSH}")" && pwd)/$(basename "${OUT_SQSH}")"
if [[ -e "${OUT_SQSH}" || -L "${OUT_SQSH}" ]]; then
    echo "ERROR: ${OUT_SQSH} already exists." >&2
    exit 1
fi

FLASHINFER_HEAD_SHA=$(git ls-remote --exit-code "${FLASHINFER_REPO}" "refs/heads/${FLASHINFER_BRANCH}" | cut -f1) || {
    echo "ERROR: could not resolve branch ${FLASHINFER_BRANCH} in ${FLASHINFER_REPO}." >&2
    exit 1
}
if [[ ! "${FLASHINFER_HEAD_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: expected one branch commit, got: ${FLASHINFER_HEAD_SHA}" >&2
    exit 1
fi
if [[ -n "${EXPECTED_FLASHINFER_SHA:-}" && "${FLASHINFER_HEAD_SHA}" != "${EXPECTED_FLASHINFER_SHA}" ]]; then
    echo "ERROR: branch moved: expected ${EXPECTED_FLASHINFER_SHA}, got ${FLASHINFER_HEAD_SHA}." >&2
    exit 1
fi
export FLASHINFER_HEAD_SHA

# Execute a snapshot so edits while this job queues cannot change its build.
SNAP=$(mktemp "$(dirname "${OUT_SQSH}")/.flashinfer-build.XXXXXX")
trap 'rm -f "${SNAP}"' EXIT
cp "$0" "${SNAP}"
MOUNTS="$(dirname "${SNAP}"):$(dirname "${SNAP}")"
[[ ! -d /lustre ]] || MOUNTS="${MOUNTS},/lustre:/lustre"

echo "Base   ${BASE_IMAGE}"
echo "Source ${FLASHINFER_REPO} ${FLASHINFER_BRANCH} @ ${FLASHINFER_HEAD_SHA}"
echo "Output ${OUT_SQSH}"
if ! srun \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION:-batch}" \
    --job-name=super-vl-evals-flashinfer \
    --nodes=1 --ntasks=1 --segment=1 \
    --gpus-per-node=4 --mem=0 \
    --time="${SLURM_TIME:-01:00:00}" \
    --container-image="${BASE_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --no-container-mount-home \
    --qos="${SLURM_QOS:-interactive}" \
    --container-save="${OUT_SQSH}" \
    --export=ALL \
    bash "${SNAP}" __inside_build
then
    echo "ERROR: build failed; removing ${OUT_SQSH}." >&2
    rm -f "${OUT_SQSH}"
    exit 1
fi

ls -lh "${OUT_SQSH}"
sha256sum "${OUT_SQSH}" | tee "${OUT_SQSH}.sha256"
echo "Manifest: ${BUILD_ROOT}/flashinfer-build.env inside the saved image"
