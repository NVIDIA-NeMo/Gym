#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Layer a CUDA 13 Mooncake source build onto a vLLM image using Slurm/Pyxis.
# Run on a login node with a shared output directory:
#   SLURM_ACCOUNT=my_account BASE_IMAGE=/lustre/images/custom-vllm.sqsh \
#     bash custom_mooncake_version.sh /lustre/images/custom-mooncake.sqsh
# Defaults to bxyu-nvidia/Mooncake main. Optional: MOONCAKE_REPO,
# MOONCAKE_BRANCH, EXPECTED_MOONCAKE_SHA, CUDA_HOME (default /usr/local/cuda),
# SLURM_PARTITION, SLURM_QOS, SLURM_TIME (default 02:00:00),
# SLURM_CPUS_PER_TASK (default 32), CMAKE_BUILD_PARALLEL_LEVEL (default 8).
# Use the output as CONTAINER with sbatch_external_vllm.sh; it automatically
# reuses the baked Mooncake installation when ENABLE_MOONCAKE=1.
set -Eeuo pipefail

export MOONCAKE_REPO="${MOONCAKE_REPO:-https://github.com/bxyu-nvidia/Mooncake.git}"
export MOONCAKE_BRANCH="${MOONCAKE_BRANCH:-main}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
BUILD_ROOT=/opt/super-vl-evals

if [[ "${1:-}" == __inside_build ]]; then
    : "${BASE_IMAGE:?}" "${MOONCAKE_HEAD_SHA:?}"
    echo ">>> Mooncake ${MOONCAKE_BRANCH} @ ${MOONCAKE_HEAD_SHA} on $(hostname)"
    if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]] || ! "${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 13\.'; then
        echo "ERROR: BASE_IMAGE needs a CUDA 13 development toolkit at CUDA_HOME=${CUDA_HOME}." >&2
        exit 1
    fi
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    # Install the C++ Store/TE dependencies directly; dependencies.sh also
    # installs Go and optional backends that this serving configuration omits.
    apt-get install -y --no-install-recommends \
        git ca-certificates build-essential cmake ninja-build pkg-config python3-dev \
        libibverbs-dev libgoogle-glog-dev libgflags-dev libjsoncpp-dev libunwind-dev \
        libnuma-dev libboost-dev libssl-dev libyaml-cpp-dev libcurl4-openssl-dev \
        liburing-dev libmsgpack-dev libzstd-dev libasio-dev libxxhash-dev
    # Newer Ubuntu releases split the C++ headers out of libmsgpack-dev.
    if apt-cache show libmsgpack-cxx-dev >/dev/null 2>&1; then
        apt-get install -y --no-install-recommends libmsgpack-cxx-dev
    fi
    command -v uv >/dev/null || python3 -m pip install uv

    mkdir -p "${BUILD_ROOT}"
    BUILD_DIR=$(mktemp -d "${BUILD_ROOT}/mooncake-build.XXXXXX")
    trap 'cd /; rm -rf "${BUILD_DIR}"' EXIT
    git init "${BUILD_DIR}/source"
    cd "${BUILD_DIR}/source"
    git remote add origin "${MOONCAKE_REPO}"
    # Use the commit resolved on the login node even if main moves while queued.
    git fetch --depth=1 origin "${MOONCAKE_HEAD_SHA}"
    git checkout --detach FETCH_HEAD
    test "$(git rev-parse HEAD)" = "${MOONCAKE_HEAD_SHA}"
    git submodule update --init --recursive --depth=1

    # Both distributions own the same mooncake package and CLI entry points.
    uv pip uninstall --system mooncake-transfer-engine-cuda13 mooncake-transfer-engine
    uv pip install --system . \
        --config-settings=cmake.build-type=Release \
        --config-settings=cmake.define.USE_CUDA=ON \
        --config-settings="cmake.define.CUDAToolkit_ROOT=${CUDA_HOME}" \
        --config-settings=cmake.define.WITH_TE=ON \
        --config-settings=cmake.define.WITH_STORE=ON \
        --config-settings=cmake.define.BUILD_EXAMPLES=OFF \
        --config-settings=cmake.define.BUILD_UNIT_TESTS=OFF \
        --config-settings=cmake.define.BUILD_BENCHMARK=OFF \
        --config-settings=cmake.define.WITH_EP=OFF \
        --config-settings=cmake.define.WITH_STORE_GO=OFF \
        --config-settings=cmake.define.WITH_STORE_RUST=OFF \
        --config-settings=cmake.define.WITH_STORE_RS=OFF \
        --config-settings=cmake.define.WITH_P2P_STORE=OFF

    # Verify only installed artifacts, after removing all build-tree libraries.
    cd /
    rm -rf "${BUILD_DIR}"
    export WITH_NVIDIA_PEERMEM=0
    timeout --kill-after=10s 120s python3 -I - <<'PY'
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from importlib.metadata import version

from mooncake.engine import TransferEngine
from mooncake.store import MooncakeDistributedStore

# TCP exercises the packaged master and native Store without assuming this
# build node has the serving cluster's RDMA NIC names. RDMA still needs a PD run.
with socket.socket() as rpc_socket, socket.socket() as metrics_socket:
    rpc_socket.bind(("127.0.0.1", 0))
    metrics_socket.bind(("127.0.0.1", 0))
    rpc_port = rpc_socket.getsockname()[1]
    metrics_port = metrics_socket.getsockname()[1]

with tempfile.TemporaryFile(mode="w+") as log:
    master = subprocess.Popen(
        [
            "mooncake_master", f"-rpc_port={rpc_port}", "-rpc_thread_num=4",
            f"-metrics_port={metrics_port}", "-enable_metric_reporting=false", "-logtostderr",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    store = None
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 30
        while True:
            if master.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Mooncake master failed its startup health check")
            try:
                with opener.open(f"http://127.0.0.1:{metrics_port}/health", timeout=2):
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.2)
        store = MooncakeDistributedStore()
        assert store.setup(
            "127.0.0.1", "P2PHANDSHAKE", 64 * 1024 * 1024, 4 * 1024 * 1024,
            "tcp", "", f"127.0.0.1:{rpc_port}",
        ) == 0, "Store setup failed"
        value = b"custom-mooncake-build-smoke"
        assert store.put("build-smoke", value) == 0, "Store put failed"
        assert store.get("build-smoke") == value, "Store get returned incorrect data"
        print(f"Mooncake {version('mooncake-transfer-engine')}: native imports, master health, TCP put/get passed")
    finally:
        try:
            if store is not None:
                store.close()
        finally:
            master.terminate()
            try:
                master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                master.kill()
                master.wait()
            log.seek(0)
            print(log.read())
PY

    # A separate manifest preserves the vLLM and FlashInfer layer provenance.
    {
        printf 'BASE_IMAGE=%s\nMOONCAKE_REPO=%s\nMOONCAKE_BRANCH=%s\nMOONCAKE_SHA=%s\n' \
            "${BASE_IMAGE}" "${MOONCAKE_REPO}" "${MOONCAKE_BRANCH}" "${MOONCAKE_HEAD_SHA}"
        python3 -c 'from importlib.metadata import version; print("MOONCAKE_VERSION=" + version("mooncake-transfer-engine"))'
        printf 'MOONCAKE_USE_CUDA=ON\nMOONCAKE_TCP_SMOKE_VERIFIED=1\n'
        "${CUDA_HOME}/bin/nvcc" --version
    } > "${BUILD_ROOT}/mooncake-build.env"
    cat "${BUILD_ROOT}/mooncake-build.env"
    uv cache clean
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    exit 0
fi

if [[ $# -ne 1 || -z "${SLURM_ACCOUNT:-}" || -z "${BASE_IMAGE:-}" ]]; then
    echo "Usage: SLURM_ACCOUNT=<account> BASE_IMAGE=<vllm image or .sqsh> $0 <OUTPUT.sqsh>" >&2
    exit 2
fi
export BASE_IMAGE
OUT_SQSH="$1"
mkdir -p "$(dirname "${OUT_SQSH}")"
OUT_SQSH="$(cd "$(dirname "${OUT_SQSH}")" && pwd)/$(basename "${OUT_SQSH}")"
if [[ -e "${OUT_SQSH}" || -L "${OUT_SQSH}" ]]; then
    echo "ERROR: ${OUT_SQSH} already exists." >&2
    exit 1
fi

MOONCAKE_HEAD_SHA=$(git ls-remote --exit-code "${MOONCAKE_REPO}" "refs/heads/${MOONCAKE_BRANCH}" | cut -f1) || {
    echo "ERROR: could not resolve branch ${MOONCAKE_BRANCH} in ${MOONCAKE_REPO}." >&2
    exit 1
}
if [[ ! "${MOONCAKE_HEAD_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: expected one branch commit, got: ${MOONCAKE_HEAD_SHA}" >&2
    exit 1
fi
if [[ -n "${EXPECTED_MOONCAKE_SHA:-}" && "${MOONCAKE_HEAD_SHA}" != "${EXPECTED_MOONCAKE_SHA}" ]]; then
    echo "ERROR: branch moved: expected ${EXPECTED_MOONCAKE_SHA}, got ${MOONCAKE_HEAD_SHA}." >&2
    exit 1
fi
export MOONCAKE_HEAD_SHA

SNAP=$(mktemp "$(dirname "${OUT_SQSH}")/.mooncake-build.XXXXXX")
trap 'rm -f "${SNAP}"' EXIT
cp "$0" "${SNAP}"
MOUNTS="$(dirname "${SNAP}"):$(dirname "${SNAP}")"
[[ ! -d /lustre ]] || MOUNTS="${MOUNTS},/lustre:/lustre"

echo "Base   ${BASE_IMAGE}"
echo "Source ${MOONCAKE_REPO} ${MOONCAKE_BRANCH} @ ${MOONCAKE_HEAD_SHA}"
echo "Output ${OUT_SQSH}"
if ! srun \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION:-batch}" \
    --job-name=super-vl-evals-mooncake \
    --nodes=1 --ntasks=1 --segment=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --gpus-per-node=4 --mem=0 \
    --time="${SLURM_TIME:-02:00:00}" \
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
echo "Manifest: ${BUILD_ROOT}/mooncake-build.env inside the saved image"
