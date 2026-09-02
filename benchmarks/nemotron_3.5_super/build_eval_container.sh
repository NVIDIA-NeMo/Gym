#!/bin/bash
# 
#SBATCH --output=slurm-logs/%j-%x.log
#SBATCH --job-name=gym-build_eval_container

set -euo pipefail

# Input arguments and validation
INPUT_CONTAINER=$INPUT_CONTAINER
OUTPUT_CONTAINER=$OUTPUT_CONTAINER
MOUNTS=$MOUNTS
GYM_CONFIG=$GYM_CONFIG
NEMO_GYM_GIT_URL=${NEMO_GYM_GIT_URL:-https://github.com/NVIDIA-NeMo/Gym}
NEMO_GYM_GIT_REF=${NEMO_GYM_GIT_REF:-main}
TAU_2_MOUNT_BASE_GYM_DIR=${TAU_2_MOUNT_BASE_GYM_DIR:-""}

# vllm-router build inputs. See the vllm-router section of the inner build for
# why we do not install the released wheel. Point VLLM_ROUTER_WHEEL at a wheel
# reachable inside the container (mount it via MOUNTS) to reuse an artifact
# from build_vllm_router_wheel.sh instead of rebuilding from source here.
VLLM_ROUTER_WHEEL=${VLLM_ROUTER_WHEEL:-""}
VLLM_ROUTER_GIT_URL=${VLLM_ROUTER_GIT_URL:-https://github.com/vllm-project/router}
VLLM_ROUTER_COMMIT=${VLLM_ROUTER_COMMIT:-9e6fce282a877c65185468692c6ba8a483409d9b}
VLLM_ROUTER_RUST_TOOLCHAIN=${VLLM_ROUTER_RUST_TOOLCHAIN:-1.95.0}

if [[ -n "$TAU_2_MOUNT_BASE_GYM_DIR" ]]; then
    MOUNTS="$MOUNTS,$TAU_2_MOUNT_BASE_GYM_DIR:$TAU_2_MOUNT_BASE_GYM_DIR"
fi

# pyxis --container-save exports the image when the step tears down, whatever the
# inner script exited with, and it overwrites whatever already sits at the target.
# Verified on pyxis v0.23.0: a step that exits 1 still produces a bootable sqsh, and
# re-running it clobbers a previously verified image. So build to a staging path and
# publish only on success -- otherwise a failed build silently replaces a good
# container with one whose provenance marker lies about what is installed.
staged_container="$OUTPUT_CONTAINER.partial"
rm -f "$staged_container"
save_status=0

publish_or_fail() {
    local status=$1 staged=$2 final=$3
    if (( status != 0 )); then
        rm -f "$staged"
        echo "Build failed (exit $status). $final left untouched." >&2
        exit "$status"
    fi
    if [[ ! -s "$staged" ]]; then
        echo "Build reported success but $staged is missing or empty." >&2
        exit 1
    fi
    mv -f "$staged" "$final"
    echo ">>> Published $final"
}

srun --nodes=1 --ntasks=1 \
    --container-image=$INPUT_CONTAINER \
    --container-mounts=$MOUNTS \
    --no-container-mount-home \
    --container-save="$staged_container" \
    bash -s <<INNER_BUILD || save_status=$?
set -xeuo pipefail

# Hardlink, not clone to save space
export UV_LINK_MODE=hardlink

# ldconfig segfaults in this image on the cluster's 64K-page aarch64 kernels, so
# the libc-bin dpkg trigger fails and apt returns non-zero even though every
# package unpacked and configured fine. Ubuntu's glibc searches the multiarch
# dirs without ld.so.cache, so a stale cache is harmless here; check that the
# tools we asked for actually landed instead of trusting apt's exit code.
apt_install() {
    apt-get install -y --no-install-recommends "\$@" \
        || { echo ">>> apt exited non-zero, re-running dpkg --configure"; dpkg --configure -a || true; }
}
require() {
    local missing=0
    for item in "\$@"; do
        if [[ "\$item" == /* ]]; then
            [[ -e "\$item" ]] || { echo "MISSING FILE: \$item" >&2; missing=1; }
        else
            command -v "\$item" > /dev/null || { echo "MISSING BINARY: \$item" >&2; missing=1; }
        fi
    done
    return "\$missing"
}

apt-get update
apt_install git
require git
rm -rf /var/lib/apt/lists/*

########################################
# START vllm-router
########################################
# The released vllm-router resets every worker's in-flight load counter from
# the registry health checker, every 10 health-check cycles. The cache-aware
# policy reads those counters to decide when to abandon prefix affinity for
# shortest-queue routing, so the reset makes an already-saturated worker look
# idle and P/D evals keep piling requests onto the same decode node.
#   bug: https://github.com/vllm-project/router/issues/197
#   fix: https://github.com/vllm-project/router/pull/216 (unmerged upstream)
# So we install the PR head instead of the released PyPI wheel.
#
# The build below is duplicated from build_vllm_router_wheel.sh on purpose:
# this script runs before the Gym checkout exists, so it cannot source it.
if [[ -n "$VLLM_ROUTER_WHEEL" ]]; then
    echo ">>> Installing prebuilt vllm-router wheel: $VLLM_ROUTER_WHEEL"
    uv pip install --system --reinstall-package vllm-router "$VLLM_ROUTER_WHEEL"
else
    echo ">>> Building vllm-router from $VLLM_ROUTER_GIT_URL @ $VLLM_ROUTER_COMMIT"
    apt-get update
    # protobuf-compiler: opentelemetry-otlp's grpc-tonic feature runs protoc at
    # build time. libssl-dev/pkg-config: openssl-sys needs the headers. libzmq is
    # vendored by zmq-sys, so it needs no system package.
    apt_install build-essential pkg-config libssl-dev protobuf-compiler curl ca-certificates
    require cc c++ pkg-config protoc curl /usr/include/openssl/ssl.h
    rm -rf /var/lib/apt/lists/*

    router_build_root=/tmp/vllm-router-build
    rm -rf "\$router_build_root"
    mkdir -p "\$router_build_root"
    (
        export CARGO_HOME="\$router_build_root/cargo"
        export RUSTUP_HOME="\$router_build_root/rustup"
        export CARGO_TARGET_DIR="\$router_build_root/target"
        export PATH="\$CARGO_HOME/bin:\$PATH"
        export CARGO_INCREMENTAL=0

        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --no-modify-path --profile minimal \
                --default-toolchain "$VLLM_ROUTER_RUST_TOOLCHAIN"

        mkdir -p "\$router_build_root/router"
        cd "\$router_build_root/router"
        git init -q
        git remote add origin "$VLLM_ROUTER_GIT_URL"
        # Fetching a bare SHA works because GitHub serves any commit reachable
        # from a ref, and PR heads are refs.
        git fetch --depth 1 origin "$VLLM_ROUTER_COMMIT"
        git checkout -q FETCH_HEAD
        test "\$(git rev-parse HEAD)" = "$VLLM_ROUTER_COMMIT"

        # Fail loudly if the pinned tree does not actually carry the fix.
        if grep -rq "LOAD_RESET_INTERVAL" src/; then
            echo "ERROR: periodic load reset still present; expected vllm-project/router#216" >&2
            exit 1
        fi

        # Build in a throwaway venv seeded from the container's own interpreter:
        # the extension module then matches the Python that runs vllm-router at
        # eval time, without dropping setuptools-rust/build into the image's
        # system site-packages alongside vLLM.
        buildenv="\$router_build_root/buildenv"
        uv venv --python "\$(command -v python3)" "\$buildenv"
        VIRTUAL_ENV="\$buildenv" uv pip install setuptools wheel setuptools-rust build
        "\$buildenv/bin/python" -m build --wheel --no-isolation --outdir dist
        uv pip install --system --reinstall-package vllm-router dist/*.whl
    )
    rm -rf "\$router_build_root"
fi

uv pip show --system vllm-router
vllm-router --help > /dev/null && echo ">>> vllm-router OK"

# The PR does not bump the version, so pip metadata cannot tell a patched router
# from the stock 0.1.15 wheel. The compiled extension can: upstream carries the
# periodic-reset log line, and only the fix carries the clamped-decrement warning.
router_so=\$(python3 -c "import vllm_router_rs; print(vllm_router_rs.__file__)")
if grep -aqF "Resetting worker loads (cycle" "\$router_so"; then
    echo "ERROR: installed vllm-router still resets worker loads periodically" >&2
    exit 1
fi
if ! grep -aqF "Attempted to decrement load counter that is already at 0" "\$router_so"; then
    echo "ERROR: installed vllm-router is missing the #216 load-guard changes" >&2
    exit 1
fi
echo ">>> vllm-router binary carries vllm-project/router#216"
########################################
# END vllm-router
########################################

cd /opt
# Python 3.13.14 is Gym main's Python version.
uv venv --python 3.13.14 Gym_venv
source Gym_venv/bin/activate

# We use this flow to support use cases where env.yaml, etc config files are mounted
# In these cases, git clone throws a non-empty directory error.
mkdir -p Gym
cd Gym
git init
git remote add origin $NEMO_GYM_GIT_URL
git fetch origin $NEMO_GYM_GIT_REF
git checkout $NEMO_GYM_GIT_REF

uv sync --active

########################################
# START Benchmark specific preparation
########################################

# See benchmarks/scicode/README.md
uv pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR" \
    -O benchmarks/scicode/data

if [[ -n "$TAU_2_MOUNT_BASE_GYM_DIR" ]]; then
    echo "Copying Tau2 and Tau3 data from mounted Gym dir: $TAU_2_MOUNT_BASE_GYM_DIR"
    cp -r "$TAU_2_MOUNT_BASE_GYM_DIR/benchmarks/tau2/nemo_gym_data" benchmarks/tau2/nemo_gym_data
    cp -r "$TAU_2_MOUNT_BASE_GYM_DIR/responses_api_agents/tau2/tau2_data" responses_api_agents/tau2/tau2_data
fi

########################################
# END Benchmark specific preparation
########################################

gym eval prepare +num_prepare_benchmark_processes=4 --config $GYM_CONFIG

gym env start \
    --config $GYM_CONFIG \
    ++dry_run=true \
    ++uv_venv_dir=/opt/uv_venvs

echo ">>> Inner build complete. Container will now be packed into sqsh."
INNER_BUILD
publish_or_fail "$save_status" "$staged_container" "$OUTPUT_CONTAINER"
