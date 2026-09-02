#!/bin/bash
#
#SBATCH --output=slurm-logs/%j-%x.log
#SBATCH --job-name=gym-build_vllm_router_wheel

# Build a patched vllm-router wheel inside the eval base container.
#
# Why we do not just `uv pip install vllm-router`:
#   The released router resets every worker's in-flight load counter from the
#   registry health checker (every 10 health-check cycles). The cache-aware
#   policy uses those counters to decide when to abandon prefix affinity for
#   shortest-queue routing, so the reset makes a saturated worker look idle and
#   requests keep piling onto it.
#     bug: https://github.com/vllm-project/router/issues/197
#     fix: https://github.com/vllm-project/router/pull/216 (unmerged upstream)
#
# This is the only place the router is built. build_eval_container.sh takes the
# resulting wheel via its required VLLM_ROUTER_WHEEL input and asserts that what it
# installs really carries the fix, so a container cannot be built with the bad
# router by omission. The wheel is built inside the eval base image, so its
# extension module matches the Python that runs vllm-router at eval time.
#
# Usage:
#   SBATCH_ACCOUNT=... SBATCH_PARTITION=cpu SBATCH_QOS=cpu-short \
#   CONTAINER=/path/to/vllm-openai.sqsh \
#   sbatch benchmarks/nemotron_3.5_super/build_vllm_router_wheel.sh

set -euo pipefail

CONTAINER=$CONTAINER
OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/results/vllm_router}
VLLM_ROUTER_GIT_URL=${VLLM_ROUTER_GIT_URL:-https://github.com/vllm-project/router}
# Head of vllm-project/router#216 ("[Bugfix]: Fix health checker to preserve
# worker load during checks"). Fetching a bare SHA works because GitHub serves
# any commit reachable from a ref, and PR heads are refs.
VLLM_ROUTER_COMMIT=${VLLM_ROUTER_COMMIT:-9e6fce282a877c65185468692c6ba8a483409d9b}
RUST_TOOLCHAIN=${RUST_TOOLCHAIN:-1.95.0}
# Load-accounting unit tests from the fix. Compiling the test binary roughly
# doubles the job, so it is opt-out.
RUN_TESTS=${RUN_TESTS:-1}

mkdir -p "$OUTPUT_DIR" slurm-logs

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_ON_NODE:-16}" \
    --container-image="$CONTAINER" \
    --container-mounts="$OUTPUT_DIR:/out" \
    --no-container-mount-home \
    bash -s <<INNER_BUILD
set -xeuo pipefail

echo ">>> Container baseline"
cat /etc/os-release | head -3
uname -m
python3 -VV
(uv pip show --system vllm-router 2>/dev/null || pip3 show vllm-router 2>/dev/null || echo "vllm-router not installed in base image")

# ldconfig segfaults in this image on the cluster's 64K-page aarch64 kernels, so
# the libc-bin dpkg trigger fails and apt returns non-zero even though every
# package unpacked and configured fine. Ubuntu's glibc searches the multiarch
# dirs without ld.so.cache, so a stale cache is harmless here; check that the
# toolchain we asked for actually landed instead of trusting apt's exit code.
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

# protobuf-compiler: opentelemetry-otlp's grpc-tonic feature runs protoc at build time.
# libssl-dev/pkg-config: openssl-sys needs the headers. libzmq is vendored by zmq-sys.
apt-get update
apt_install build-essential pkg-config libssl-dev protobuf-compiler curl git ca-certificates
require cc c++ pkg-config protoc curl git /usr/include/openssl/ssl.h
rm -rf /var/lib/apt/lists/*

# Cargo needs tens of GB of scratch; fall back to the mounted output dir when
# the container's writable layer is small.
build_root=/tmp/vllm-router-build
mkdir -p "\$build_root"
avail_kb=\$(df -Pk "\$build_root" | awk 'NR == 2 {print \$4}')
if (( avail_kb < 25000000 )); then
    echo ">>> /tmp has only \$((avail_kb / 1024)) MiB free, building under /out instead"
    build_root=/out/build
    rm -rf "\$build_root"
    mkdir -p "\$build_root"
fi
export CARGO_HOME="\$build_root/cargo"
export RUSTUP_HOME="\$build_root/rustup"
export CARGO_TARGET_DIR="\$build_root/target"
export PATH="\$CARGO_HOME/bin:\$PATH"
# Reproducible CI-style builds; incremental only helps repeated local builds.
export CARGO_INCREMENTAL=0

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
rustc --version
cargo --version

src="\$build_root/router"
rm -rf "\$src"
mkdir -p "\$src"
cd "\$src"
git init -q
git remote add origin "$VLLM_ROUTER_GIT_URL"
git fetch --depth 1 origin "$VLLM_ROUTER_COMMIT"
git checkout -q FETCH_HEAD
test "\$(git rev-parse HEAD)" = "$VLLM_ROUTER_COMMIT"
echo ">>> Building vllm-router at \$(git rev-parse HEAD)"

# Assert the fix is actually in this tree: the health checker must no longer
# reset load counters. Guards against a moved/rewritten upstream ref.
if grep -rn "LOAD_RESET_INTERVAL" src/; then
    echo "ERROR: periodic load reset still present; expected vllm-project/router#216" >&2
    exit 1
fi

if [[ "$RUN_TESTS" == "1" ]]; then
    cargo test --lib -- \
        core::worker::tests::test_worker_load_guard \
        core::worker::tests::test_decrement_at_zero_clamps \
        core::worker::tests::test_load_guard \
        core::worker_registry::tests::test_health_checker_preserves
fi

# Build in a throwaway venv seeded from the container's own interpreter, so the
# extension module matches the Python that will run vllm-router at eval time
# without mutating the image's system site-packages. Mirrors the equivalent
# section of build_eval_container.sh.
buildenv="\$build_root/buildenv"
rm -rf "\$buildenv"
uv venv --python "\$(command -v python3)" "\$buildenv"
VIRTUAL_ENV="\$buildenv" uv pip install setuptools wheel setuptools-rust build
rm -rf dist
"\$buildenv/bin/python" -m build --wheel --no-isolation --outdir dist

mkdir -p /out/wheels
cp dist/*.whl /out/wheels/
ls -la /out/wheels

echo ">>> Smoke test: install the freshly built wheel and start the router"
uv pip install --system --force-reinstall /out/wheels/*.whl
python3 -c "import vllm_router_rs; print('vllm_router_rs import OK')"
vllm-router --help > /dev/null && echo "vllm-router --help OK"

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
uv pip show --system vllm-router

# The low-disk fallback puts build_root under the bind-mounted /out, where a
# rustup toolchain, the cargo registry and both target profiles would otherwise
# outlive the job on Lustre. Clean unconditionally; the wheel is already copied.
rm -rf "\$build_root"

echo ">>> Wheel build complete"
INNER_BUILD
