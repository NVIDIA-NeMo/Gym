#!/bin/bash
set -euo pipefail

# From the Gym root: bash benchmarks/launcher/build/build_vllm_router.sh (live build output)
# Build https://github.com/bxyu-nvidia/router/pull/4 as a standalone ARM64 image.
BASE_IMAGE=${BASE_IMAGE:-ghcr.io#astral-sh/uv:python3.12-bookworm-slim}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/images}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
image="$OUTPUT_DIR/vllm-router-pr4-$(date +%Y%m%d-%H%M%S)-$$.sqsh"

srun --job-name=build-router --account="${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT to your Slurm account}" \
    --partition=cpu --qos=cpu-normal --cpus-per-task=16 --mem=64G --time=02:00:00 \
    --nodes=1 --ntasks=1 --unbuffered --container-image="$BASE_IMAGE" \
    --container-workdir=/ --container-save="$image" \
    --no-container-mount-home bash -s <<'BUILD'
set -euo pipefail
[[ $(uname -m) == aarch64 ]]
apt-get update
apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev protobuf-compiler curl git ca-certificates
export CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup
export PATH="$CARGO_HOME/bin:$PATH"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain 1.95.0

git init -q /opt/router
cd /opt/router
git remote add origin https://github.com/bxyu-nvidia/router.git
git fetch --depth 1 origin refs/pull/4/head
git checkout -q FETCH_HEAD
git rev-parse HEAD | tee /opt/vllm-router-commit.txt
uv build --wheel --python "$(command -v python3)"
uv pip install --system dist/*.whl
python3 -c 'import vllm_router_rs; print("Router extension import OK")'
vllm-router --help > /dev/null
BUILD

echo "Router image ready: $image"
