#!/bin/bash
#
#SBATCH --output=slurm-logs/%j-%x.log
#SBATCH --job-name=gym-patch_container_vllm_router

# Swap the vllm-router in an already-built eval container for a patched wheel,
# leaving everything else byte-identical.
#
# Use this instead of build_eval_container.sh when the only thing that needs to
# change is the router: a full rebuild re-runs `gym eval prepare`, which needs
# live credentials for the gated benchmarks (GPQA) and re-downloads everything.
# It is also what you want for a controlled A/B, since the resulting image
# differs from its input only by the router.
#
# See the vllm-router section of build_eval_container.sh for why the released
# wheel is not used: https://github.com/vllm-project/router/issues/197
#
# Usage:
#   SBATCH_ACCOUNT=... SBATCH_PARTITION=... \
#   INPUT_CONTAINER=/path/to/with_gym.sqsh \
#   OUTPUT_CONTAINER=/path/to/with_gym_patched.sqsh \
#   VLLM_ROUTER_WHEEL=results/vllm_router/wheels/vllm_router-*.whl \
#   sbatch benchmarks/nemotron_3.5_super/patch_container_vllm_router.sh

set -euo pipefail

INPUT_CONTAINER=$INPUT_CONTAINER
OUTPUT_CONTAINER=$OUTPUT_CONTAINER
# Resolve before srun so a bad path fails here with a message, rather than as a
# bare `readlink -f` non-zero under `set -e` with an empty log. The wheel must be on
# shared storage: its directory is bind-mounted into the step, so a node-local path
# like /tmp/... exists on the submit host but not on the compute node.
if [[ ! -f "$VLLM_ROUTER_WHEEL" ]]; then
    echo "VLLM_ROUTER_WHEEL not found: $VLLM_ROUTER_WHEEL" >&2
    echo "It must be readable from the compute node (Lustre, not node-local /tmp)." >&2
    exit 1
fi
VLLM_ROUTER_WHEEL=$(readlink -f "$VLLM_ROUTER_WHEEL")
# Recorded in the image so a later "uv pip show vllm-router" (which still reports
# the upstream 0.1.15) can be traced back to the tree it was built from.
VLLM_ROUTER_COMMIT=${VLLM_ROUTER_COMMIT:-9e6fce282a877c65185468692c6ba8a483409d9b}

wheel_dir=$(dirname "$VLLM_ROUTER_WHEEL")
wheel_name=$(basename "$VLLM_ROUTER_WHEEL")

mkdir -p slurm-logs

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
    --container-image="$INPUT_CONTAINER" \
    --container-mounts="$wheel_dir:/wheels" \
    --no-container-mount-home \
    --container-save="$staged_container" \
    bash -s <<INNER_BUILD || save_status=$?
set -xeuo pipefail

echo ">>> vllm-router before"
uv pip show --system vllm-router || echo "not installed"

uv pip install --system --reinstall-package vllm-router "/wheels/$wheel_name"

echo ">>> vllm-router after"
uv pip show --system vllm-router
python3 -c "import vllm_router_rs; print('vllm_router_rs import OK')"
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

# Written only after the assertions pass, so a shipped image can never claim a
# provenance its binary does not have.
cat > /opt/vllm_router_patch.txt <<MARKER
vllm-router built from https://github.com/vllm-project/router @ $VLLM_ROUTER_COMMIT
(head of PR #216, fix for issue #197: health checker resetting in-flight worker
load counters, which made cache-aware routing hot-spot a single decode worker).
The package version is still 0.1.15 because the PR does not bump it.
Wheel: $wheel_name
MARKER
cat /opt/vllm_router_patch.txt
INNER_BUILD
publish_or_fail "$save_status" "$staged_container" "$OUTPUT_CONTAINER"
