#!/bin/bash
# 
#SBATCH --output=slurm-logs/%j-%x.log
#SBATCH --job-name=gym-add_vllm_router

set -euo pipefail

# Input arguments and validation
INPUT_CONTAINER=$INPUT_CONTAINER
OUTPUT_CONTAINER=$OUTPUT_CONTAINER
VLLM_ROUTER_WHEEL=$VLLM_ROUTER_WHEEL

VLLM_ROUTER_WHEEL=$(readlink -f "$VLLM_ROUTER_WHEEL")
MOUNTS="$(dirname "$VLLM_ROUTER_WHEEL"):$(dirname "$VLLM_ROUTER_WHEEL")"

# pyxis --container-save exports the image when the step tears down, whatever the
# inner script exited with, and it overwrites whatever already sits at the target.
# So stage the build and publish only on success; otherwise a failed build silently
# replaces a good container with a broken one.
staged_container="$OUTPUT_CONTAINER.partial"
rm -f "$staged_container"
save_status=0

srun --nodes=1 --ntasks=1 \
    --container-image=$INPUT_CONTAINER \
    --container-mounts=$MOUNTS \
    --no-container-mount-home \
    --container-save="$staged_container" \
    bash -s <<INNER_BUILD || save_status=$?
set -xeuo pipefail

# Hardlink, not clone to save space
export UV_LINK_MODE=hardlink

uv pip install --system --reinstall-package vllm-router "$VLLM_ROUTER_WHEEL"
uv pip show --system vllm-router

echo ">>> Inner build complete. Container will now be packed into sqsh."
INNER_BUILD

if (( save_status != 0 )); then
    rm -f "$staged_container"
    echo "Build failed (exit $save_status). $OUTPUT_CONTAINER left untouched." >&2
    exit "$save_status"
fi
mv -f "$staged_container" "$OUTPUT_CONTAINER"
echo ">>> Published $OUTPUT_CONTAINER"
