#!/bin/bash
# Build the per-task PinchBench container image (Node 22 + openclaw@2026.6.5 + brave plugin).
#   Docker (default):   bash build_image.sh
#   Apptainer/Slurm:    bash build_image.sh --apptainer   (also builds pinchbench.sif)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # responses_api_agents/pinchbench
TAG="${PINCHBENCH_IMAGE:-pinchbench-openclaw:latest}"

# The Dockerfile clones the PinchBench skill (pinned tag) + applies nvidia-pinchbench.patch,
# and COPYs run_task.sh + the patch from this dir, so the build context is the agent dir.
docker build -f "$HERE/Dockerfile.benchmark" -t "$TAG" "$HERE"
echo "built docker image: $TAG"

if [ "${1:-}" = "--apptainer" ]; then
  SIF="${PINCHBENCH_SIF:-$HERE/pinchbench.sif}"
  apptainer build "$SIF" "docker-daemon://$TAG"
  echo "built $SIF — set container_runtime=apptainer and sif_path=$SIF in configs/pinchbench.yaml"
fi
