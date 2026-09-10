#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Example Slurm/Pyxis launcher for SWE-bench Pro with sandboxed Hermes.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch --partition=cpu --cpus-per-task=8 --mem=32G}"
: "${HERMES_WORKSPACE:?Workspace containing Gym/ and hermes-runtime/}"
: "${HERMES_CONTAINER_IMAGE:?Pyxis image providing Apptainer and Gym system dependencies}"
TASK_ROOT="$HERMES_WORKSPACE"
TASK_REAL=$(realpath "$TASK_ROOT")
export HERMES_RUNTIME_DIR="${HERMES_RUNTIME_DIR:-$TASK_ROOT/hermes-runtime}"
TASK_SCRATCH="/var/tmp/hermes-pro-${SLURM_JOB_ID}"
export HERMES_OVERLAY_ROOT="$TASK_SCRATCH/overlays"
if [[ "${1:-}" != inner ]]; then
  mkdir -p -m 700 "$HERMES_OVERLAY_ROOT"
  TASK_SCRATCH_FS=$(stat -f -c %T "$HERMES_OVERLAY_ROOT")
  case "$TASK_SCRATCH_FS" in
    tmpfs|ramfs) echo "Overlay storage is RAM: $HERMES_OVERLAY_ROOT" >&2; exit 1 ;;
  esac
  echo "Overlay root: $HERMES_OVERLAY_ROOT; filesystem: $TASK_SCRATCH_FS"
  df -hT "$HERMES_OVERLAY_ROOT"
  trap 'rmdir "$HERMES_OVERLAY_ROOT" "$TASK_SCRATCH" 2>/dev/null || true' EXIT
  TASK_MOUNTS="$TASK_ROOT:$TASK_ROOT,$TASK_SCRATCH:$TASK_SCRATCH"
  if [[ "$TASK_REAL" != "$TASK_ROOT" ]]; then
    TASK_MOUNTS+=",$TASK_REAL:$TASK_REAL"
  fi
  if [[ -n "${HERMES_EXTRA_MOUNTS:-}" ]]; then
    TASK_MOUNTS+=",$HERMES_EXTRA_MOUNTS"
  fi
  srun --unbuffered \
    --container-image="$HERMES_CONTAINER_IMAGE" \
    --container-mounts="$TASK_MOUNTS" \
    --container-workdir="$TASK_ROOT/Gym" \
    bash "$TASK_ROOT/Gym/responses_api_agents/hermes_sandboxed_agent/examples/slurm/run_pro.sh" inner "$@"
  exit $?
fi
shift
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$TASK_ROOT/Gym" RAY_TMPDIR=/tmp APPTAINER_TMPDIR=/tmp
export APPTAINER_CACHEDIR="$TASK_ROOT/apptainer-cache"
if [[ "${1:-}" == --check-verifier ]]; then
  shift
  exec "$TASK_ROOT/Gym/.venv/bin/python" -m responses_api_agents.hermes_sandboxed_agent.check_verifier "$@"
fi
TASK_DATASET="$TASK_ROOT/Gym/resources_servers/swebench_pro/data/example.jsonl"

# Pull only selected tasks; cache identity uses the digest, with a checksum manifest.
TASK_INSTANCES=()
for ((TASK_ARG=1; TASK_ARG<=$#; TASK_ARG++)); do
  if [[ "${!TASK_ARG}" == --instance-id ]]; then
    TASK_ARG=$((TASK_ARG + 1))
    TASK_INSTANCES+=(--instance-id "${!TASK_ARG}")
  fi
done
"$TASK_ROOT/Gym/.venv/bin/python" -m resources_servers.swebench_pro.image_cache \
  --dataset "$TASK_DATASET" --image-dir "$TASK_ROOT/pro-sifs" "${TASK_INSTANCES[@]}"

exec "$TASK_ROOT/Gym/.venv/bin/python" -m responses_api_agents.hermes_sandboxed_agent.smoke \
  --dataset "$TASK_DATASET" \
  --provider-config "$TASK_ROOT/Gym/responses_api_agents/hermes_sandboxed_agent/examples/slurm/cluster-provider.yaml" \
  --output "$TASK_ROOT/results/pro-job-${SLURM_JOB_ID}" \
  --sif-dir "$TASK_ROOT/pro-sifs" "$@"
