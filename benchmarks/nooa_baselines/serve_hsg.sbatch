#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reconstructed from the recorded 2 October shared-model settings, not a copy
# of the inaccessible historical serving scripts. Submit with --export=NONE.
#SBATCH --job-name=nooa-policy
#SBATCH --partition=batch_long
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=144
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --export=NONE

set -euo pipefail
umask 077
run_dir=${1:?Pass a fresh absolute run directory containing these scripts}
case "$run_dir" in /*) ;; *) echo 'Run directory must be absolute' >&2; exit 2 ;; esac
test -d "$run_dir"
test ! -e "$run_dir/model.json"
test -r "$run_dir/serve_in_container.sh"
test -r "$run_dir/serving.py"
model=${2:?Pass the checkpoint directory}
image=${3:?Pass the compatible vLLM SquashFS image}
model_name=${4:?Pass the served model name}
mounts=${5:?Pass explicit Pyxis mounts covering the checkpoint, image and run directory}
test -r "$model/config.json"
test -r "$image"
scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | \
  awk '/^(JobId|JobState|Account|Partition|QOS|NumNodes|NumCPUs|NumTasks|CPUs\/Task|AllocTRES|ReqTRES|NodeList|TimeLimit)=/' \
  > "$run_dir/allocation.txt"

# No host Python, library paths, cache, credentials, or home bind enter vLLM.
exec srun --export=NONE --nodes=1 --ntasks=1 --kill-on-bad-exit=1 \
  --container-image="$image" --container-mounts="$mounts" \
  --no-container-mount-home --container-workdir="$run_dir" \
  /bin/bash --noprofile --norc "$run_dir/serve_in_container.sh" \
  "$run_dir" "$model" "$image" "$SLURM_JOB_ID" "$model_name"
