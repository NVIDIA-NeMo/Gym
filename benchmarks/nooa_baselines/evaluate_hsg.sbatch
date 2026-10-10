#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=nooa-three-baselines
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
set -euo pipefail
umask 077
run_root=${1:?}
image=${2:?Pass the controller runtime SquashFS image}
mounts=${3:?Pass explicit Pyxis mounts covering the run directory and sandbox assets}
runtime_env=${4:?Pass a private shell environment file readable inside the controller}
shift 4
exec srun --export=NONE --nodes=1 --ntasks=1 --gres=none \
  --container-image="$image" --container-mounts="$mounts" \
  --no-container-mount-home --container-workdir="$run_root/Gym" \
  /bin/bash --noprofile --norc -c \
  'unset PYTHONPATH PYTHONHOME VIRTUAL_ENV UV_CONSTRAINT PIP_CONSTRAINT; root=$1; runtime_env=$2; shift 2; set -a; source "$runtime_env"; set +a; exec "$root/controller-venv/bin/python" "$root/Gym/benchmarks/nooa_baselines/run_parallel.py" --run-root "$root" "$@"' \
  nooa-eval "$run_root" "$runtime_env" "$@"
