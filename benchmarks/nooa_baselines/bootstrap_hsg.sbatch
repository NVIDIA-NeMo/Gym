#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=nooa-baseline-setup
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
set -euo pipefail
umask 077
run_root=${1:?}
image=${2:?Pass the controller runtime SquashFS image}
mounts=${3:?Pass explicit Pyxis mounts covering the run directory and sandbox assets}
exec srun --export=NONE --nodes=1 --ntasks=1 --gres=none \
  --container-image="$image" --container-mounts="$mounts" \
  --no-container-mount-home --container-workdir="$run_root/Gym" \
  /bin/bash --noprofile --norc "$run_root/Gym/benchmarks/nooa_baselines/bootstrap_eval.sh" "$run_root"
