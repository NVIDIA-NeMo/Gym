#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --requeue

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: job.sbatch RUN MODEL LANGUAGE" >&2
  exit 2
fi

: "${GPQA_CONTAINER_IMAGE:?Export GPQA_CONTAINER_IMAGE before running launch.py}"
GPQA_CONTAINER_MOUNTS="${GPQA_CONTAINER_MOUNTS:-/lustre:/lustre}"
GPQA_PYTHON="${GPQA_PYTHON:-python}"

exec srun \
  --container-image="${GPQA_CONTAINER_IMAGE}" \
  --container-mounts="${GPQA_CONTAINER_MOUNTS}" \
  "${GPQA_PYTHON}" "$1/source/evaluate.py" "$1" "$2" "$3"
