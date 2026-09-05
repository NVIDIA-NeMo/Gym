#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=gdpval-e2e-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --open-mode=append

set -euo pipefail
: "${RUN_DIR:?set RUN_DIR}"
: "${E2E_DIR:?set E2E_DIR}"
exec "$E2E_DIR/run_checkpoint_e2e.sh" _compute-preflight "$RUN_DIR"
