#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# One recipe run inside one allocation: preflight, then the run, into an attempt directory that is
# never reused. Submit from the Gym checkout at the recipe's gym_commit, with its venv prepared:
#
#   sbatch --account=<acct> --partition=cpu --job-name=atom-tb21 recipes/nooa-atom/slurm/run.sbatch terminal-bench-2.1 <attempts dir>
#
# Needs in the checkout: .env (keys), env.yaml (endpoint), and NOOA_RUNTIME_WHEELHOUSE exported or set in .env.
# LIMIT and PARALLEL pass through to run.sh. The job name is the idempotency key: refuse to submit a second
# job with the same name while one is pending or running (check squeue before sbatch).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
set -euo pipefail
umask 077

BENCH="${1:?recipe name}"
ATTEMPTS="${2:?attempts directory}"
ATTEMPT="$ATTEMPTS/${SLURM_JOB_NAME:-run}-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-nojob}"
mkdir -p "$ATTEMPT"
exec > >(tee -a "$ATTEMPT/job.log") 2>&1
echo "attempt $ATTEMPT on $(hostname) ($(uname -m)); checkout $(git rev-parse HEAD)"

[ -f .env ] || { echo ".env missing in $PWD" >&2; exit 3; }
set -a; source .env; set +a
[ -x .venv/bin/gym ] || [ -x "${VIRTUAL_ENV:-/nonexistent}/bin/gym" ] || { echo "no Gym venv in $PWD" >&2; exit 3; }
[ -x .venv/bin/gym ] && export PATH="$PWD/.venv/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"   # uv, for Gym's per-server venvs
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS="${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS:-1}"

# Preflight writes its own one-row result; the full run only starts if it passes.
OUT="$ATTEMPT/preflight" recipes/nooa-atom/preflight.sh "$BENCH"
OUT="$ATTEMPT/run" recipes/nooa-atom/run.sh "$BENCH"
echo "done: $ATTEMPT/run/evaluator_rollouts.jsonl"
