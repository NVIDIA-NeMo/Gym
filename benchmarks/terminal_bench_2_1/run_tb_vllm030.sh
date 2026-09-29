#!/usr/bin/env bash
set -euo pipefail

# Terminal Bench 2.1 via NeMo Gym + OpenCode on Slurm, disaggregated (prefill/decode) vLLM 0.30.0 with the
# decode-engine watchdog. See benchmarks/terminal_bench_2_1/VLLM030_WATCHDOG.md.
#
# Usage (runs out of the Gym checkout this script lives in):
#   CONTAINER=/path/to/vllm-0.30.0-with-gym.sqsh SBATCH_ACCOUNT=<acct> \
#     bash benchmarks/terminal_bench_2_1/run_tb_vllm030.sh /path/to/checkpoint/hf [extra hydra overrides...]
#   DRY_RUN=1 ... prints the command instead of submitting.

GYM_DIR="${GYM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:?set SBATCH_ACCOUNT}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:4}"
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
[[ -n "${SLURM_COMMENT:-}" ]] && export SLURM_COMMENT

# Restart a frozen decode engine in place (py-spy + nvidia-smi dumps land in the job log). 0 disables.
export VLLM_ENGINE_WATCHDOG="${VLLM_ENGINE_WATCHDOG:-1}"

NUM_PREFILL_NODES="${NUM_PREFILL_NODES:-2}"
NUM_DECODE_NODES="${NUM_DECODE_NODES:-4}"
VLLM_CONFIG="${VLLM_CONFIG:-benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_lightning_vllm030.sh}"
# vllm/vllm-openai:v0.30.0 image with Gym's server venvs installed; this checkout is mounted over /opt/Gym.
CONTAINER="${CONTAINER:?set CONTAINER to a vLLM 0.30.0 + Gym container image}"
BENCH_CONFIG="${BENCH_CONFIG:-benchmarks/terminal_bench_2_1/opencode.yaml}"

if [[ "${1:-}" != "" && "${1:-}" != -* ]]; then
  MODEL="$1"; shift
fi
MODEL="${MODEL:?usage: run_tb_vllm030.sh /path/to/checkpoint/hf [overrides...]}"
MODEL="${MODEL%/}"
if [[ "$(basename "${MODEL}")" == "hf" ]]; then
  MODEL_NAME="${MODEL_NAME:-$(basename "$(dirname "${MODEL}")")}"
else
  MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL}")}"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opencode_terminal_bench_2_1/${MODEL_NAME}}"
DRY_RUN="${DRY_RUN:-0}"
# Add the filesystems holding the checkpoint (and anything else the job reads) to MOUNTS.
MOUNTS="${MOUNTS:-${GYM_DIR}:/opt/Gym}"

[[ -f "${GYM_DIR}/${VLLM_CONFIG}" ]] || { echo "ERROR: VLLM_CONFIG missing: ${GYM_DIR}/${VLLM_CONFIG}"; exit 1; }
[[ -d "${MODEL}" ]] || { echo "ERROR: checkpoint dir not found: ${MODEL}"; exit 1; }
[[ -f "${GYM_DIR}/env.yaml" ]] \
  || { echo "ERROR: missing ${GYM_DIR}/env.yaml (opensandbox domain+api_key, wandb_api_key, hf_token; chmod 600)"; exit 1; }
[[ -f "${CONTAINER}" ]] || { echo "ERROR: container not found: ${CONTAINER}"; exit 1; }
# Each job prepares the benchmark data inside the checkout on startup. Jobs that start at the same time in a fresh
# clone race on it (one fails with `assert num_samples == 89` in prepare.py), so let the first job prepare it.
if [[ ! -s "${GYM_DIR}/benchmarks/terminal_bench_2_1/data/benchmark_prepare.jsonl" ]]; then
  echo "WARNING: benchmark data not prepared in this clone yet. Submit ONE run first and wait until" >&2
  echo "         benchmarks/terminal_bench_2_1/data/benchmark_prepare.jsonl exists (~1 min after it starts)" >&2
  echo "         before submitting more runs from this clone." >&2
fi

echo "Terminal Bench 2.1 (Gym + OpenCode, disaggregated vLLM 0.30.0, watchdog=${VLLM_ENGINE_WATCHDOG})"
echo "  gym clone:  ${GYM_DIR} ($(git -C "${GYM_DIR}" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "  checkpoint: ${MODEL}"
echo "  nodes:      ${NUM_PREFILL_NODES} prefill + ${NUM_DECODE_NODES} decode   account: ${SBATCH_ACCOUNT}   time: ${SBATCH_TIME}"
echo "  experiment: ${EXPERIMENT_NAME}"
echo

pass=(
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
  --config "${BENCH_CONFIG}"
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
  ++num_samples_in_parallel=1024
)

cd "${GYM_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] bash benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh ${pass[*]} $*"
  exit 0
fi
MODEL="${MODEL}" MODEL_NAME="${MODEL_NAME}" VLLM_CONFIG="${VLLM_CONFIG}" EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
NUM_PREFILL_NODES="${NUM_PREFILL_NODES}" NUM_DECODE_NODES="${NUM_DECODE_NODES}" \
CONTAINER="${CONTAINER}" MOUNTS="${MOUNTS}" \
bash benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh "${pass[@]}" "$@"
