#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
GYM_MAIN_DIR=${GYM_MAIN_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
TRAIN_PYTHON=${TRAIN_PYTHON:-${BUNDLE_DIR}/venvs/nemo-rl-gym-peft/bin/python}
CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
SOURCE_DATA_DIR=${SOURCE_DATA_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/inferno/data/rdkit-chemistry-no-tool/prepared-train1024-test1000}
ES_ADAPTER_DIR=${ES_ADAPTER_DIR:-${GYM_MAIN_DIR}/artifacts/inferno-rdkit-nemotron3-nano-post-step-0140-lora-r8}
ES140_MERGED_MODEL_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
BASELINE_SUMMARY=${BASELINE_SUMMARY:-${BUNDLE_DIR}/results/es140_lora/baseline_step0.json}
MERGE_PARITY_SUMMARY=${MERGE_PARITY_SUMMARY:-${BUNDLE_DIR}/results/es140_lora/merged_model_validation.json}

DATA_TOKEN_ARGS=()
if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
  DATA_TOKEN_ARGS=(--max-output-tokens "${MAX_NEW_TOKENS}")
fi
BASELINE_ARGS=()
if [[ "${REQUIRE_BASELINE_GATE:-0}" == "1" ]]; then
  BASELINE_ARGS=(--require-baseline-gate --baseline-summary "${BASELINE_SUMMARY}")
fi

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-2}" \
  --no-container-mount-home \
  --container-image="${CONTAINER_IMAGE_PATH}" \
  --container-mounts="/lustre:/lustre" \
  --container-workdir="${GYM_DIR}" \
  --chdir="${GYM_DIR}" \
  /bin/bash -lc "
    set -euo pipefail
    export PYTHONPATH='${GYM_DIR}'
    '${TRAIN_PYTHON}' '${BUNDLE_DIR}/prepare_direct_data.py' \\
      --root '${GYM_DIR}' \\
      --source-dir '${SOURCE_DATA_DIR}' ${DATA_TOKEN_ARGS[*]}
    '${TRAIN_PYTHON}' '${BUNDLE_DIR}/validate_bundle.py' \\
      --root '${GYM_DIR}' \\
      --require-container-preflight \\
      --require-adapter --adapter-dir '${ES_ADAPTER_DIR}' \\
      --require-merged-model --merged-model-dir '${ES140_MERGED_MODEL_DIR}' \\
      --require-merge-parity --merge-parity-summary '${MERGE_PARITY_SUMMARY}' \\
      ${BASELINE_ARGS[*]} ${DATA_TOKEN_ARGS[*]}
  "
