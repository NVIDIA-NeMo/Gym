#!/usr/bin/env bash
set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
GYM_MAIN_DIR=${GYM_MAIN_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym}
INFERNO_DIR=${INFERNO_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/inferno}
export GYM_DIR GYM_MAIN_DIR
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
RAY_SUB="${BUNDLE_DIR}/nemo_rl_assets/ray.sub"
PREFLIGHT_SUB="${BUNDLE_DIR}/preflight_submission.sbatch"
RUNNER="${BUNDLE_DIR}/nemo_rl_assets/run_grpo_nemo_gym.py"
CONFIG="${BUNDLE_DIR}/rdkit_no_tool_grpo.yaml"
TRAIN_PYTHON=${TRAIN_PYTHON:-${BUNDLE_DIR}/venvs/nemo-rl-gym-peft/bin/python}

SLURM_ACCOUNT=${SLURM_ACCOUNT:-healthcareeng_research}
SLURM_PARTITION=${SLURM_PARTITION:-pool0}
PREFLIGHT_PARTITION=${PREFLIGHT_PARTITION:-cpu}
NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
START_DEPENDENCY=${START_DEPENDENCY:-}
CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
SOURCE_DATA_DIR=${SOURCE_DATA_DIR:-${INFERNO_DIR}/data/rdkit-chemistry-no-tool/prepared-train1024-test1000}
ES_ADAPTER_DIR=${ES_ADAPTER_DIR:-${GYM_MAIN_DIR}/artifacts/inferno-rdkit-nemotron3-nano-post-step-0140-lora-r8}
export ES140_MERGED_MODEL_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
export HF_HOME=${HF_HOME:-${INFERNO_DIR}/.local/cache/huggingface}
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-${BUNDLE_DIR}/results/model_cache/megatron}

RESULTS_DIR=${RESULTS_DIR:-${BUNDLE_DIR}/results/es140_lora}
BASELINE_SUMMARY=${BASELINE_SUMMARY:-${RESULTS_DIR}/baseline_step0.json}
export WANDB_RUN_ID=${WANDB_RUN_ID:-rdkit-nemotron3-nano-grpo-lora-r8-a8-es140-64p16g-i200-lr3e-6-32k-iad-p0-64g}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-rdkit-nemotron3-nano-grpo-lora-r8-a8-es140-64p16g-i200-lr3e-6-32k-iad-p0-64g}
export WANDB_GROUP=${WANDB_GROUP:-rdkit-es140-lora-grpo}
export WANDB_ENTITY=${WANDB_ENTITY:-nemo-llm-service}
export WANDB_DIR=${WANDB_DIR:-${BUNDLE_DIR}/results/wandb}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-600}
export CONTAINER_IMAGE_PATH SOURCE_DATA_DIR ES_ADAPTER_DIR TRAIN_PYTHON

if [[ "${NUM_ACTOR_NODES}" -ne 8 || "${GPUS_PER_NODE}" -ne 8 ]]; then
  echo "The production baseline requires the validated 8x8 GPU layout" >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not present in the submission environment" >&2
  exit 1
fi

mkdir -p \
  "${BUNDLE_DIR}/logs" \
  "${RESULTS_DIR}/baseline_logs" \
  "${RESULTS_DIR}/baseline_checkpoints" \
  "${NRL_MEGATRON_CHECKPOINT_DIR}" \
  "${WANDB_DIR}"

COMMAND=$(cat <<EOF
set -euo pipefail
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT}"
cd "${GYM_DIR}"
"${TRAIN_PYTHON}" "${RUNNER}" \\
  --config="${CONFIG}" \\
  --validation-only \\
  --validation-summary="${BASELINE_SUMMARY}" \\
  --min-validation-accuracy=0.518 \\
  --max-validation-accuracy=0.578 \\
  ++cluster.num_nodes=${NUM_ACTOR_NODES} \\
  ++cluster.gpus_per_node=${GPUS_PER_NODE} \\
  ++grpo.val_period=1 \\
  ++grpo.val_at_start=false \\
  ++checkpointing.enabled=false \\
  ++checkpointing.checkpoint_dir="${RESULTS_DIR}/baseline_checkpoints" \\
  ++logger.log_dir="${RESULTS_DIR}/baseline_logs" \\
  ++logger.wandb.name="${WANDB_RUN_NAME}" \\
  ++logger.wandb.id="${WANDB_RUN_ID}" \\
  ++logger.wandb.group="${WANDB_GROUP}"
EOF
)

export COMMAND
export CONTAINER="${CONTAINER_IMAGE_PATH}"
export MOUNTS="/lustre:/lustre"
export BASE_LOG_DIR="${BUNDLE_DIR}/logs"
export HOST_WORKDIR="${GYM_DIR}"
export GPUS_PER_NODE

dependency=()
if [[ -n "${START_DEPENDENCY}" ]]; then
  dependency=(--dependency="${START_DEPENDENCY}")
fi

cd "${GYM_DIR}"
preflight_job_id=$(sbatch \
  --parsable \
  --nodes=1 \
  --cpus-per-task=2 \
  --mem=8G \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PREFLIGHT_PARTITION}" \
  --time=00:20:00 \
  --job-name=rdkit-es140-baseline-preflight \
  --output="${BUNDLE_DIR}/logs/%x-%j.out" \
  --error="${BUNDLE_DIR}/logs/%x-%j.err" \
  --export=ALL \
  --chdir="${GYM_DIR}" \
  "${dependency[@]}" \
  "${PREFLIGHT_SUB}")
preflight_job_id="${preflight_job_id%%;*}"

job_id=$(sbatch \
  --parsable \
  --nodes="${NUM_ACTOR_NODES}" \
  --gpus-per-node="${GPUS_PER_NODE}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --time=06:00:00 \
  --job-name=rdkit-es140-lora-baseline \
  --output="${BUNDLE_DIR}/logs/%x-%j.out" \
  --error="${BUNDLE_DIR}/logs/%x-%j.err" \
  --dependency="afterok:${preflight_job_id}" \
  --export=ALL \
  --chdir="${GYM_DIR}" \
  "${RAY_SUB}")

echo "preflight: ${preflight_job_id}"
echo "baseline: ${job_id%%;*}"
echo "summary: ${BASELINE_SUMMARY}"
echo "W&B run id: ${WANDB_RUN_ID}"
