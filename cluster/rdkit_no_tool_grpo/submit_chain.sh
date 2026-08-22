#!/usr/bin/env bash
set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
GYM_MAIN_DIR=${GYM_MAIN_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym}
INFERNO_DIR=${INFERNO_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/inferno}
export GYM_DIR GYM_MAIN_DIR
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
RAY_SUB="${BUNDLE_DIR}/nemo_rl_assets/ray.sub"
PREFLIGHT_SUB="${BUNDLE_DIR}/preflight_submission.sbatch"
TRAIN_PYTHON=${TRAIN_PYTHON:-/opt/nemo_rl_venv/bin/python}

SLURM_ACCOUNT=${SLURM_ACCOUNT:-healthcareeng_research}
SLURM_PARTITION=${SLURM_PARTITION:-pool0}
PREFLIGHT_PARTITION=${PREFLIGHT_PARTITION:-cpu}
PREFLIGHT_MEM=${PREFLIGHT_MEM:-64G}
NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CHAIN_JOBS=${CHAIN_JOBS:-3}
CHAIN_DEPENDENCY_TYPE=${CHAIN_DEPENDENCY_TYPE:-afterok}
START_DEPENDENCY=${START_DEPENDENCY:-}
JOB_INDEX_OFFSET=${JOB_INDEX_OFFSET:-0}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-rdkit-es140-lora-grpo}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-12:00:00}
CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0-gym-peft.sqsh}
SOURCE_DATA_DIR=${SOURCE_DATA_DIR:-${INFERNO_DIR}/data/rdkit-chemistry-no-tool/prepared-train1024-test1000}
ES_ADAPTER_DIR=${ES_ADAPTER_DIR:-${GYM_MAIN_DIR}/artifacts/inferno-rdkit-nemotron3-nano-post-step-0140-lora-r8}
export ES140_MERGED_MODEL_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-${BUNDLE_DIR}/results/model_artifacts/megatron}

RESULTS_DIR=${RESULTS_DIR:-${BUNDLE_DIR}/results/es140_lora}
TRAIN_LOG_DIR=${TRAIN_LOG_DIR:-${RESULTS_DIR}/logs}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}
BASELINE_SUMMARY=${BASELINE_SUMMARY:-${RESULTS_DIR}/baseline_step0.json}
export REQUIRE_BASELINE_GATE=1 BASELINE_SUMMARY

export WANDB_RUN_ID=${WANDB_RUN_ID:-rdkit-es140-grpo-v06-r8}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-rdkit-nemotron3-nano-grpo-lora-r8-a8-es140-64p16g-i200-lr3e-6-32k-iad-p0-64g}
export WANDB_GROUP=${WANDB_GROUP:-rdkit-es140-lora-grpo}
export WANDB_ENTITY=${WANDB_ENTITY:-nemo-llm-service}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-600}
export CONTAINER_IMAGE_PATH SOURCE_DATA_DIR ES_ADAPTER_DIR TRAIN_PYTHON

if [[ -z "${START_DEPENDENCY}" ]]; then
  echo "START_DEPENDENCY must identify the successful validation-only baseline job" >&2
  exit 1
fi
if [[ "${CHAIN_JOBS}" -lt 1 || "${NUM_ACTOR_NODES}" -ne 8 || "${GPUS_PER_NODE}" -ne 8 ]]; then
  echo "Production requires at least one job and the validated 8x8 GPU layout" >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not present in the submission environment" >&2
  exit 1
fi
if (( ${#WANDB_RUN_ID} > 32 )); then
  echo "WANDB_RUN_ID must be at most 32 characters so rollout-table artifact names remain valid" >&2
  exit 1
fi

mkdir -p \
  "${BUNDLE_DIR}/logs" \
  "${TRAIN_LOG_DIR}" \
  "${CHECKPOINT_DIR}" \
  "${NRL_MEGATRON_CHECKPOINT_DIR}"

COMMAND=$(cat <<EOF
set -euo pipefail
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT}"
cd "\${GYM_DIR}"
"${TRAIN_PYTHON}" "\${GYM_DIR}/cluster/rdkit_no_tool_grpo/nemo_rl_assets/run_grpo_nemo_gym.py" \\
  --config="\${GYM_DIR}/cluster/rdkit_no_tool_grpo/rdkit_no_tool_grpo.yaml" \\
  ++cluster.num_nodes=${NUM_ACTOR_NODES} \\
  ++cluster.gpus_per_node=${GPUS_PER_NODE} \\
  ++grpo.max_num_epochs=100000 \\
  ++grpo.max_num_steps=200 \\
  ++grpo.val_at_start=false \\
  ++grpo.val_period=5 \\
  ++checkpointing.checkpoint_dir="${CHECKPOINT_DIR}" \\
  ++logger.log_dir="${TRAIN_LOG_DIR}" \\
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

cd "${GYM_DIR}"
preflight_job_id=$(sbatch \
  --parsable \
  --nodes=1 \
  --cpus-per-task=2 \
  --mem="${PREFLIGHT_MEM}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PREFLIGHT_PARTITION}" \
  --time=00:20:00 \
  --job-name="${JOB_NAME_PREFIX}-preflight" \
  --output="${BUNDLE_DIR}/logs/%x-%j.out" \
  --error="${BUNDLE_DIR}/logs/%x-%j.err" \
  --dependency="${START_DEPENDENCY}" \
  --export=ALL \
  --chdir="${GYM_DIR}" \
  "${PREFLIGHT_SUB}")
preflight_job_id="${preflight_job_id%%;*}"
echo "${JOB_NAME_PREFIX}-preflight: ${preflight_job_id}"

prev_job=""
for idx in $(seq 1 "${CHAIN_JOBS}"); do
  job_num=$((idx + JOB_INDEX_OFFSET))
  job_name=$(printf "%s-%02d" "${JOB_NAME_PREFIX}" "${job_num}")
  if [[ -n "${prev_job}" ]]; then
    dependency="${CHAIN_DEPENDENCY_TYPE}:${prev_job}"
  else
    dependency="afterok:${preflight_job_id}"
  fi
  job_id=$(sbatch \
    --parsable \
    --nodes="${NUM_ACTOR_NODES}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --time="${JOB_TIME_LIMIT}" \
    --job-name="${job_name}" \
    --output="${BUNDLE_DIR}/logs/%x-%j.out" \
    --error="${BUNDLE_DIR}/logs/%x-%j.err" \
    --dependency="${dependency}" \
    --export=ALL \
    --chdir="${GYM_DIR}" \
    "${RAY_SUB}")
  prev_job="${job_id%%;*}"
  echo "${job_name}: ${prev_job}"
done

echo "W&B run id: ${WANDB_RUN_ID}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
