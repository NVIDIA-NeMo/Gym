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
START_DEPENDENCY=${START_DEPENDENCY:-}
CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0-gym-peft.sqsh}
SOURCE_DATA_DIR=${SOURCE_DATA_DIR:-${INFERNO_DIR}/data/rdkit-chemistry-no-tool/prepared-train1024-test1000}
ES_ADAPTER_DIR=${ES_ADAPTER_DIR:-${GYM_MAIN_DIR}/artifacts/inferno-rdkit-nemotron3-nano-post-step-0140-lora-r8}
export ES140_MERGED_MODEL_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-${BUNDLE_DIR}/results/model_artifacts/megatron}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
export WANDB_RUN_ID=${WANDB_RUN_ID:-rdkit-es140-lora-grpo-smoke-${RUN_STAMP}}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-rdkit-es140-lora-grpo-smoke-${RUN_STAMP}}
export WANDB_GROUP=${WANDB_GROUP:-rdkit-es140-lora-grpo-smoke}
export WANDB_ENTITY=${WANDB_ENTITY:-nemo-llm-service}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-600}
export CONTAINER_IMAGE_PATH SOURCE_DATA_DIR ES_ADAPTER_DIR TRAIN_PYTHON

for required in "${CONTAINER_IMAGE_PATH}" "${ES140_MERGED_MODEL_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not present in the submission environment" >&2
  exit 1
fi

mkdir -p \
  "${BUNDLE_DIR}/logs" \
  "${BUNDLE_DIR}/results/smoke/logs" \
  "${BUNDLE_DIR}/results/smoke/checkpoints" \
  "${NRL_MEGATRON_CHECKPOINT_DIR}"

COMMAND=$(cat <<EOF
set -euo pipefail
cd "\${GYM_DIR}"
"${TRAIN_PYTHON}" "\${GYM_DIR}/cluster/rdkit_no_tool_grpo/nemo_rl_assets/run_grpo_nemo_gym.py" \\
  --config="\${GYM_DIR}/cluster/rdkit_no_tool_grpo/rdkit_no_tool_grpo.yaml" \\
  ++cluster.num_nodes=${NUM_ACTOR_NODES} \\
  ++cluster.gpus_per_node=${GPUS_PER_NODE} \\
  ++grpo.max_num_epochs=1 \\
  ++grpo.max_num_steps=1 \\
  ++grpo.val_at_start=true \\
  ++grpo.val_at_end=false \\
  ++grpo.val_period=0 \\
  ++grpo.num_val_generations_per_prompt=1 \\
  ++data.train.data_path="\${GYM_DIR}/cluster/rdkit_no_tool_grpo/data/train_smoke.jsonl" \\
  ++data.validation.data_path="\${GYM_DIR}/cluster/rdkit_no_tool_grpo/data/test_smoke.jsonl" \\
  ++checkpointing.enabled=false \\
  ++checkpointing.checkpoint_dir="${BUNDLE_DIR}/results/smoke/checkpoints" \\
  ++logger.log_dir="${BUNDLE_DIR}/results/smoke/logs" \\
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
  --mem="${PREFLIGHT_MEM}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PREFLIGHT_PARTITION}" \
  --time=00:20:00 \
  --job-name=rdkit-es140-smoke-preflight \
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
  --time=04:00:00 \
  --job-name=rdkit-es140-lora-smoke \
  --output="${BUNDLE_DIR}/logs/%x-%j.out" \
  --error="${BUNDLE_DIR}/logs/%x-%j.err" \
  --dependency="afterok:${preflight_job_id}" \
  --export=ALL \
  --chdir="${GYM_DIR}" \
  "${RAY_SUB}")

echo "preflight: ${preflight_job_id}"
echo "smoke: ${job_id%%;*}"
echo "W&B run id: ${WANDB_RUN_ID}"
