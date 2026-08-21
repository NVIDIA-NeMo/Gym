#!/usr/bin/env bash
set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
GYM_MAIN_DIR=${GYM_MAIN_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
RAY_SUB="${BUNDLE_DIR}/nemo_rl_assets/ray.sub"
TRAIN_PYTHON=${TRAIN_PYTHON:-/opt/nemo_rl_venv/bin/python}

SLURM_ACCOUNT=${SLURM_ACCOUNT:-healthcareeng_research}
SLURM_PARTITION=${SLURM_PARTITION:-pool0}
NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-01:00:00}
START_DEPENDENCY=${START_DEPENDENCY:-}
CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0-gym-peft.sqsh}
export ES140_MERGED_MODEL_DIR=${ES140_MERGED_MODEL_DIR:-${GYM_MAIN_DIR}/artifacts/nemotron3-nano-rdkit-es140-merged-hf}
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR:-${BUNDLE_DIR}/results/model_artifacts/megatron}

RESULTS_DIR=${RESULTS_DIR:-${BUNDLE_DIR}/results/conversion_rollout_gate}
VALIDATION_SUMMARY=${VALIDATION_SUMMARY:-${RESULTS_DIR}/validation_summary.json}
HF_OVERRIDE_HASH=705d80b18ca9
MODEL_CACHE_KEY="model_${ES140_MERGED_MODEL_DIR//\//_}__hfovr_${HF_OVERRIDE_HASH}"
EXPECTED_RUN_CONFIG="${NRL_MEGATRON_CHECKPOINT_DIR}/${MODEL_CACHE_KEY}/iter_0000000/run_config.yaml"

if [[ "${NUM_ACTOR_NODES}" -ne 4 || "${GPUS_PER_NODE}" -ne 8 ]]; then
  echo "The conversion-rollout gate requires the validated 4x8 GPU layout" >&2
  exit 1
fi
for required in \
  "${CONTAINER_IMAGE_PATH}" \
  "${ES140_MERGED_MODEL_DIR}" \
  "${BUNDLE_DIR}/data/test.jsonl"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${BUNDLE_DIR}/logs" "${RESULTS_DIR}/logs" "${NRL_MEGATRON_CHECKPOINT_DIR}"

COMMAND=$(cat <<EOF
set -euo pipefail
cd "\${GYM_DIR}"
ONE_ROLLOUT_DATA="\${GYM_DIR}/cluster/rdkit_no_tool_grpo/data/conversion_rollout_one.jsonl"
sed -n '1p' "\${GYM_DIR}/cluster/rdkit_no_tool_grpo/data/test.jsonl" > "\${ONE_ROLLOUT_DATA}"
test "\$(wc -l < "\${ONE_ROLLOUT_DATA}")" -eq 1
"${TRAIN_PYTHON}" "\${GYM_DIR}/cluster/rdkit_no_tool_grpo/nemo_rl_assets/run_grpo_nemo_gym.py" \
  --config="\${GYM_DIR}/cluster/rdkit_no_tool_grpo/rdkit_no_tool_grpo.yaml" \
  --validation-only \
  --validation-summary="${VALIDATION_SUMMARY}" \
  ++cluster.num_nodes=${NUM_ACTOR_NODES} \
  ++cluster.gpus_per_node=${GPUS_PER_NODE} \
  ++grpo.max_num_epochs=1 \
  ++grpo.max_num_steps=1 \
  ++grpo.val_at_start=false \
  ++grpo.val_at_end=false \
  ++grpo.val_period=0 \
  ++data.validation.data_path="\${ONE_ROLLOUT_DATA}" \
  ++checkpointing.enabled=false \
  ++logger.wandb_enabled=false \
  ++logger.tensorboard_enabled=false \
  ++logger.log_dir="${RESULTS_DIR}/logs"
test -f "${EXPECTED_RUN_CONFIG}"
test -f "${VALIDATION_SUMMARY}"
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
job_id=$(sbatch \
  --parsable \
  --nodes="${NUM_ACTOR_NODES}" \
  --gpus-per-node="${GPUS_PER_NODE}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --time="${JOB_TIME_LIMIT}" \
  --job-name=rdkit-es140-convert-rollout \
  --output="${BUNDLE_DIR}/logs/%x-%j.out" \
  --error="${BUNDLE_DIR}/logs/%x-%j.err" \
  "${dependency[@]}" \
  --export=ALL \
  --chdir="${GYM_DIR}" \
  "${RAY_SUB}")

echo "conversion-rollout gate: ${job_id%%;*}"
echo "Megatron run config: ${EXPECTED_RUN_CONFIG}"
echo "Rollout summary: ${VALIDATION_SUMMARY}"
