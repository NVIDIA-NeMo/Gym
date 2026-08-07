#!/bin/bash
# Reference launcher: DAPO math x OpenCode-in-sandbox, NeMo-RL GRPO on Slurm.
# Sanitized -- every site-specific value comes from the environment. See README.md.
set -euo pipefail

# ---- required ---------------------------------------------------------------
: "${JOB_NAME:?set JOB_NAME (drives both the singleton key and the checkpoint dir)}"
: "${MODEL:?set MODEL: path to the HF-format policy model}"
: "${TRAIN:?set TRAIN: path to the training jsonl}"
: "${NRL_DIR:?set NRL_DIR: NeMo-RL checkout}"
: "${GYM_DIR:?set GYM_DIR: this Gym checkout}"
: "${CONTAINER:?set CONTAINER: container image (.sqsh)}"
: "${OPENSANDBOX_DOMAIN:?export OPENSANDBOX_DOMAIN}"
: "${OPENSANDBOX_API_KEY:?export OPENSANDBOX_API_KEY}"
: "${WANDB_API_KEY:?export WANDB_API_KEY (or disable the logger below)}"

# ---- sizing -----------------------------------------------------------------
NODES="${NODES:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GBS="${GBS:-256}"           # global batch; must equal PPS * GPP
PPS="${PPS:-16}"            # prompts per step
GPP="${GPP:-16}"            # generations per prompt -- too few and whole groups
                            # come back all-wrong, giving zero advantage
CONC="${CONC:-$GBS}"        # keep agent concurrency == gbs
SEQ="${SEQ:-49152}"         # >= the largest max_output_tokens any CLI asks for
MAX_STEPS="${MAX_STEPS:-500}"
WALL="${WALL:-04:00:00}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"

# Parallelism. MoE models want TP/EP > 1; a dense model wants all 1s (and MTP off).
TP="${TP:-4}"; EP="${EP:-8}"; PP="${PP:-1}"; CP="${CP:-8}"

# Ray sizes its workers from this -- must not exceed the node's CPU count.
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-128}"

ACCOUNT="${ACCOUNT:-default}"
PARTITION="${PARTITION:-batch}"
BASE="${BASE:-$PWD}"
LOG_DIR="$BASE/logs/$JOB_NAME"
CKPT="$BASE/results/$JOB_NAME/checkpoints"
CACHE="$BASE/.cache/$JOB_NAME"
mkdir -p "$LOG_DIR" "$CKPT" "$CACHE"
export BASE_LOG_DIR="$LOG_DIR" LOG_DIR GPUS_PER_NODE

CT=/opt/nemo-rl
export MOUNTS="$NRL_DIR/nemo_rl:$CT/nemo_rl,$NRL_DIR/examples:$CT/examples,$GYM_DIR:$CT/3rdparty/Gym-workspace/Gym"

# Chain: math_with_judge (verifier + simple agent) then this recipe's env config.
CFG="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
CFG="$CFG,resources_servers/math_with_judge/configs/math_with_judge.yaml"
CFG="$CFG,recipes/dapo_math_opencode/config.yaml"

export COMMAND="cd $CT && \
NEMO_GYM_VENV_DIR=$CACHE/gym_venvs HF_HOME=$CACHE/hf_home UV_CACHE_DIR=$CACHE/uv \
OPENSANDBOX_DOMAIN=$OPENSANDBOX_DOMAIN OPENSANDBOX_API_KEY=$OPENSANDBOX_API_KEY \
uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config ${NRL_CONFIG:-examples/configs/grpo_math_8B.yaml} \
  policy.model_name=$MODEL policy.tokenizer.name=$MODEL \
  cluster.num_nodes=$NODES cluster.gpus_per_node=$GPUS_PER_NODE \
  policy.megatron_cfg.tensor_model_parallel_size=$TP \
  policy.megatron_cfg.expert_model_parallel_size=$EP \
  policy.megatron_cfg.pipeline_model_parallel_size=$PP \
  policy.megatron_cfg.context_parallel_size=$CP \
  policy.max_total_sequence_length=$SEQ policy.generation.max_new_tokens=$SEQ \
  policy.generation.vllm_cfg.max_model_len=$SEQ \
  policy.generation.vllm_cfg.tensor_parallel_size=$TP \
  policy.generation.colocated.enabled=true \
  policy.train_global_batch_size=$GBS \
  grpo.num_prompts_per_step=$PPS grpo.num_generations_per_prompt=$GPP \
  grpo.val_period=0 grpo.max_num_steps=$MAX_STEPS \
  '++env.nemo_gym.config_paths=[$CFG]' \
  '++env.nemo_gym.sandbox_opencode_math.responses_api_agents.sandbox_agent.concurrency=$CONC' \
  env.nemo_gym.nemo_gym_log_dir=$LOG_DIR/nemo_gym \
  data.train.data_path=$TRAIN data.validation.data_path=$TRAIN \
  checkpointing.checkpoint_dir=$CKPT checkpointing.save_period=$SAVE_PERIOD \
  logger.log_dir=$LOG_DIR logger.wandb_enabled=True logger.wandb.name=$JOB_NAME"

# Same JOB_NAME + singleton => chunks serialize and each resumes from the last checkpoint.
sbatch --dependency=singleton \
  --nodes="$NODES" --account="$ACCOUNT" --partition="$PARTITION" --time="$WALL" \
  --job-name="$JOB_NAME" --exclusive --mem=0 \
  --output="$LOG_DIR/%j.out" --error="$LOG_DIR/%j.err" \
  "$NRL_DIR/ray.sub"
