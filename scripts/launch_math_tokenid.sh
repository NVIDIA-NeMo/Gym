#!/bin/bash
# launch_math_tokenid.sh — launch on-policy RL training for a single agent harness
# using the online token-ID injection (forward_prefix_token_ids).
#
# Usage:
#   HARNESS=hermes bash scripts/launch_math_tokenid.sh
#   HARNESS=codex  GBS=256 bash scripts/launch_math_tokenid.sh
#
# Required env:
#   HARNESS          — one of: hermes codex opencode openclaw pi
#   NRL_TRAIN_PATH   — path to the training JSONL (Responses API format)
#   NRL_VAL_PATH     — path to the validation JSONL
#
# Optional env (defaults shown):
#   GBS              — global batch size (default 512 = 16 prompts x 32 gens)
#   MAX_STEPS        — max training steps (default 150)
#   VAL_PERIOD       — validate every N steps (default 10)
#   MAX_SEQ_LEN      — max sequence length in tokens (default 131072 = 128k)
#   WANDB_PROJECT    — wandb project (default cmunley-dfw-n35)
#   JOB_NAME         — SLURM job name (default math-$HARNESS-tokenid)
#   NUM_NODES        — number of GPU nodes (default 8)
#   PARTITION        — SLURM partition (default batch)
#   ACCOUNT          — SLURM account (default nemotron_rl_algo)
#   WALLTIME         — job walltime (default 04:00:00)

set -euo pipefail

HARNESS="${HARNESS:?must set HARNESS=hermes|codex|opencode|openclaw|pi}"
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:?must set NRL_TRAIN_PATH}"
NRL_VAL_PATH="${NRL_VAL_PATH:-}"

GBS="${GBS:-512}"
PPS="${PPS:-16}"  # prompts per step
GPP=$(( GBS / PPS ))  # gens per prompt

MAX_STEPS="${MAX_STEPS:-150}"
VAL_PERIOD="${VAL_PERIOD:-10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"
WANDB_PROJECT="${WANDB_PROJECT:-cmunley-dfw-n35}"
JOB_NAME="${JOB_NAME:-math-${HARNESS}-tokenid}"
NUM_NODES="${NUM_NODES:-8}"
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-nemotron_rl_algo}"
WALLTIME="${WALLTIME:-04:00:00}"

# resolve paths relative to this script's repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== launching math tokenid training ==="
echo "  harness:    $HARNESS"
echo "  train:      $NRL_TRAIN_PATH"
echo "  val:        ${NRL_VAL_PATH:-none}"
echo "  GBS:        $GBS (${PPS} prompts x ${GPP} gens)"
echo "  max_steps:  $MAX_STEPS  val_period: $VAL_PERIOD"
echo "  max_seq:    ${MAX_SEQ_LEN} tokens"
echo "  job:        $JOB_NAME  nodes: $NUM_NODES  partition: $PARTITION"
echo ""

# The launch script is in n35-rl-tokenid (or wherever NeMo-RL lives)
# Set NRL_DIR to your NeMo-RL checkout that mounts this Gym via USE_SNAPSHOT=0
NRL_DIR="${NRL_DIR:-$(pwd)}"

# build the config path for this harness
CFG="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
CFG="$CFG,resources_servers/math_with_judge/configs/math_with_judge.yaml"
CFG="$CFG,resources_servers/math_with_judge/configs/math_with_judge_${HARNESS}_agent.yaml"

# judge overrides (use policy model as judge placeholder, judge off)
JUDGE_ARGS=(
  "~env.nemo_gym.abstention"
  "~env.nemo_gym.jailbreak_detection"
  "~env.nemo_gym.inverse_if"
  "~env.nemo_gym.multichallenge"
  "~env.nemo_gym.equivalence_llm_judge"
  "~env.nemo_gym.genrm_compare_resources_server"
  "~env.nemo_gym.lc_judge"
  "~env.nemo_gym.code_gen"
  "~env.nemo_gym.math_formal_lean_refinement_agent"
  "++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model"
  "++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.should_use_judge=false"
  "++env.nemo_gym.policy_model.responses_api_models.vllm_model.forward_prefix_token_ids=true"
)

GBS_ARGS=(
  "grpo.num_prompts_per_step=${PPS}"
  "grpo.num_generations_per_prompt=${GPP}"
  "policy.train_global_batch_size=${GBS}"
  "grpo.val_period=${VAL_PERIOD}"
)

SEQ_ARGS=(
  "policy.max_total_sequence_length=${MAX_SEQ_LEN}"
  "policy.generation.max_new_tokens=${MAX_SEQ_LEN}"
  "policy.generation.vllm_cfg.max_model_len=${MAX_SEQ_LEN}"
)

SKIP_ARGS=(
  "++env.nemo_gym.skip_venv_if_present=true"
)

export JOB_NAME WANDB_NAME="${JOB_NAME}-$(date +%s)"
export USE_SNAPSHOT=0
export NUM_TOTAL_NODES=$NUM_NODES
export NRL_MAX_STEPS=$MAX_STEPS
export NRL_TRAIN_PATH
export NRL_VAL_PATH
export WANDB_PROJECT
export CFG_DEEPDIVE="$CFG"
export WALLTIME PARTITION SLURM_ACCOUNT="$ACCOUNT"

cd "$NRL_DIR"
bash scripts/launch_nano35_dfw_dapomath.sh \
  "${JUDGE_ARGS[@]}" \
  "${GBS_ARGS[@]}" \
  "${SEQ_ARGS[@]}" \
  "${SKIP_ARGS[@]}"
