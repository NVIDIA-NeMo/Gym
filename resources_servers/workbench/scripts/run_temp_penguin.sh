WANDB_API_KEY=<wandb api key> \
SLURM_ACCOUNT=llmservice_modelalignment_sft \
SLURM_PARTITION=batch_block1 \
EXP_NAME=penguin_grpo/qwen3_4binstruct/8nodes/grpo-qwen3-4b-instruct-pre-cot \
NUM_ACTOR_NODES=8 \
REPO_LOCATION=/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl \
    bash penguin_submit.sh \
    --config=examples/penguin/grpo_workbench_qwen3_4binstruct.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"


WANDB_API_KEY=<wandb api key> \
SLURM_ACCOUNT=llmservice_modelalignment_sft \
SLURM_PARTITION=batch_block1 \
EXP_NAME=penguin_grpo/qwen3_4binstruct/8nodes/grpo-qwen3-4b-instruct-pre-cot \
NUM_ACTOR_NODES=8 \
REPO_LOCATION=/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl \
    bash penguin_submit.sh \
    --config=examples/penguin/grpo_workbench_qwen3_4binstruct.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"


WANDB_API_KEY=<wandb api key> \
SLURM_ACCOUNT=llmservice_modelalignment_sft \
SLURM_PARTITION=batch_block1 \
EXP_NAME=penguin_grpo/qwen3_4binstruct/8nodes/grpo-qwen3-4b-instruct-pre-cot \
NUM_ACTOR_NODES=8 \
REPO_LOCATION=/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl \
    bash penguin_submit.sh \
    --config=examples/penguin/grpo_workbench_qwen3_4binstruct.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"