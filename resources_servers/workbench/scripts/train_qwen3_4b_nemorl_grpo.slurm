#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --account=ACCOUNT
#SBATCH --job-name=JOB_NAME
#SBATCH --partition=PARTITION
#SBATCH --time=4:0:0
#SBATCH --dependency=singleton
#SBATCH --job-name=traj-coll-sft-gpt-oss-120b-workbench-nemogym
#SBATCH --ntasks=1
#SBATCH --output=logs_slurm/train_nemorl_grpo_qwen3_4b/%A_%a_%x.out
#SBATCH --gres=gpu:8


pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"
EXP_NAME="$(date +%Y%m%d)/workbench/qwen3_4binstruct/workbench-test"
CONFIG_PATH=examples/penguin/grpo_workbench_qwen3_4binstruct.yaml
HF_HOME=.cache/ \
WANDB_API_KEY=ac17639e76dcbd0dbb3aea36aec1ae8286de56bc \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="$USER-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    grpo.val_at_start=false \
    ++grpo.num_prompts_per_step=8 \
    ++grpo.max_num_steps=300 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    ++max_val_samples=544 \
    ++val_period=1 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &