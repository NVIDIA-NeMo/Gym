#!/bin/bash

export TRAINING_SCRIPT="Gym/resources_servers/workbench/scripts/train_qwen3_4b_nemorl_grpo.slurm"
export NAME="nemorl-grpo-qwen3-4b"
export NUM_NODES=1

sbatch \
    --nodes=${NUM_NODES} \
    --account=llmservice_modelalignment_ppo \
    --job-name=${NAME} \
    --partition=batch \
    --time=4:0:0 \
    --gres=gpu:8 \
    --exclusive \
    --dependency=singleton \
    ${TRAINING_SCRIPT}