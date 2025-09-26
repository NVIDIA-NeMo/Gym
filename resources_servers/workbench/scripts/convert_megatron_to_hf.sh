set -euo pipefail

REPO_PATH=/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl
CKPT_DIR=/lustre/fsw/portfolios/llmservice/users/abhibhag/nemo-rl/results/20250924/workbench/qwen3_4binstruct/workbench-test/step_57

# srun -A llmservice_modelalignment_sft \
#   --container-image=/lustre/fsw/portfolios/llmservice/users/jkyi/containers/nemo-rl:main-2aea5add.squashfs \
#   --container-mounts=/lustre:/lustre \
#   --no-container-mount-home \
#   --gpus-per-node=8 \
#   -p batch_short \
#   -t 02:00:00 \
#   --pty bash -c "
#     cd $REPO_PATH
#     uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
#       --config="${BASE_PATH}/config.yaml" \
#       --megatron-ckpt-path="${BASE_PATH}/policy/weights" \
#       --hf-ckpt-path=$HF_CKPT_PATH
#     cp \"${BASE_PATH}/config.yaml\" \"${HF_CKPT_PATH}/config.yaml\"
# "

srun -A llmservice_modelalignment_sft \
  --container-image=/lustre/fsw/portfolios/llmservice/users/jkyi/containers/nemo-rl:main-2aea5add.squashfs \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  --gpus-per-node=8 \
  -p batch_short \
  -t 02:00:00 \
  --pty bash -c "
    cd $REPO_PATH
    uv run --extra mcore python  examples/converters/convert_dcp_to_hf.py  \
      --config="${CKPT_DIR}/config.yaml" \
      --dcp-ckpt-path="${CKPT_DIR}/policy/weights" \
      --hf-ckpt-path="${CKPT_DIR}-hf""


# # Change to your appropriate checkpoint directory
# CKPT_DIR=results/sft/step_10

# uv run examples/convert_dcp_to_hf.py --config=$CKPT_DIR/config.yaml --dcp-ckpt-path=$CKPT_DIR/policy/weights --hf-ckpt-path=${CKPT_DIR}-hf
# rsync -ahP $CKPT_DIR/policy/tokenizer ${CKPT_DIR}-hf/