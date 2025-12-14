(training-nemo-rl-grpo-single-node-training)=

# Single Node Training

Now that you've completed the {doc}`Setup Instructions <setup>`, you're ready to launch a single-node training run!

**Estimated Time:** ~15 minutes

Run these commands **from inside the container**.

:::{tip}
Coming back from a break on a pre-existing filesystem setup? Just run these commands below once you enter the container.
```bash
source /opt/nemo_rl_venv/bin/activate
uv sync --group={build,docs,dev,test} --extra nemo_gym
uv run nemo_rl/utils/prefetch_venvs.py
```
:::

Download NVIDIA [Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2).
```bash
HF_HOME=$PWD/.cache/ \
HF_TOKEN={your HF token} \
    hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2
```

The Nemotron Nano 9B v2 model uses a custom chat template that we need to modify in order to support modern RL training. This step will setup the chat template we will be using for training. Please note that we are modifying the cached version of the chat template!
```bash
tokenizer_config_path=$(find $PWD/.cache/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2 -name tokenizer_config.json)
sed -i 's/enable_thinking=true/enable_thinking=false/g' $tokenizer_config_path
sed -i 's/{%- if messages\[-1\]\['\''role'\''\] == '\''assistant'\'' -%}{%- set ns.last_turn_assistant_content = messages\[-1\]\['\''content'\''\].strip() -%}{%- set messages = messages\[:-1\] -%}{%- endif -%}//g' $tokenizer_config_path
```

Clean up any existing or leftover Ray/vLLM processes
```bash
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"
```

Run training! By default we only training for 3 steps `grpo.max_num_steps=3` as a small test run in preparation for multi-node.

If you are using a single node for the full training run, you can remove this value. The full training will take several hours.
```bash
# Set experiment name with timestamp
EXP_NAME="$(date +%Y%m%d)/nemo_gym_grpo/nemotron_nano_v2_9b/workplace_assistant_001"
mkdir -p results/$EXP_NAME

# Configuration file path
CONFIG_PATH=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Launch training
# Set these environment variables before running:
#   WANDB_API_KEY: Your Weights & Biases API key for logging
#   logger.wandb.project: Fill in your username
TORCH_CUDA_ARCH_LIST="9.0 10.0" \
HF_HOME=$PWD/.cache/ \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY={your W&B API key} \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config=$CONFIG_PATH \
    ++logger.wandb.project="${Your Username}-nemo-gym-rl-integration" \
    ++logger.wandb.name=$EXP_NAME \
    ++logger.log_dir=results/$EXP_NAME \
    ++policy.generation.vllm_cfg.tool_parser_plugin=$(find $PWD/.cache -name nemotron_toolcall_parser_no_streaming.py) \
    ++grpo.max_num_steps=3 \
    ++checkpointing.checkpoint_dir=results/$EXP_NAME &> results/$EXP_NAME/output.log &

# Watch the logs
tail -f results/$EXP_NAME/output.log
```

:::{tip}
The end of the command above will do the following
```bash
&> results/$EXP_NAME/output.log &
```

1. `&> results/$EXP_NAME/output.log`: Pipe the terminal outputs into a file at `results/$EXP_NAME/output.log` that you can view.
2. `&`: This final ampersand will run the job in the background, which frees up your terminal to do other things. You can view all the background jobs using the `jobs` command. If you need to quit the training run, you can use the `fg` command to bring the job from the background into the foreground and then ctrl+c like normal!
:::

**Success criteria**: Take 3 steps on single node without any issues.
