(training-nemo-rl-grpo-single-node-training)=

# Single Node Training

With your environment set up and data prepared, you're ready to run training. But before committing to a multi-hour, multi-node job, it's important to verify everything works correctly on a single node first.

:::{card}

**Goal**: Run a single-node GRPO training session to validate your environment.

^^^

**In this section, you will**:

1. Download the Nemotron Nano 9B v2 model
2. Configure the model's chat template
3. Clean up existing processes
4. Run a test training session with 3 steps

:::

:::{button-ref} training-nemo-rl-grpo-setup
:color: secondary
:outline:
:ref-type: ref

← Previous: Setup
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed the {doc}`Setup <setup>` instructions
- ✅ Access to a running container session with GPUs
- ✅ (Optional) Weights & Biases API key for experiment tracking

:::{tip}
Coming back from a break on a pre-existing filesystem setup? Run these commands once you enter the container:

```bash
source /opt/nemo_rl_venv/bin/activate
uv sync --group={build,docs,dev,test} --extra nemo_gym
uv run nemo_rl/utils/prefetch_venvs.py
```
:::

---

## 1. Download the Model

**Estimated time**: ~5-10 minutes

Download NVIDIA [Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2):

```bash
HF_HOME=$PWD/.cache/ \
HF_TOKEN={your HF token} \
    hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2
```

**✅ Success Check**: Model files are downloaded to `.cache/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2/`.

---

## 2. Configure the Chat Template

**Estimated time**: ~1 minute

The Nemotron Nano 9B v2 model uses a custom chat template that must be modified for RL training. This step modifies the cached version of the chat template:

```bash
tokenizer_config_path=$(find $PWD/.cache/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2 -name tokenizer_config.json)
sed -i 's/enable_thinking=true/enable_thinking=false/g' $tokenizer_config_path
sed -i 's/{%- if messages\[-1\]\['\''role'\''\] == '\''assistant'\'' -%}{%- set ns.last_turn_assistant_content = messages\[-1\]\['\''content'\''\].strip() -%}{%- set messages = messages\[:-1\] -%}{%- endif -%}//g' $tokenizer_config_path
```

**✅ Success Check**: The `sed` commands complete without errors.

---

## 3. Clean Up Existing Processes

**Estimated time**: ~1 minute

Clean up any existing or leftover Ray/vLLM processes:

```bash
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"
```

**✅ Success Check**: Commands complete without errors. It is okay if some processes are not found.

---

## 4. Run Training

**Estimated time**: ~15-30 minutes

By default, this runs only 3 training steps (`grpo.max_num_steps=3`) as a small test run in preparation for multi-node training. If you are using a single node for the full training run, you can remove this value. The full training will take several hours.

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
The end of the command above does the following:

```bash
&> results/$EXP_NAME/output.log &
```

1. `&> results/$EXP_NAME/output.log`: Pipes the terminal outputs into a file at `results/$EXP_NAME/output.log` that you can view.
2. `&`: This final ampersand runs the job in the background, which frees up your terminal to do other things. You can view all the background jobs using the `jobs` command. If you need to quit the training run, you can use the `fg` command to bring the job from the background into the foreground and then Ctrl+C like normal.
:::

**✅ Success Check**: Training completes 3 steps on single node without any issues. Check the logs for errors and verify that training steps are progressing.

---

:::{button-ref} training-nemo-rl-grpo-multi-node-training
:color: primary
:ref-type: ref

Next: Multi-Node Training →
:::
