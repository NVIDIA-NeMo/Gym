(training-nemo-rl-grpo-single-node-training)=

# Single Node Training

Now that you've completed the {doc}`Setup Instructions <setup>`, you're ready to launch a single-node training run!

**Estimated Time:** ~2-4 hours

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

```bash
# Set experiment name with timestamp
EXP_NAME="$(date +%Y%m%d)/nemo_gym_grpo/nemotron_nano_v2_9b/workplace_assistant_001"

# Configuration file path
CONFIG_PATH=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Launch training
# Set these environment variables before running:
#   HF_TOKEN: Your Hugging Face token for model downloads
#   WANDB_API_KEY: Your Weights & Biases API key for logging
TORCH_CUDA_ARCH_LIST="9.0 10.0" \
HF_HOME=$PWD/.cache/ \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY={your W&B API key} \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="${USER}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    ++policy.generation.vllm_cfg.tool_parser_plugin=$(find $PWD/.cache -name nemotron_toolcall_parser_no_streaming.py) \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.max_num_steps=3 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &> results/$EXP_NAME/output.log &

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

## Expected Results

Monitor these metrics in W&B to track progress:
1. `train:reward_mean`: The average reward of your model on this training environment. The reward may be noisy, but it should go up.
2. `val:accuracy`: The validation performance of your model on this training environment. This should go up steadily.

The best checkpoint (highest `val:accuracy`) is retained based on `checkpointing.keep_top_k: 3`. You can find checkpoints at the following path:
```bash
ls results/$EXP_NAME
```


Training is successful when:
- Reward mean increases consistently over steps.
- Validation accuracy consistently improves.
- No OOM (Out of Memory) errors.
- Checkpoints are saved at specified intervals.

---

## Measuring Real-World Improvement

The Workplace Assistant environment's tool-calling tasks correlate with performance on the [Berkeley Function Calling Leaderboard (BFCL) v3](https://gorilla.cs.berkeley.edu/leaderboard.html) benchmark. To measure improvement, evaluate the Nemotron Nano v2 9B model on BFCL v3 before and after training, and compare the results. You should observe measurable improvement in tool-calling accuracy.

You can run BFCL v3 evaluations using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator), which supports BFCL v3. Refer to the [NeMo Evaluator docs](https://github.com/NVIDIA-NeMo/Evaluator#-supported-benchmarks-and-evaluation-harnesses) for full setup instructions and supported benchmarks.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| HuggingFace rate limits | Specify your HF API token and/or add `HF_HUB_OFFLINE=1` after the initial download |
| vLLM process not shutting down | Run `pkill -f VllmAsyncGenerationWorker` before training |
| Ray cluster issues | Run `ray stop --force` before training |
| CUDA OOM | Increase `tensor_parallel_size`, lower batch sizes |
| Slow initial startup | Set `NRL_FORCE_REBUILD_VENVS=true` on first run only; if `uv` gets rate limited, set this back to `false` |
