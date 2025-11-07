# GRL Tetris Resource Server

FastAPI-based resource server that exposes the GRL Tetris environment through NeMo Gym conventions. The environment logic lives under `resources_servers/grl_tetris/env` and is a standalone adaptation of the upstream GRL implementation.

## Why it exists
- **Domain**: Classic falling-block Tetris on a configurable grid.
- **Evaluation**: Agents must clear at least one line; `/verify` rewards the cumulative score only when the environment reports success.
- **Independence**: No runtime dependency on the GRL repository—the environment is vendored and self-contained.

## Setup

Please follow the setup instructions as outlined in: https://github.com/NVIDIA-NeMo/Gym/blob/main/docs/tutorials/02-setup.md#step-1-clone-and-install. 

## Running
Launch the resource server together with a compatible agent:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/grl_tetris/configs/grl_tetris.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect sample rollouts once datasets are available:
```bash
ng_collect_rollouts +agent_name=grl_tetris_game_agent \
    +input_jsonl_fpath=resources_servers/grl_tetris/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/grl_tetris/data/example_rollouts.jsonl \
    +limit=5
```

# Launch the rollout viewer
```bash
ng_viewer +jsonl_fpath=resources_servers/grl_tetris/data/example_rollouts.jsonl
```

## Generating Test Examples for Reward Profiling

For CONTRIBUTING.md reward profiling requirements, generate ~500 diverse test examples with varying seeds and board dimensions:

```bash
cd resources_servers/grl_tetris
python generate_test_examples.py --num-examples 500
```

This creates `data/test_examples.jsonl` with diverse configurations:
- **Board sizes**: [4,4] to [6,6] with various aspect ratios
- **Box types**: 0-3 (single block, I/-, and I/-/O piece sets)
- **Seeds**: Randomized for unique, reproducible games

Use the generated test set for reward profiling (see next section).

## Automated Evaluation Pipeline (Qwen3-4B)

An automated evaluation pipeline is available for running comprehensive evaluations following CONTRIBUTING.md requirements (200 prompts × 16 rollouts = 3,200 total).

### Quick Start

**Recommended: Auto-retry loop** (handles crashes automatically):
```bash
cd resources_servers/grl_tetris
./run_qwen3_4b_eval_loop.sh
```

This will:
- Start vLLM with Qwen3-4B
- Start NeMo Gym servers
- Collect 3,200 rollouts (200 prompts × 16 repeats)
- Automatically resume from checkpoints if interrupted
- Generate reward analysis report

See [`data/qwen3_4b_eval/README.md`](data/qwen3_4b_eval/README.md) for full documentation.

### Features

- **Checkpoint & Resume**: Automatically detects partial progress and resumes from where it left off
- **Auto-Retry**: Handles Ray crashes and timeouts gracefully (up to 20 attempts)
- **Progress Tracking**: Shows rollout counts and completion percentage
- **Comprehensive Logging**: All logs saved to `data/qwen3_4b_eval/logs/`
- **Reward Analysis**: Automated profiling with statistics and correlation analysis

### Manual Run

For single execution without auto-retry:
```bash
cd resources_servers/grl_tetris
./run_qwen3_4b_eval.sh
```

If interrupted, simply run again—it will automatically resume from the last checkpoint.

### Configuration

Edit `run_qwen3_4b_eval.sh` to customize:
- Model: `MODEL_NAME="Qwen/Qwen3-4B"`
- GPU settings: `TENSOR_PARALLEL_SIZE`, `GPU_MEMORY_UTILIZATION`
- Rollout params: `temperature`, `top_p`, `max_output_tokens`

### Output

All results are saved to `data/qwen3_4b_eval/`:
- `rollouts.jsonl`: Collected rollouts
- `reward_analysis.md`: Statistics report
- `logs/`: Detailed logs for debugging

### Results Summary (Qwen3-4B)

**Dataset**: 3,200 rollouts (200 prompts × 16 repeats)

**Performance Metrics**:
- **Success Rate**: 5.09% (163/3,200 rollouts)
- **Mean Reward**: -0.29 (range: -2.00 to 19.20)
- **Median Reward**: -0.80

**Key Findings**:
- Most rollouts (21%) received reward of -0.90 (piece dropped without clearing lines)
- Successful line clears achieved rewards of ~9.0-9.2
- Average 7.48 tool calls per rollout
- Weak negative correlation between tool calls and reward (-0.06)

**Top Reward Distribution**:
- `-0.9`: 672 rollouts (21.0%) - piece dropped, no line clear
- `-0.8`: 603 rollouts (18.8%)
- `-0.7`: 495 rollouts (15.5%)
- `9.1`: 29 rollouts (0.9%) - successful line clear
- `8.9`: 26 rollouts (0.8%)

The relatively low success rate (5.09%) suggests that Tetris line-clearing is challenging for the model, requiring precise spatial reasoning and action sequencing. Most rollouts result in pieces dropping without clearing lines (negative rewards from -0.1 per action step).

See [`data/qwen3_4b_eval/reward-analysis.md`](data/qwen3_4b_eval/reward-analysis.md) for complete analysis.

## Running with vLLM for Reward Profiling (Manual Setup)

For reward profiling and RL training (as per CONTRIBUTING.md), use vLLM with local models like Qwen3-30B-A3B.

### 1. Start vLLM Server

**Prerequisites:**
```bash
pip install vllm hf_transfer
```

**Single GPU (A100 80GB):**
```bash
HF_HOME=.cache/ \
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240 \
    --max-model-len 8192 \
    --trust-remote-code
```

**Multi-GPU (4x GPUs):**
```bash
HF_HOME=.cache/ \
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240 \
    --max-model-len 8192 \
    --trust-remote-code
```

**Wait 2-5 minutes for model loading.** Verify server is ready:
```bash
curl http://localhost:10240/v1/models
```

### 2. Start NeMo Gym Agent with vLLM

In a new terminal:
```bash
# Set environment variables
export policy_base_url="http://localhost:10240/v1"
export policy_api_key="dummy"
export policy_model_name="Qwen/Qwen3-30B-A3B"

# Start agent with vLLM model configuration
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/grl_tetris/configs/grl_tetris.yaml"
ng_run "+config_paths=[${config_paths}]"
```

### 3. Collect Rollouts for Reward Profiling

**Using the test examples dataset (500 diverse games, 16 responses per prompt):**
```bash
ng_collect_rollouts +agent_name=grl_tetris_game_agent \
    +input_jsonl_fpath=resources_servers/grl_tetris/data/test_examples.jsonl \
    +output_jsonl_fpath=resources_servers/grl_tetris/data/test_rollouts.jsonl \
    +limit=null \
    +num_repeats=1 \
    +num_samples_in_parallel=128 \
    +responses_create_params.temperature=0.8 \
    +responses_create_params.max_output_tokens=4096
```

**For quick testing with example.jsonl:**
```bash
ng_collect_rollouts +agent_name=grl_tetris_game_agent \
    +input_jsonl_fpath=resources_servers/grl_tetris/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/grl_tetris/data/qwen3_30b_a3b_rollouts.jsonl \
    +limit=null \
    +num_repeats=1 \
    +num_samples_in_parallel=128 \
    +responses_create_params.temperature=0.8 \
    +responses_create_params.max_output_tokens=4096
```

**Parameters explained:**
- `num_repeats=16`: Generate 16 rollouts per prompt (required for reward profiling)
- `limit=null`: Process all prompts (or set to 500 for a subset)
- `num_samples_in_parallel=4`: Concurrent requests (adjust based on GPU capacity)
- `temperature=0.8`: Higher temperature for diverse responses

### 4. Analyze Reward Distribution

**Interactive viewer:**
```bash
ng_viewer +jsonl_fpath=resources_servers/grl_tetris/data/qwen3_30b_a3b_rollouts.jsonl
```

**Command-line analysis:**
```bash
# Reward distribution
jq '.reward' resources_servers/grl_tetris/data/qwen3_30b_a3b_rollouts.jsonl | sort -n | uniq -c

# Statistics (min, max, avg)
jq -s 'map(.reward) | {
    min: min,
    max: max,
    avg: (add / length),
    count: length
}' resources_servers/grl_tetris/data/qwen3_30b_a3b_rollouts.jsonl

# Success rate
jq -s 'map(select(.success == true)) | length' \
    resources_servers/grl_tetris/data/qwen3_30b_a3b_rollouts.jsonl

# Tool call metrics (average per rollout)
jq -s 'map([.output[] | select(.type == "function_call")] | length) | add / length' \
    resources_servers/grl_tetris/data/qwen3_30b_a3b_rollouts.jsonl
```

### Other Recommended Models

**For math/coding tasks:** `Qwen/Qwen3-235B-Thinking`  
**For agents/instruction following:** `Qwen/Qwen3-235B-Instruct`

Adjust `--tensor-parallel-size` based on available GPUs (235B models typically need 8 GPUs).

## Dataset artifacts
Placeholder files belong under `resources_servers/grl_tetris/data/` (e.g., `example.jsonl`, `example_rollouts.jsonl`, `example_metrics.json`). Replace them with actual rollouts and metrics during integration.

## Tests
```bash
pytest resources_servers/grl_tetris/tests
```

## Licensing
- Code: Apache 2.0
- Data: Apache 2.0
