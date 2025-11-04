# Qwen3 30B-A3B Evaluation for GRL Sokoban

This directory contains the evaluation setup and results for the Qwen3 30B-A3B model on the GRL Sokoban task, following CONTRIBUTING.md requirements.

## Evaluation Configuration

### Dataset
- **Input**: 200 diverse Sokoban puzzles from `test_examples_200.jsonl`
- **Source**: First 200 examples from `test_examples.jsonl` (500 total)
- **Coverage**: Various room sizes [4×4 to 8×8], different box counts (1-3 boxes), randomized seeds

### Inference Setup
- **Model**: Qwen3-30B-A3B 
- **Engine**: vLLM for efficient inference
- **Rollouts per prompt**: 16
- **Total rollouts**: 3,200 (200 prompts × 16 rollouts)
- **Temperature**: 0.8
- **Max output tokens**: 4,096
- **Parallel samples**: 16

## How The Scripts Work

Two automated scripts are available for running evaluations:

### `run_qwen3_30b_eval.sh` (Single Run)

The main evaluation script orchestrates the entire evaluation pipeline:

1. **Cleanup**: Stops any existing vLLM/Ray processes
2. **vLLM Server**: Launches model server with Qwen3-30B-A3B (reuses if already running)
3. **NeMo Gym Servers**: Starts environment and agent servers
4. **Checkpoint Check**: Detects partial rollouts and creates resume checkpoint
5. **Rollout Collection**: Generates 3,200 rollouts (200 prompts × 16 repeats)
6. **Reward Analysis**: Generates comprehensive evaluation report
7. **Cleanup**: Keeps vLLM server running by default (use `KEEP_VLLM=false` to stop)

### `run_qwen3_30b_eval_loop.sh` (Auto-Retry Loop)

A wrapper script that provides automatic crash recovery:

1. **Progress Check**: Counts existing rollouts to determine completion status
2. **Run Main Script**: Executes `run_qwen3_30b_eval.sh`
3. **Handle Failures**: If the script crashes (Ray timeout, OOM, etc.):
   - Reports current progress
   - Waits 10 seconds for cleanup
   - Automatically restarts the evaluation
4. **Resume from Checkpoint**: The main script's checkpoint system handles resuming
5. **Success Detection**: Stops when 3,200 rollouts are collected
6. **Max Attempts**: Stops after 20 retry attempts to prevent infinite loops

**Use Case**: Ideal for long-running evaluations where Ray or vLLM may crash intermittently. The loop ensures the evaluation completes even with multiple failures.

### Checkpoint/Resume System

The script includes **automatic checkpoint/resume** functionality:

- **If the script fails** partway through rollout collection, simply **run it again**
- The script automatically detects partial rollouts and resumes from where it left off
- No manual intervention needed - it handles everything automatically

**How it works:**
1. Script checks if `rollouts.jsonl` already exists
2. If partial (< 3200 rollouts), it analyzes which prompts are incomplete
3. Creates `remaining_prompts.jsonl` with only the needed prompts
4. Backs up existing rollouts before resuming
5. Continues collection, appending to the same output file

**Example resume scenario:**
```bash
# First run - fails after 1500/3200 rollouts
./run_qwen3_30b_eval.sh
# [Script stops due to error]

# Second run - automatically resumes
./run_qwen3_30b_eval.sh
# [INFO] Found 1500 existing rollouts (expected 3200)
# [INFO] Creating checkpoint to resume from remaining prompts...
# [INFO] Resuming collection with 106 remaining prompts
# [continues from where it left off]
```

### vLLM Server Reuse

The script automatically detects and reuses an existing vLLM server if one is already running on port 10240. This saves significant time (15-30 minutes) on subsequent runs since the model doesn't need to be reloaded.

- **First run**: Loads the model (15-30 minutes for 30B on 4 GPUs)
- **Subsequent runs**: Reuses existing server (saves 15-30 minutes)
- **Force restart**: Set `REUSE_VLLM=false` to force a fresh server start
- **Stop server**: Set `KEEP_VLLM=false` or use Ctrl+C to force quit

## Quick Start

### Prerequisites

1. **Install Dependencies**:
```bash
# From Gym root directory
cd Gym
uv pip install -e ".[dev]"

# Install vLLM and dependencies
pip install vllm hf_transfer

# Install Sokoban-specific requirements
uv pip install -r resources_servers/grl_sokoban/requirements.txt
```

2. **Verify Dataset**:
```bash
cd resources_servers/grl_sokoban
wc -l data/qwen3_30b_eval/test_examples_200.jsonl
# Should output: 200
```

### Running the Evaluation

**Option 1: Auto-Retry Loop (Recommended)**:

For maximum reliability with automatic crash recovery:
```bash
cd resources_servers/grl_sokoban
./run_qwen3_30b_eval_loop.sh
```

This wrapper script:
- Automatically retries if Ray crashes or timeouts occur
- Resumes from checkpoints after each failure
- Continues until all 3,200 rollouts are collected
- Stops after 20 attempts or successful completion
- Perfect for long-running evaluations on unstable systems

**Option 2: Single Run**:

For a single execution attempt:
```bash
cd resources_servers/grl_sokoban
./run_qwen3_30b_eval.sh
```

The script automatically:
1. Cleans up any existing processes
2. Reuses existing vLLM server if available (saves 15-30 minutes)
3. Launches vLLM server with Qwen3 30B-A3B (if needed)
4. Starts NeMo Gym servers
5. Collects 3,200 rollouts (200 prompts × 16 rollouts)
6. Analyzes reward distribution
7. Generates comprehensive report

**Expected Runtime**:
- Single GPU: Not recommended (model too large)
- Multi-GPU (4× A100 80GB): ~3-6 hours (depending on GPU and model loading time)
  - First run: +15-30 minutes for model loading
  - Subsequent runs: Reuses existing server (saves time)

**When to Use Each Option**:
- Use `run_qwen3_30b_eval_loop.sh` if you expect Ray crashes or want unattended execution
- Use `run_qwen3_30b_eval.sh` for single runs or debugging
- Both scripts support automatic checkpoint/resume if interrupted

### Manual Step-by-Step Execution

If you prefer manual control, follow these steps:

#### 1. Start vLLM Server

**Multi-GPU (4× A100 80GB) - Recommended**:
```bash
cd resources_servers/grl_sokoban
HF_HOME=.cache/ \
vllm serve Qwen/Qwen3-30B-A3B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240 \
    --max-model-len 32768 \
    --trust-remote-code
```

Wait 15-30 minutes for model loading, then verify:
```bash
curl http://localhost:10240/v1/models
```

#### 2. Start NeMo Gym Servers

In a new terminal:
```bash
export policy_base_url="http://localhost:10240/v1"
export policy_api_key="dummy"
export policy_model_name="Qwen/Qwen3-30B-A3B"

ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_sokoban/configs/grl_sokoban.yaml]"
```

Wait for: `All 3 / 3 servers ready!`

#### 3. Collect Rollouts

In another terminal:
```bash
ng_collect_rollouts \
    +agent_name=grl_sokoban_game_agent \
    +input_jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_30b_eval/test_examples_200.jsonl \
    +output_jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_30b_eval/rollouts.jsonl \
    +limit=null \
    +num_repeats=16 \
    +num_samples_in_parallel=16 \
    +responses_create_params.temperature=0.8 \
    +responses_create_params.max_output_tokens=4096
```

#### 4. Analyze Results

```bash
cd resources_servers/grl_sokoban

python analyze_rewards.py \
    --rollouts-path data/qwen3_30b_eval/rollouts.jsonl \
    --model-name "Qwen3-30B-A3B" \
    --output data/qwen3_30b_eval/reward_analysis.md

# View the report
cat data/qwen3_30b_eval/reward_analysis.md
```

## Output Files

After running the evaluation, this directory will contain:

```
qwen3_30b_eval/
├── README.md                    # This file
├── test_examples_200.jsonl      # Input prompts (200 puzzles)
├── rollouts.jsonl               # Generated rollouts (3,200 entries)
├── reward_analysis.md           # Comprehensive reward analysis report
└── logs/                        # Execution logs
    ├── vllm_server.log          # vLLM server output
    ├── nemo_gym_servers.log     # NeMo Gym server output
    ├── rollout_collection.log   # Rollout collection output
    └── reward_analysis.log      # Analysis script output
```

## Viewing Results

### Interactive Rollout Viewer

Launch the web-based viewer to explore individual rollouts:
```bash
ng_viewer +jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_30b_eval/rollouts.jsonl
```

### Command-Line Analysis

**Quick reward statistics**:
```bash
jq -s 'map(.reward) | {
    min: min,
    max: max,
    avg: (add / length),
    median: (sort | if length % 2 == 0 then (.[length/2 - 1] + .[length/2]) / 2 else .[length/2] end),
    count: length
}' data/qwen3_30b_eval/rollouts.jsonl
```

**Success rate**:
```bash
jq -s 'map(select(.success == true)) | length' data/qwen3_30b_eval/rollouts.jsonl
```

**Reward distribution**:
```bash
jq '.reward' data/qwen3_30b_eval/rollouts.jsonl | sort -n | uniq -c | sort -rn
```

**Tool call metrics**:
```bash
jq -s 'map([.output[] | select(.type == "function_call")] | length) | {
    avg: (add / length),
    min: min,
    max: max
}' data/qwen3_30b_eval/rollouts.jsonl
```

## Actual Results

Results from running Qwen3-30B-A3B on 3,200 rollouts (200 prompts × 16 rollouts):

*Results will be populated after running the evaluation.*

## Manual Checkpoint/Resume

If you want to manually create a checkpoint (without running the full script):

```bash
cd resources_servers/grl_sokoban

# Check what's left to do
python checkpoint_resume_rollouts.py \
    --input data/qwen3_30b_eval/test_examples_200.jsonl \
    --rollouts data/qwen3_30b_eval/rollouts.jsonl \
    --output data/qwen3_30b_eval/remaining_prompts.jsonl \
    --target-repeats 16

# Then collect only remaining rollouts
ng_collect_rollouts \
    +agent_name=grl_sokoban_game_agent \
    +input_jsonl_fpath=data/qwen3_30b_eval/remaining_prompts.jsonl \
    +output_jsonl_fpath=data/qwen3_30b_eval/rollouts.jsonl \
    +limit=null \
    +num_repeats=16 \
    +num_samples_in_parallel=16 \
    +responses_create_params.temperature=0.8 \
    +responses_create_params.max_output_tokens=4096
```

## Troubleshooting

### Common Issues

**vLLM server fails to start**:
```bash
# Check GPU availability
nvidia-smi

# Check port availability
lsof -i :10240

# Check logs
tail -f data/qwen3_30b_eval/logs/vllm_server.log
```

**Out of memory errors**:
- Reduce `--gpu-memory-utilization` from 0.85 to 0.75
- Reduce `--max-model-len` from 32768 to 16384
- Reduce `+num_samples_in_parallel` from 16 to 8
- Ensure you have 4× A100 80GB GPUs (minimum recommended)

**NeMo Gym servers timeout**:
```bash
# Increase wait time in script (line 139)
# Check if vLLM is responding
curl http://localhost:10240/v1/models

# Check logs
tail -f data/qwen3_30b_eval/logs/nemo_gym_servers.log
```

**Slow rollout collection**:
- Model is large (30B parameters) - expect slower inference than 4B
- Verify vLLM is using GPU (not CPU fallback)
- Check GPU utilization: `nvidia-smi` should show high GPU usage
- Consider reducing `+num_samples_in_parallel` if experiencing memory pressure

**Ray GCS connection errors**:
- The script uses `USE_SHARED_RAY=false` by default
- Each process starts its own Ray workers
- If you see connection errors, they may be transient and should resolve automatically

### Clean Restart

If you need to restart everything:
```bash
# Stop all processes
pkill -f vllm.entrypoints
pkill -f ng_run

# Clear GPU cache (if needed)
# Note: vLLM server is kept running by default to save model loading time

# Wait and restart
sleep 5
./run_qwen3_30b_eval.sh
```

## Model Configuration

### Adjusting Model Path

Edit `run_qwen3_30b_eval.sh` line 12 to use your specific Qwen3 30B-A3B model:
```bash
MODEL_NAME="Qwen/Qwen3-30B-A3B"  # Or local path: "/path/to/model"
```

### GPU Configuration

**Multi-GPU (4× A100 80GB) - Recommended**:
```bash
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=32768
```

**Multi-GPU (2×)**:
```bash
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=32768
```

**Note**: Single GPU setup is not recommended for 30B model due to memory constraints.

### vLLM Server Reuse

The script automatically detects and reuses an existing vLLM server:

- **Enable reuse** (default): `REUSE_VLLM=true` or omit the variable
- **Disable reuse**: `REUSE_VLLM=false` to force a fresh server start
- **Keep server running** (default): `KEEP_VLLM=true` or omit the variable
- **Stop server on exit**: `KEEP_VLLM=false` to stop server when script exits

Example:
```bash
# Force fresh server start and stop on exit
REUSE_VLLM=false KEEP_VLLM=false ./run_qwen3_30b_eval.sh
```
