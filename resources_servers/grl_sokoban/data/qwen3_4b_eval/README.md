# Qwen3 4B Evaluation for GRL Sokoban

This directory contains the evaluation setup and results for the Qwen3 4B model on the GRL Sokoban task, following CONTRIBUTING.md requirements.

## Evaluation Configuration

### Dataset
- **Input**: 200 diverse Sokoban puzzles from `test_examples_200.jsonl`
- **Source**: First 200 examples from `test_examples.jsonl` (500 total)
- **Coverage**: Various room sizes [4×4 to 8×8], different box counts (1-3 boxes), randomized seeds

### Inference Setup
- **Model**: Qwen3-4B 
- **Engine**: vLLM for efficient inference
- **Rollouts per prompt**: 16
- **Total rollouts**: 3,200 (200 prompts × 16 rollouts)
- **Temperature**: 0.8
- **Max output tokens**: 2,000
- **Parallel samples**: 16

### CONTRIBUTING.md Compliance

This evaluation follows the requirements from CONTRIBUTING.md Section 4 (Reward Profiling):
- ✅ ~500 small subset (using 200 prompts)
- ✅ 16 responses per prompt
- ✅ Reward distribution reporting
- ✅ Tool call metrics and correlation with rewards

## How The Scripts Work

Two automated scripts are available for running evaluations:

### `run_qwen3_4b_eval.sh` (Single Run)

The main evaluation script orchestrates the entire evaluation pipeline:

1. **Cleanup**: Stops any existing vLLM/Ray processes
2. **Ray Cluster** (multi-GPU only): Starts shared Ray cluster
3. **vLLM Server**: Launches model server with Qwen3-4B
4. **NeMo Gym Servers**: Starts environment and agent servers
5. **Checkpoint Check**: Detects partial rollouts and creates resume checkpoint
6. **Rollout Collection**: Generates 3,200 rollouts (200 prompts × 16 repeats)
7. **Reward Analysis**: Generates comprehensive evaluation report
8. **Cleanup**: Terminates all background processes on exit

### `run_qwen3_4b_eval_loop.sh` (Auto-Retry Loop)

A wrapper script that provides automatic crash recovery:

1. **Progress Check**: Counts existing rollouts to determine completion status
2. **Run Main Script**: Executes `run_qwen3_4b_eval.sh`
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
./run_qwen3_4b_eval.sh
# [Script stops due to error]

# Second run - automatically resumes
./run_qwen3_4b_eval.sh
# [INFO] Found 1500 existing rollouts (expected 3200)
# [INFO] Creating checkpoint to resume from remaining prompts...
# [INFO] Resuming collection with 106 remaining prompts
# [continues from where it left off]
```

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
wc -l data/qwen3_4b_eval/test_examples_200.jsonl
# Should output: 200
```

### Running the Evaluation

**Option 1: Auto-Retry Loop (Recommended)**:

For maximum reliability with automatic crash recovery:
```bash
cd resources_servers/grl_sokoban
./run_qwen3_4b_eval_loop.sh
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
./run_qwen3_4b_eval.sh
```

The script automatically:
1. Cleans up any existing processes
2. Starts Ray cluster (if multi-GPU)
3. Launches vLLM server with Qwen3 4B
4. Starts NeMo Gym servers
5. Collects 3,200 rollouts (200 prompts × 16 rollouts)
6. Analyzes reward distribution
7. Generates comprehensive report

**Expected Runtime**:
- Single GPU: ~2-4 hours (depending on GPU and model loading time)
- Multi-GPU (4×): ~1-2 hours

**When to Use Each Option**:
- Use `run_qwen3_4b_eval_loop.sh` if you expect Ray crashes or want unattended execution
- Use `run_qwen3_4b_eval.sh` for single runs or debugging
- Both scripts support automatic checkpoint/resume if interrupted

### Manual Step-by-Step Execution

If you prefer manual control, follow these steps:

#### 1. Start vLLM Server

**Single GPU**:
```bash
HF_HOME=.cache/ vllm serve Qwen/Qwen3-4B \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240 \
    --max-model-len 8192 \
    --trust-remote-code
```

**Multi-GPU (4×)**:
```bash
# Start Ray cluster first
ray stop --force
ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats
sleep 3

# Start vLLM with tensor parallelism
HF_HOME=.cache/ vllm serve Qwen/Qwen3-4B \
    --dtype auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240 \
    --max-model-len 8192 \
    --trust-remote-code
```

Wait 2-5 minutes, then verify:
```bash
curl http://localhost:10240/v1/models
```

#### 2. Start NeMo Gym Servers

In a new terminal:
```bash
export policy_base_url="http://localhost:10240/v1"
export policy_api_key="dummy"
export policy_model_name="Qwen/Qwen3-4B"

cd Gym

# Single GPU
ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_sokoban/configs/grl_sokoban.yaml]"

# Multi-GPU (with shared Ray cluster)
ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_sokoban/configs/grl_sokoban.yaml]" \
    "+ray_head_node_address=127.0.0.1:6379"
```

Wait for: `All 3 / 3 servers ready!`

#### 3. Collect Rollouts

In another terminal:
```bash
ng_collect_rollouts \
    +agent_name=grl_sokoban_game_agent \
    +input_jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_4b_eval/test_examples_200.jsonl \
    +output_jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_4b_eval/rollouts.jsonl \
    +limit=null \
    +num_repeats=16 \
    +num_samples_in_parallel=32 \
    +responses_create_params.temperature=0.8 \
    +responses_create_params.max_output_tokens=4096
```

#### 4. Analyze Results

```bash
cd resources_servers/grl_sokoban

python analyze_rewards.py \
    --rollouts-path data/qwen3_4b_eval/rollouts.jsonl \
    --model-name "Qwen3-4B" \
    --output data/qwen3_4b_eval/reward_analysis.md

# View the report
cat data/qwen3_4b_eval/reward_analysis.md
```

## Output Files

After running the evaluation, this directory will contain:

```
qwen3_4b_eval/
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
ng_viewer +jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_4b_eval/rollouts.jsonl
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
}' data/qwen3_4b_eval/rollouts.jsonl
```

**Success rate**:
```bash
jq -s 'map(select(.success == true)) | length' data/qwen3_4b_eval/rollouts.jsonl
```

**Reward distribution**:
```bash
jq '.reward' data/qwen3_4b_eval/rollouts.jsonl | sort -n | uniq -c | sort -rn
```

**Tool call metrics**:
```bash
jq -s 'map([.output[] | select(.type == "function_call")] | length) | {
    avg: (add / length),
    min: min,
    max: max
}' data/qwen3_4b_eval/rollouts.jsonl
```

## Actual Results

Results from running Qwen3-4B on 3,200 rollouts (200 prompts × 16 rollouts):

### Overall Metrics
- **Total Rollouts**: 3,200
- **Success Rate**: 13.47% (431 / 3,200)
- **Mean Reward**: 0.9305
- **Median Reward**: 0.0000
- **Min Reward**: -8.9000
- **Max Reward**: 10.9000

### Tool Call Statistics
- **Average Tool Calls**: 2.64 per rollout
- **Min Tool Calls**: 1
- **Max Tool Calls**: 11
- **Correlation (tool calls ↔ reward)**: -0.2338 (negative correlation)

### Reward Distribution
- **0.0 reward**: 2,134 occurrences (66.7%) - immediate failures
- **10.8 reward**: 206 occurrences (6.4%)
- **10.9 reward**: 72 occurrences (2.2%)
- **10.7 reward**: 51 occurrences (1.6%)
- **Negative rewards**: ~800 occurrences (25%) - invalid moves/failures

### Performance by Tool Call Count
| Tool Calls | Mean Reward | Rollout Count | Notes |
|------------|-------------|---------------|-------|
| 1          | 0.0000      | 2,112         | Immediate failures (66%) |
| 2          | 7.0948      | 174           | Quick successes |
| 3          | 8.0076      | 314           | Best average performance |
| 4          | 4.9391      | 87            | Moderate attempts |
| 5          | 3.0453      | 53            | Declining performance |
| 10         | -3.5120     | 409           | Getting stuck in loops |

### Key Observations
1. **High Early Failure Rate**: 66.7% of rollouts fail immediately with only 1 tool call, suggesting the model often doesn't properly engage with the task
2. **Negative Correlation**: More tool calls correlate with worse outcomes (-0.2338), indicating the model gets stuck in invalid move patterns
3. **Sweet Spot**: Rollouts with 2-3 tool calls perform best (mean rewards ~7-8), suggesting successful puzzles are solved quickly
4. **Success Pattern**: When successful, the model typically completes puzzles in 2-3 moves, but this only happens in ~15% of cases

## Manual Checkpoint/Resume

If you want to manually create a checkpoint (without running the full script):

```bash
cd resources_servers/grl_sokoban

# Check what's left to do
python checkpoint_resume_rollouts.py \
    --input data/qwen3_4b_eval/test_examples_200.jsonl \
    --rollouts data/qwen3_4b_eval/rollouts.jsonl \
    --output data/qwen3_4b_eval/remaining_prompts.jsonl \
    --target-repeats 16

# Then collect only remaining rollouts
ng_collect_rollouts \
    +agent_name=grl_sokoban_game_agent \
    +input_jsonl_fpath=data/qwen3_4b_eval/remaining_prompts.jsonl \
    +output_jsonl_fpath=data/qwen3_4b_eval/rollouts.jsonl \
    +limit=null \
    +num_repeats=16 \
    +num_samples_in_parallel=32 \
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
tail -f data/qwen3_4b_eval/logs/vllm_server.log
```

**NeMo Gym servers timeout**:
```bash
# Increase wait time in script (line 139)
# Check if vLLM is responding
curl http://localhost:10240/v1/models

# Check logs
tail -f data/qwen3_4b_eval/logs/nemo_gym_servers.log
```

**Out of memory errors**:
- Reduce `--gpu-memory-utilization` from 0.85 to 0.7
- Reduce `--max-model-len` from 8192 to 4096
- Reduce `+num_samples_in_parallel` from 32 to 16

**Slow rollout collection**:
- Increase `+num_samples_in_parallel` (if memory allows)
- Use multi-GPU setup with tensor parallelism
- Verify vLLM is using GPU (not CPU fallback)

### Clean Restart

If you need to restart everything:
```bash
# Stop all processes
pkill -f vllm.entrypoints
ray stop --force
pkill -f ng_run

# Clear GPU cache
python clear_gpu_cache.py  # From Gym root

# Wait and restart
sleep 5
./run_qwen3_4b_eval.sh
```

## Model Configuration

### Adjusting Model Path

Edit `run_qwen3_4b_eval.sh` line 11 to use your specific Qwen3 4B model:
```bash
MODEL_NAME="Qwen/Qwen3-4B"  # Or local path: "/path/to/model"
```

### GPU Configuration

**Single GPU** (default):
```bash
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.85
```

**Multi-GPU (2×)**:
```bash
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9
```

**Multi-GPU (4×)**:
```bash
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
```

## Contributing Back

When contributing these results to the main repository:

1. **Include in PR**:
   - `reward_analysis.md` (reward distribution report)
   - Sample rollouts (5 examples showing diverse outcomes)
   - Tool call metrics and correlation analysis

2. **Document**:
   - Model configuration used
   - GPU setup and parallelism
   - Any modifications to default parameters
   - Observed success rates and patterns

3. **Verification**:
   - Verify rollout count: `wc -l rollouts.jsonl` (should be 3200)
   - Check for errors: `grep -i error logs/*.log`
   - Review sample outputs for correctness

## References

- [NeMo Gym Contributing Guide](../../../CONTRIBUTING.md)
- [GRL Sokoban README](../../README.md)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Test Examples Generation](../../generate_test_examples.py)

