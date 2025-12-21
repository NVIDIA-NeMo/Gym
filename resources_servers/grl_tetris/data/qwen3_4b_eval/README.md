# Qwen3-4B Evaluation for GRL Tetris

This directory contains the automated evaluation pipeline for running Qwen3-4B on the GRL Tetris environment, following CONTRIBUTING.md requirements.

## Overview

- **Model**: Qwen3-4B (via vLLM)
- **Test Set**: 200 diverse Tetris puzzles (`test_examples_200.jsonl`)
- **Rollouts per prompt**: 16
- **Total rollouts**: 3,200
- **Reward Analysis**: Automated profiling with statistics and visualizations

## Directory Structure

```
data/qwen3_4b_eval/
├── README.md                   # This file
├── test_examples_200.jsonl     # Input: 200 test prompts
├── rollouts.jsonl              # Output: Collected rollouts (created during run)
├── remaining_prompts.jsonl     # Checkpoint: Remaining prompts after crash (auto-generated)
├── reward_analysis.md          # Analysis: Reward statistics report (created during run)
└── logs/                       # Logs directory (created during run)
    ├── vllm_server.log         # vLLM server logs
    ├── nemo_gym_servers.log    # NeMo Gym server logs
    ├── rollout_collection.log  # Rollout collection logs
    ├── checkpoint.log          # Checkpoint creation logs
    └── reward_analysis.log     # Analysis logs
```

## Quick Start

### Option 1: Auto-Retry Loop (Recommended)

The loop script automatically handles crashes and resumes from checkpoints:

```bash
cd /workspace/Gym/resources_servers/grl_tetris
./run_qwen3_4b_eval_loop.sh
```

This will:
- Run the evaluation pipeline
- Automatically restart on Ray crashes or timeouts
- Resume from checkpoints if interrupted
- Continue until all 3,200 rollouts are collected (up to 20 attempts)

### Option 2: Single Run

For manual control:

```bash
cd /workspace/Gym/resources_servers/grl_tetris
./run_qwen3_4b_eval.sh
```

If it crashes, simply run the same command again—it will automatically detect partial progress and resume.

## Pipeline Stages

### 1. Environment Setup
- Cleans up any existing Ray/vLLM processes
- Optionally starts Ray cluster (for multi-GPU setups)

### 2. vLLM Server Launch
- Downloads and loads Qwen3-4B model (if not cached)
- Configures for tool calling with Hermes parser
- Waits for server to be ready (up to 10 minutes for first-time download)

### 3. NeMo Gym Server Launch
- Starts resource server (Tetris environment)
- Starts agent server (game-playing agent)
- Connects to vLLM for inference

### 4. Checkpoint & Resume Logic
- Detects existing `rollouts.jsonl` if present
- Calculates completed prompts (total_rollouts ÷ 16)
- Creates `remaining_prompts.jsonl` with only unprocessed prompts
- Truncates partial rollouts to ensure clean continuation
- Backs up existing rollouts before resuming

### 5. Rollout Collection
- Processes prompts sequentially (abc → aabbcc pattern with 16 repeats)
- Collects 16 rollouts per prompt
- In resume mode: collects new rollouts to temp file, then appends
- Saves to `rollouts.jsonl`

### 6. Reward Analysis
- Computes reward statistics (min, max, mean, median)
- Calculates success rate
- Analyzes tool call correlation with rewards
- Generates comprehensive report (`reward_analysis.md`)

### 7. Cleanup
- Gracefully stops vLLM server
- Gracefully stops NeMo Gym servers
- Stops Ray cluster (if multi-GPU)

## Configuration

Edit `run_qwen3_4b_eval.sh` to adjust:

```bash
MODEL_NAME="Qwen/Qwen3-4B"           # Model path
VLLM_PORT=10240                      # vLLM server port
TENSOR_PARALLEL_SIZE=1               # GPU parallelism (1 for single GPU)
GPU_MEMORY_UTILIZATION=0.85          # GPU memory fraction
MAX_MODEL_LEN=32768                  # Max context length
```

Rollout parameters (in step 5):
```bash
+num_repeats=16                      # Rollouts per prompt
+num_samples_in_parallel=16          # Parallel sampling
+responses_create_params.temperature=0.6
+responses_create_params.top_p=0.95
+responses_create_params.max_output_tokens=4096
```

## Checkpoint & Resume

The pipeline automatically handles interruptions:

1. **Detection**: Checks `rollouts.jsonl` on startup
2. **Calculation**: Determines completed prompts (e.g., 800 rollouts = 50 prompts done)
3. **Truncation**: Discards partial rollouts (e.g., 803 → 800)
4. **Resume File**: Creates `remaining_prompts.jsonl` with prompts 51-200
5. **Collection**: Collects new rollouts to temp file
6. **Append**: Safely appends new rollouts to existing file

### Manual Checkpoint Tools

Inspect progress:
```bash
# Count rollouts
wc -l rollouts.jsonl

# Calculate completed prompts (assuming 16 rollouts per prompt)
python -c "print($(wc -l < rollouts.jsonl) // 16)"

# Create checkpoint file manually
python ../../checkpoint_resume_rollouts.py \
    --input test_examples_200.jsonl \
    --rollouts rollouts.jsonl \
    --output remaining_prompts.jsonl \
    --target-repeats 16
```

## Analyzing Results

### View Reward Analysis Report
```bash
cat reward_analysis.md
```

### Launch Interactive Viewer
```bash
ng_viewer +jsonl_fpath=data/qwen3_4b_eval/rollouts.jsonl
```

### Manual Statistics
```bash
# Reward distribution
jq '.reward' rollouts.jsonl | sort -n | uniq -c

# Success rate
jq 'select(.success == true)' rollouts.jsonl | wc -l

# Average reward
jq -s 'map(.reward) | add / length' rollouts.jsonl
```

## Troubleshooting

### vLLM Won't Start
- Check logs: `tail -f logs/vllm_server.log`
- Verify GPU availability: `nvidia-smi`
- Reduce `GPU_MEMORY_UTILIZATION` if OOM errors occur
- Ensure port 10240 is free: `lsof -i :10240`

### NeMo Gym Won't Start
- Check logs: `tail -f logs/nemo_gym_servers.log`
- Verify vLLM is running: `curl http://localhost:10240/v1/models`
- Check for Python errors in log

### Ray Crashes During Collection
- This is expected with long runs
- The loop script will automatically restart
- Progress is saved—no rollouts are lost
- Check `logs/rollout_collection.log` for details

### Rollouts Not Resuming Correctly
- Verify `remaining_prompts.jsonl` was created
- Check `logs/checkpoint.log` for errors
- Manually inspect rollout count: `wc -l rollouts.jsonl`
- If needed, manually truncate to multiple of 16: `head -n 800 rollouts.jsonl > rollouts_fixed.jsonl`

### Analysis Fails
- Ensure `pandas` is installed: `pip install pandas`
- Check `logs/reward_analysis.log` for errors
- Verify rollout format: `head -n 1 rollouts.jsonl | jq .`

## Expected Timeline

- **First run** (cold start with model download): 15-30 minutes setup + collection time
- **Subsequent runs** (model cached): 5-10 minutes setup + collection time
- **Collection**: ~2-10 hours depending on GPU and model performance
  - Estimate: 3,200 rollouts × 10-20 seconds per rollout = 8-18 hours
  - Parallelism helps but is limited by GPU memory

## Output Files

### rollouts.jsonl Format
Each line is a JSON object:
```json
{
  "prompt": {
    "game_id": 1,
    "seed": 93810,
    "dim_board": [5, 5],
    "box_type": 0,
    "responses_create_params": {...}
  },
  "response": {
    "output": [...],  // Tool calls and text
    "finish_reason": "stop"
  },
  "reward": 9.7,
  "success": true,
  "num_steps": 8
}
```

### reward_analysis.md Format
Markdown report with:
- Dataset overview
- Reward distribution (min, max, mean, median)
- Success rate
- Reward histogram
- Tool call metrics
- Correlation analysis
- Per-prompt performance breakdown

## Contributing

This evaluation pipeline follows the requirements in `CONTRIBUTING.md`:
- ✅ 200 diverse test prompts
- ✅ 16 rollouts per prompt
- ✅ Automated reward profiling
- ✅ Checkpoint & resume support
- ✅ Comprehensive logging

## License

Apache 2.0 (see top-level LICENSE file)

## Questions?

- Pipeline issues: Check logs in `logs/` directory
- Tetris environment questions: See `resources_servers/grl_tetris/README.md`
- NeMo Gym setup: See top-level `README.md`

