# GRL Sokoban Resource Server

Single-box Sokoban puzzle environment served via FastAPI with NeMo Gym conventions. The environment is implemented locally under `resources_servers/grl_sokoban/env`, mirroring GRL’s behaviour without requiring the external repository.

## Why it exists
- **Domain**: Deterministic Sokoban puzzles.
- **Evaluation**: Agents must push a box onto its target with minimal invalid moves.
- **Verifier**: `/verify` rewards the cumulative Sokoban score only when `success` is reported by the environment.

## Setup
1. **Install NeMo Gym locally (one-time)**
   ```bash
   uv pip install -e ".[dev]"
   ```
   This makes the `ng_*` CLI available in your active environment.
2. **Install Sokoban-specific dependencies**
   ```bash
   uv pip install -r resources_servers/grl_sokoban/requirements.txt
   ```
3. (Optional) prepare datasets using `ng_collect_rollouts` once custom rollouts are available.

## Running
Spin up the server alongside a compatible agent:
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/grl_sokoban/configs/grl_sokoban.yaml"
ng_run "+config_paths=[$config_paths]"
```

Collect trajectories:
```bash
ng_collect_rollouts +agent_name=grl_sokoban_game_agent \
    +input_jsonl_fpath=resources_servers/grl_sokoban/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/grl_sokoban/data/example_rollouts.jsonl \
    +limit=5
```

# Launch the rollout viewer
```bash
ng_viewer +jsonl_fpath=resources_servers/grl_sokoban/data/example_rollouts.jsonl
```

## Generating Test Examples for Reward Profiling

For CONTRIBUTING.md reward profiling requirements, generate ~500 diverse test examples with varying seeds and room dimensions:

```bash
cd resources_servers/grl_sokoban
python generate_test_examples.py --num-examples 500
```

This creates `data/test_examples.jsonl` with diverse configurations:
- **Room sizes**: [4,4] to [8,8] with various aspect ratios
- **Num boxes**: Weighted distribution (62% 1-box, 25% 2-box, 13% 3-box)
- **Seeds**: Randomized for unique, solvable puzzles

Use the generated test set for reward profiling (see next section).

## Running with vLLM for Reward Profiling

For reward profiling and RL training (as per CONTRIBUTING.md), use vLLM with local models like Qwen3-30B-A3B.

**Choose your setup:**
- **Single GPU?** → Follow the "Quick Start (Single GPU)" section below
- **Multi-GPU (2+ GPUs)?** → Follow the "Multi-GPU Setup" section below

---

## Quick Start (Single GPU)

### 1. Start vLLM Server

**Prerequisites:**
```bash
uv pip install vllm hf_transfer
```

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

**Wait 2-5 minutes for model loading.** Verify server is ready:
```bash
curl http://localhost:10240/v1/models
```

### 2. Start NeMo Gym Servers

In a new terminal:
```bash
# Set environment variables
export policy_base_url="http://localhost:10240/v1"
export policy_api_key="dummy"
export policy_model_name="Qwen/Qwen3-30B-A3B"

# Start servers (no Ray cluster needed for single GPU)
ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_sokoban/configs/grl_sokoban.yaml]"
```

**Wait until you see:** `All 3 / 3 servers ready!` before proceeding.

### 3. Collect Rollouts

**In a new terminal** (keep servers running):
```bash
ng_collect_rollouts +agent_name=grl_sokoban_game_agent \
    +input_jsonl_fpath=resources_servers/grl_sokoban/data/test_examples.jsonl \
    +output_jsonl_fpath=resources_servers/grl_sokoban/data/test_rollouts.jsonl \
    +limit=null \
    +num_repeats=4 \
    +num_samples_in_parallel=32 \
    +responses_create_params.temperature=0.8 \
    +responses_create_params.max_output_tokens=3000
```

---

## Multi-GPU Setup (4+ GPUs)

### 1. Start vLLM Server with Multi-GPU
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

### 2. Start Shared Ray Cluster

**Important for multi-GPU setups:** To avoid slow startup and port conflicts, start a shared Ray cluster first:

```bash
# Clean up any existing Ray sessions
ray stop --force

# Start a shared Ray cluster
ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats

# Wait a few seconds for cluster to be ready
sleep 3
```

### 3. Start NeMo Gym Servers

In a new terminal (or the same terminal after Ray starts):
```bash
# Set environment variables
export policy_base_url="http://localhost:10240/v1"
export policy_api_key="dummy"
export policy_model_name="Qwen/Qwen3-30B-A3B"

# Start servers with shared Ray cluster
ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_sokoban/configs/grl_sokoban.yaml]" \
"+ray_head_node_address=127.0.0.1:6379"
```

**Wait until you see:** `All 3 / 3 servers ready!` before proceeding.

### 4. Collect Rollouts

**In a new terminal** (keep servers running):

**Using the test examples dataset (500 diverse puzzles, with high parallelism):**
```bash
ng_collect_rollouts +agent_name=grl_sokoban_game_agent \
    +input_jsonl_fpath=resources_servers/grl_sokoban/data/test_examples.jsonl \
    +output_jsonl_fpath=resources_servers/grl_sokoban/data/test_rollouts.jsonl \
    +limit=null \
    +num_repeats=1 \
    +num_samples_in_parallel=128 \
    +responses_create_params.temperature=0.8 \
    +responses_create_params.max_output_tokens=3000
```

---

## Analyze Reward Distribution (Both Setups)

### Automated Analysis (Recommended)

**Generate comprehensive reward profiling report** (required for CONTRIBUTING.md):

```bash
cd resources_servers/grl_sokoban

# Install pandas if not already installed
pip install pandas

# Generate report for Qwen3-30B-A3B
python analyze_rewards.py \
    --rollouts-path data/test_rollouts.jsonl \
    --model-name "Qwen3-30B-A3B" \
    --output data/reward_analysis_qwen3_30b.md

# View the report
cat data/reward_analysis_qwen3_30b.md
```

This generates a complete report including:
- Reward distribution statistics (min, max, mean, median)
- Success rate analysis
- Reward histogram
- Tool call metrics and correlation with rewards
- Per-prompt performance breakdown
- Top/bottom performing prompts

**For Qwen3-235B-Instruct** (second required model):
```bash
# After collecting rollouts with 235B model, run:
python analyze_rewards.py \
    --rollouts-path data/test_rollouts_qwen3_235b.jsonl \
    --model-name "Qwen3-235B-Instruct" \
    --output data/reward_analysis_qwen3_235b.md
```

### Results Summary (Qwen3-4B)

**Dataset**: 3,200 rollouts (200 prompts × 16 repeats)

**Performance Metrics**:
- **Success Rate**: 13.47% (431/3,200 rollouts)
- **Mean Reward**: 0.93 (range: -8.90 to 10.90)
- **Median Reward**: 0.00

**Key Findings**:
- Most rollouts (66.7%) received reward of 0.00 (no valid actions taken)
- Successful puzzle solutions achieved rewards of ~10.5-10.9
- Average 2.64 tool calls per rollout
- Moderate negative correlation between tool calls and reward (-0.23)

**Top Reward Distribution**:
- `0.0`: 2,134 rollouts (66.7%) - no valid actions or early termination
- `10.8`: 206 rollouts (6.4%) - successful puzzle completion
- `10.9`: 72 rollouts (2.2%) - successful puzzle completion
- `10.7`: 51 rollouts (1.6%) - successful puzzle completion
- Negative rewards: Invalid moves or non-optimal solutions

The moderate success rate (13.47%) indicates that Sokoban puzzle-solving requires spatial planning and understanding of box-pushing mechanics. Most failures result from the model not taking valid actions (reward 0.0), while successful completions achieve consistent high rewards (~10.5-10.9). The negative correlation between tool calls and reward suggests that longer sequences often lead to invalid moves or dead-end states.

See [`data/qwen3_4b_eval/reward-analysis.md`](data/qwen3_4b_eval/reward-analysis.md) for complete analysis.

### Interactive Viewer

**Visual exploration of rollouts:**
```bash
ng_viewer +jsonl_fpath=resources_servers/grl_sokoban/data/test_rollouts.jsonl
```

### Manual Command-Line Analysis

**Quick stats** (if you prefer manual analysis):
```bash
# Reward distribution
jq '.reward' resources_servers/grl_sokoban/data/test_rollouts.jsonl | sort -n | uniq -c

# Statistics (min, max, avg)
jq -s 'map(.reward) | {
    min: min,
    max: max,
    avg: (add / length),
    count: length
}' resources_servers/grl_sokoban/data/test_rollouts.jsonl

# Success rate
jq -s 'map(select(.success == true)) | length' \
    resources_servers/grl_sokoban/data/test_rollouts.jsonl

# Tool call metrics (average per rollout)
jq -s 'map([.output[] | select(.type == "function_call")] | length) | add / length' \
    resources_servers/grl_sokoban/data/test_rollouts.jsonl
```

### Other Recommended Models

**For math/coding tasks:** `Qwen/Qwen3-235B-Thinking`  
**For agents/instruction following:** `Qwen/Qwen3-235B-Instruct`

Adjust `--tensor-parallel-size` based on available GPUs (235B models typically need 8 GPUs).

## Dataset artifacts
Placeholder files live under `data/` (`example.jsonl`, `example_metrics.json`, `example_rollouts.jsonl`). Replace them with generated rollouts and metrics when integrating into training pipelines.

## Tests
```bash
pytest resources_servers/grl_sokoban/tests
```

## Licensing
- Code: Apache 2.0
- Data: Apache 2.0
