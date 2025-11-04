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

For reward profiling and RL training (as per CONTRIBUTING.md), use vLLM with local models. We support both **Qwen3-4B** and **Qwen3-30B-A3B** models.

**Choose your setup:**
- **Using bash scripts (recommended)?** → Follow the "Automated Bash Scripts" section below
- **Manual step-by-step?** → Follow the "Manual Setup" section below

---

## Automated Bash Scripts

### Prerequisites

```bash
uv pip install vllm hf_transfer
```

### Qwen3-4B Model

**For single GPU or multi-GPU evaluation (recommended: auto-retry loop):**

```bash
cd resources_servers/grl_sokoban

# Edit the script to configure:
# - TENSOR_PARALLEL_SIZE (1 for single GPU, 2+ for multi-GPU)
# Then run:
./run_qwen3_4b_eval_loop.sh
```

This auto-retry loop script:
- Collects 3,200 rollouts (200 prompts × 16 repeats)
- Supports automatic checkpoint/resume
- Automatically retries if Ray crashes or timeouts occur
- Continues until all rollouts are collected
- Stops after 20 attempts or successful completion
- Generates reward analysis report
- Works with both single GPU and multi-GPU setups

**Alternative (single run):**
```bash
./run_qwen3_4b_eval.sh
```

See [`data/qwen3_4b_eval/README.md`](data/qwen3_4b_eval/README.md) for detailed documentation.

### Qwen3-30B-A3B Model

**For single GPU or multi-GPU evaluation (recommended: auto-retry loop):**

```bash
cd resources_servers/grl_sokoban

# First, create the 200 prompts file (if not exists):
# head -n 200 data/test_examples.jsonl > data/qwen3_30b_eval/test_examples_200.jsonl

# Edit the script to configure:
# - TENSOR_PARALLEL_SIZE (1 for single GPU, 4 for multi-GPU)
# Then run:
./run_qwen3_30b_eval_loop.sh
```

This auto-retry loop script:
- Collects 3,200 rollouts (200 prompts × 16 repeats) - matching 4B setup
- Supports automatic checkpoint/resume
- Automatically retries if Ray crashes or timeouts occur
- Continues until all rollouts are collected
- Stops after 20 attempts or successful completion
- Generates reward analysis report
- Works with both single GPU and multi-GPU setups

**Alternative (single run):**
```bash
./run_qwen3_30b_eval.sh
```

The scripts automatically handle cleanup, model loading, server startup, and rollout collection.

---

## Manual Setup

### Qwen3-4B Model

#### Single GPU Setup

**1. Start vLLM Server:**
```bash
cd resources_servers/grl_sokoban
HF_HOME=.cache/ \
vllm serve Qwen/Qwen3-4B \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240 \
    --max-model-len 32768 \
    --trust-remote-code
```

**2. Start NeMo Gym Servers (in a new terminal):**
```bash
export policy_base_url="http://localhost:10240/v1"
export policy_api_key="dummy"
export policy_model_name="Qwen/Qwen3-4B"

ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_sokoban/configs/grl_sokoban.yaml]"
```

**3. Collect Rollouts (in a new terminal):**
```bash
ng_collect_rollouts +agent_name=grl_sokoban_game_agent \
    +input_jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_4b_eval/test_examples_200.jsonl \
    +output_jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_4b_eval/rollouts.jsonl \
    +limit=null \
    +num_repeats=16 \
    +num_samples_in_parallel=16 \
    +responses_create_params.temperature=0.6 \
    +responses_create_params.max_output_tokens=4096
```

### Qwen3-30B-A3B Model


#### Multi-GPU Setup (4+ GPUs)

**1. Start vLLM Server:**
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

**2. Start NeMo Gym Servers (in a new terminal):**
```bash
export policy_base_url="http://localhost:10240/v1"
export policy_api_key="dummy"
export policy_model_name="Qwen/Qwen3-30B-A3B"

ng_run "+config_paths=[responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/grl_sokoban/configs/grl_sokoban.yaml]"
```

**3. Collect Rollouts (in a new terminal):**
```bash
ng_collect_rollouts +agent_name=grl_sokoban_game_agent \
    +input_jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_30b_eval/test_examples_200.jsonl \
    +output_jsonl_fpath=resources_servers/grl_sokoban/data/qwen3_30b_eval/rollouts.jsonl \
    +limit=null \
    +num_repeats=16 \
    +num_samples_in_parallel=16 \
    +responses_create_params.temperature=0.8 \
    +responses_create_params.max_output_tokens=4096
```

---

## Analyze Reward Distribution (Both Setups)

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

### Results Summary (Qwen3-30B-A3B)

**Dataset**: 3,200 rollouts (200 prompts × 16 repeats)

**Performance Metrics**:
- **Success Rate**: 38.56% (1,234/3,200 rollouts)
- **Mean Reward**: 4.00 (range: -5.40 to 10.90)
- **Median Reward**: 0.00

**Key Findings**:
- Most rollouts (43.9%) received reward of 0.00 (no valid actions taken)
- Successful puzzle solutions achieved rewards of ~10.5-10.9
- Average 2.10 tool calls per rollout
- Moderate positive correlation between tool calls and reward (0.22)

**Top Reward Distribution**:
- `0.0`: 1,405 rollouts (43.9%) - no valid actions or early termination
- `10.8`: 477 rollouts (14.9%) - successful puzzle completion
- `10.6`: 183 rollouts (5.7%) - successful puzzle completion
- `10.7`: 172 rollouts (5.4%) - successful puzzle completion
- `10.9`: 157 rollouts (4.9%) - successful puzzle completion
- Negative rewards: Invalid moves or non-optimal solutions

The higher success rate (38.56%) compared to Qwen3-4B indicates that the larger model performs significantly better at spatial planning and understanding box-pushing mechanics. While the majority of failures still result from not taking valid actions (reward 0.0), the model achieves nearly 3x the success rate of the 4B variant. The positive correlation between tool calls and reward suggests that the model can effectively use longer action sequences to solve puzzles.

See [`data/qwen3_30b_eval/reward-analysis.md`](data/qwen3_30b_eval/reward-analysis.md) for complete analysis.

### Interactive Viewer

**Visual exploration of rollouts:**
```bash
ng_viewer +jsonl_fpath=resources_servers/grl_sokoban/data/test_rollouts.jsonl
```

## Tests
```bash
pytest resources_servers/grl_sokoban/tests
```

## Licensing
- Code: Apache 2.0
- Data: Apache 2.0
