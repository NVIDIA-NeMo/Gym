# NewtonBench Resource Server

NeMo Gym environment for [NewtonBench](https://github.com/HKUST-KnowComp/NewtonBench), to train and test LLM agents to discover scientific laws through interactive experimentation. The benchmark includes **324 scientific law discovery tasks** across **12 physics domains** (Gravitation, Electrostatics, Magnetostatics, Thermal Conduction, Geometrical Optics, Nuclear Physics, Oscillations, Physical Optics, Acoustics, Elasticity, Statistical Mechanics, Calorimetry).

## Prerequisites

Clone the [NewtonBench](https://github.com/HKUST-KnowComp/NewtonBench) repository into the NeMo Gym repository root so that experiment modules can be loaded when running tools (e.g. `run_experiment_*`).

```bash
git clone https://github.com/HKUST-KnowComp/NewtonBench.git
```

Run the command above from the NeMo Gym repository root so that the `NewtonBench` directory exists at the root (e.g. `Gym/NewtonBench/`).

## Dataset Generation

Generate discovery tasks with varying physics domains, equation difficulties, system complexities, and noise levels:

**Generate full training dataset:**
```bash
python resources_servers/newton_bench/generate_dataset.py
```

**Generate training dataset by specific modules and equation difficulties:**
```bash
python resources_servers/newton_bench/generate_dataset.py \
    --modules m0_gravity,m1_coulomb_force \
    --difficulties easy,medium
```

**Generate training dataset by specific system complexities and noise levels:**
```bash
python resources_servers/newton_bench/generate_dataset.py \
    --systems vanilla_equation,complex_system \
    --noise-levels 0.0,0.01
```

**Generate training dataset with python code execution tool enabled:**
```bash
python resources_servers/newton_bench/generate_dataset.py --code-assisted
```

## Rollout Collection

### Configure env.yaml
Configure `env.yaml` to point to your vLLM server (or OpenAI-compatible endpoint).

**Example: local vLLM server for Qwen3-VL-8B-Thinking**

```yaml
policy_base_url: http://localhost:10240/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen3-VL-8B-Thinking
```

### Example: start a vLLM server for Qwen3-VL-8B-Thinking

Run the following from the NeMo Gym repo root in a separate virtual environment configured with `vllm`:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install hf_transfer datasets vllm --torch-backend=auto

HF_HOME=.cache/ \
HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download Qwen/Qwen3-VL-8B-Thinking

HF_HOME=.cache/ \
HOME=. \
vllm serve \
    Qwen/Qwen3-VL-8B-Thinking \
    --dtype auto \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --host 0.0.0.0 \
    --port 10240
```

> Note: Do **not** pass any `--reasoning-parser` flag to vLLM. NeMo Gym's `vllm_model` handles extraction of reasoning traces (e.g., `<think>...</think>`) itself.

### Launch servers
```bash
config_paths="resources_servers/newton_bench/configs/newton_bench.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

### Collect rollouts
```bash
ng_collect_rollouts \
    +agent_name=newton_bench_simple_agent \
    +input_jsonl_fpath=resources_servers/newton_bench/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/newton_bench/data/example_rollouts.jsonl \
    +limit=5
```

## Running Tests
```bash
source resources_servers/newton_bench/.venv/bin/activate 
pytest resources_servers/newton_bench/tests/test_app.py
```

## Qwen/Qwen3-VL-8B-Thinking Evaluation Summary

A total of 432 rollouts were collected using the **Qwen/Qwen3-VL-8B-Thinking** model as the evaluator on Newton Bench. The dataset consists of 108 prompts based on version v0 of scientific laws, spanning all levels of equation difficulty and model complexity, with each prompt repeated four times.

### Reward distribution

All statistics below are computed as follows:

- **Reward summary**
  - Mean reward: ≈ **0.0675**
  - Median reward: **0.0**
  - Min reward: ≈ **-0.8786**
  - Max reward: **1.0**

- **Reward buckets (width 0.2 from -1.0 to 1.0; counts of rollouts in each range)**:
  - [-1.0, -0.8): **16**
  - [-0.8, -0.6): **16**
  - [-0.6, -0.4): **60**
  - [-0.4, -0.2): **39**
  - [-0.2, 0.0): **24**
  - [0.0, 0.2): **150**
  - [0.2, 0.4): **46**
  - [0.4, 0.6): **2**
  - [0.6, 0.8): **1**
  - [0.8, 1.0]: **78**

### Overall tool-call usage

For each rollout, the number of tool calls is:

- `tool_calls_per_rollout = count(response.output[*].type == "function_call")`

Aggregate stats:

- Mean tool calls per rollout: ≈ **22.95**
- Median tool calls: **7.0**
- Min tool calls: **0**
- Max tool calls: **1770**

Most common exact tool-call counts (tool_calls → number of rollouts):

- **5** → 86 rollouts  
- **10** → 77 rollouts  
- **6** → 58 rollouts  
- **7** → 34 rollouts  
- **8** → 32 rollouts  
- **0** → 23 rollouts  
- **9** → 21 rollouts  
- **4** → 19 rollouts  
- **20** → 8 rollouts  
- **11–13,15** → 5–6 rollouts each (small samples)

### Mean reward by exact tool-call count

(Only showing counts with at least 5 rollouts for stability.)

| Tool Calls | Rollouts (n) | Mean Reward |
|-----------:|-------------:|------------:|
| 0         | 23           | ≈ **-0.1112** |
| 4         | 19           | ≈ **0.0624**  |
| 5         | 86           | ≈ **0.0465**  |
| 6         | 58           | ≈ **0.0664**  |
| 7         | 34           | ≈ **0.1020**  |
| 8         | 32           | ≈ **0.0079**  |
| 9         | 21           | ≈ **0.3957**  |
| 10        | 77           | ≈ **0.0864**  |
| 11        | 6            | ≈ **0.0869**  |
| 12        | 6            | ≈ **0.0971**  |
| 13        | 6            | ≈ **0.7187**  |
| 15        | 5            | ≈ **0.0448**  |
| 20        | 8            | ≈ **0.0125**  |

Tool-call counts not shown here (e.g., 1, 2, 3, 14, 16, 18, 19) either have fewer than 5 rollouts or zero rollouts (for 2 and 19), so they are omitted to avoid over-interpreting extremely small samples.

### Mean reward by tool-call count bins

Grouping tool-call counts into coarse bins:

| Tool Call Range | Rollouts (n) | Mean Reward |
|----------------:|-------------:|------------:|
| 0              | 23           | ≈ **-0.1112** |
| 1–10           | 329          | ≈ **0.0824**  |
| 11–50          | 60           | ≈ **0.1308**  |
| 51–200         | 15           | ≈ **-0.1959** |
| 201–2000       | 5            | ≈ **-0.0600** |

### Correlation and key observations

- **Correlation (tool_calls, reward)**: Pearson correlation ≈ **-0.0211**.
- **Key observations**:  
  - **Symbolic Accuracy:** Symbolic correctness is modest (~19.7% of rollouts have `symbolic_equivalent == true`), and RMSLE values show a wide spread, suggesting the model often misses both exact symbolic form and precise numeric behavior.
  - **Reward Distribution:** Rewards are centered near zero with a long tail (median 0.0, mean ~0.0675), and many rollouts are in negative ranges, indicating frequent partial or failed law discoveries.
  - **Tool Usage Sweet Spot:** Positive rewards are observed with moderate tool usage (approximately 1–50 calls), with performance peaking in the 11–50 call range. This suggests that a sufficiently large number of tool calls is necessary for the agent to gather comprehensive data and derive scientific laws.
  - **Diminishing Returns:** However, performance drops sharply after 50 calls. This indicates that once the agent has already collected enough data, additional tool calls no longer provide useful information and instead hinder performance. Therefore, simply increasing the number of tool calls is not a reliable path to higher rewards in this setup, highlighting that scientific discovery in this environment depends on reasoning and hypothesis selection, rather than sheer data volume.

## Licensing information
- **Code:** Apache 2.0
- **Data:** Apache 2.0
- **NewtonBench Benchmark:** MIT (Copyright (c) 2025 HKUST-KnowComp)

### Dependencies 
- **nemo_gym:** Apache 2.0
