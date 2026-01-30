(environment-reasoning-gym)=

# Reasoning Gym

This tutorial guides you through integrating with [open-thought/reasoning-gym](https://github.com/open-thought/reasoning-gym), a library of procedural dataset generators and algorithmically verifiable reasoning environments.

:::{card}

**Goal**: Run Reasoning Gym environments in NeMo Gym for training and inference.

^^^

**In this tutorial, you will**:

1. Create datasets from reasoning tasks
2. Configure and launch servers
3. Collect rollouts with procedurally generated tasks

:::

## What is Reasoning Gym?

Reasoning Gym provides 100+ tasks across 12 categories including logic, arithmetic, algebra, algorithmic puzzles, games, and more. Tasks are procedurally generated with adjustable complexity and algorithmically verified.

:::{tip}
**Key terminology**:
- **Resources server**: A NeMo Gym component that manages environment instances and provides the `/seed_session`, `/step`, and `/verify` endpoints.
- **Rollout**: A complete episode of agent-environment interaction, from initial prompt to final answer.
- **Procedural generation**: Tasks are created algorithmically, allowing unlimited unique examples with controllable difficulty.
- **Algorithmic verification**: Answers are verified programmatically without requiring LLM-based judges.

Unlike environments that use static datasets, Reasoning Gym can generate fresh examples on demand, which helps prevent memorization during training.
:::

---

## Before You Start

Ensure you have:

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager installed
- A CUDA-capable GPU (minimum 24 GB VRAM for the 30B model, or 8 GB for smaller models)
- The NeMo Gym repository cloned locally

### Install Dependencies

Install the Reasoning Gym package from the NeMo Gym repository root:

```bash
uv venv
source .venv/bin/activate
uv sync
uv add "reasoning-gym>=0.4.0"
```

### Available Task Categories

The integration supports all Reasoning Gym task categories:

| Category | Examples | Description |
|----------|----------|-------------|
| **logic** | knights_knaves, syllogism, propositional_logic | Logical reasoning puzzles |
| **arithmetic** | basic_arithmetic, chain_sum, leg_counting | Numerical computation |
| **algebra** | simple_equations, polynomial_equations | Algebraic problem solving |
| **algorithmic** | word_sorting, base_conversion, caesar_cipher | String and array manipulation |
| **cognition** | figlet_font, needle_haystack, rubiks_cube | Pattern recognition and spatial reasoning |
| **games** | sudoku, n_queens, tower_of_hanoi, sokoban | Strategy and puzzle games |
| **graphs** | shortest_path, course_schedule, family_relationships | Graph theory problems |
| **geometry** | simple_geometry, advanced_geometry | Geometric calculations |
| **arc** | arc_1d, arc_agi, rearc | ARC-style reasoning tasks |
| **code** | bf, codeio | Code execution and generation |
| **induction** | acre, list_functions | Inductive reasoning |
| **probability** | coin_flip | Probability calculations |

For a complete list of tasks, refer to the [Reasoning Gym Dataset Gallery](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md).

---

## Tutorial Steps

1. **Create a Dataset**

   The integration includes a helper script for creating datasets from reasoning gym tasks. Run commands from the repository root.

   :::::{tab-set}

   ::::{tab-item} Single Task

   ```bash
   python resources_servers/reasoning_gym/scripts/create_dataset.py \
       --task knights_knaves \
       --size 500 \
       --seed 42 \
       --output resources_servers/reasoning_gym/data/train_knights_knaves.jsonl
   ```

   ::::

   ::::{tab-item} Multiple Tasks

   ```bash
   python resources_servers/reasoning_gym/scripts/create_dataset.py \
       --tasks knights_knaves,syllogism,leg_counting \
       --size 1000 \
       --output resources_servers/reasoning_gym/data/train_composite.jsonl
   ```

   ::::

   ::::{tab-item} By Category

   ```bash
   python resources_servers/reasoning_gym/scripts/create_dataset.py \
       --category logic \
       --size 1000 \
       --output resources_servers/reasoning_gym/data/train_logic.jsonl
   ```

   ::::

   ::::{tab-item} All Tasks

   ```bash
   python resources_servers/reasoning_gym/scripts/create_dataset.py \
       --all-tasks \
       --size 1000 \
       --output resources_servers/reasoning_gym/data/train_all.jsonl
   ```

   ::::

   ::::{tab-item} Custom Config

   ```bash
   python resources_servers/reasoning_gym/scripts/create_dataset.py \
       --task knights_knaves \
       --size 500 \
       --config '{"n_people": 3, "depth_constraint": 3}' \
       --output resources_servers/reasoning_gym/data/train_hard.jsonl
   ```

   ::::

   :::::

   **Expected output:**

   ```text
   Creating knights_knaves dataset with 500 samples (seed=42)...
   Generated 500 entries for knights_knaves

   Dataset saved to resources_servers/reasoning_gym/data/train_knights_knaves.jsonl
   Total entries: 500
   ```

   :::{dropdown} Script Options Reference
   :icon: info

   | Option | Description |
   |--------|-------------|
   | `--task` | Single task name |
   | `--tasks` | Comma-separated task names |
   | `--category` | Task category (logic, arithmetic, algebra, etc.) |
   | `--all-tasks` | Include all available tasks |
   | `--size` | Number of examples to generate (default: 500) |
   | `--seed` | Random seed for reproducibility (default: 42) |
   | `--config` | JSON string with task-specific parameters |
   | `--output` | Output JSONL file path (required) |

   :::

2. **Configure Model Server**

   Create `env.yaml` at the repository root:

   ```yaml
   policy_base_url: "http://localhost:10240/v1"
   policy_api_key: "EMPTY"
   policy_model_name: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
   ```

   :::{note}
   For production deployments, use environment variables instead of hardcoding credentials. The `policy_api_key` field accepts a placeholder when authentication is not required.
   :::

3. **Start Model Server**

   Start a vLLM server with reasoning support:

   ```bash
   pip install -U "vllm>=0.12.0"

   wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py

   vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
     --max-num-seqs 8 \
     --tensor-parallel-size 1 \
     --max-model-len 262144 \
     --port 10240 \
     --trust-remote-code \
     --tool-call-parser qwen3_coder \
     --reasoning-parser-plugin nano_v3_reasoning_parser.py \
     --reasoning-parser nano_v3
   ```

   :::{tip}
   The reasoning parser plugin enables the model to output structured reasoning traces. Different models may require different configurations. For supported configurations, refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).
   :::

4. **Launch NeMo Gym Servers**

   Start the Reasoning Gym resources server:

   ```bash
   ng_run "+config_paths=[resources_servers/reasoning_gym/configs/reasoning_gym.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
   ```

   **Expected output** (ports may vary based on configuration):

   ```text
   INFO:     Starting servers...
   INFO:     reasoning_gym_resources_server running on http://0.0.0.0:8001
   INFO:     reasoning_gym_simple_agent running on http://0.0.0.0:8002
   INFO:     Press Ctrl+C to stop
   ```

5. **Collect Rollouts**

   In another terminal:

   ```bash
   ng_collect_rollouts \
       +agent_name=reasoning_gym_simple_agent \
       +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
       +output_jsonl_fpath=results/reasoning_gym_rollouts.jsonl \
       +limit=5
   ```

   | Option | Description |
   |--------|-------------|
   | `+limit=N` | Process first N examples only |
   | `+num_repeats=K` | Run each example K times for variance |
   | `+num_samples_in_parallel=P` | Limit concurrent requests to P |

---

## Reference

- [Reasoning Gym GitHub](https://github.com/open-thought/reasoning-gym) - Official Reasoning Gym repository
- [Dataset Gallery](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md) - Examples of all available tasks
- `resources_servers/reasoning_gym/` - NeMo Gym resources server implementation

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Multi-Step Environments
:link: /environment-tutorials/multi-step
:link-type: doc
Build sequential tool-calling workflows.
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-a-Judge
:link: /environment-tutorials/llm-as-judge
:link-type: doc
Use LLMs for flexible verification.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Start training models on your environment.
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Verifiers Integration
:link: verifiers
:link-type: doc
Explore 600+ environments from Prime Intellect.
:::

::::
