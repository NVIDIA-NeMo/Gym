---
description: "Complete guide to getting started with NeMo Gym for building RL training environments for LLMs"
personas:
  - Machine Learning Engineers
  - RL Researchers
  - Data Scientists
difficulty: beginner
content_type: tutorial
---

(gs-getting-started)=

# Getting Started with NeMo Gym

This guide walks you through NeMo Gym from installation to collecting your first verified rollouts—complete training data for reinforcement learning. By the end, you'll understand NeMo Gym's architecture and have hands-on experience with its core workflow.

:::{card}

**What You'll Achieve**

Install NeMo Gym, run a training environment, and generate verified rollouts ready for RL training—typically 15-30 minutes.

^^^

**Learning Objectives**:

1. Understand NeMo Gym's three-server architecture
2. Install and configure NeMo Gym with an LLM backend
3. Run your first training environment
4. Collect and examine verified rollouts
5. Identify your next steps based on your goals

**What you'll need**: Two terminal windows (one for servers, one for commands), OpenAI API key with available credits (~$0.01-0.10).

:::

---

## What is NeMo Gym?

NeMo Gym is a library for building **reinforcement learning (RL) training environments** for large language models. It decouples environment development from training, letting you:

- **Build environments** with tools, tasks, and verification logic
- **Collect rollouts** at scale with automatic reward scoring
- **Integrate** with any RL training framework (NeMo RL, Unsloth, TRL, VeRL)

:::{admonition} Key Term: Rollout
:class: tip

A **rollout** (or trajectory) is a complete record of one task attempt: the input prompt, model responses, tool calls, intermediate steps, and a verification score. Rollouts are the training data for RL—each one shows how the model performed and what reward it earned.
:::

**Why a server architecture?** NeMo Gym uses REST APIs instead of library imports to:

- **Scale independently** — Run model inference on GPUs, rollout collection on CPUs, verification anywhere
- **Swap components** — Change training frameworks without rewriting environment code
- **Parallelize** — Collect thousands of rollouts concurrently across distributed systems

```{mermaid}
flowchart LR
    A[Input Tasks] --> B[Agent Orchestrates]
    B --> C[Model Generates]
    C --> D[Tools Execute]
    D --> E[Verifier Scores]
    E --> F[Training Data]
    style F fill:#90EE90
```

---

## Prerequisites

### Hardware Requirements

NeMo Gym runs on standard development machines:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Any modern x86_64 or ARM64 | Multi-core processor |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 5 GB free | 10 GB+ |
| **GPU** | Not required | See note below |

:::{note}
**GPU not required for NeMo Gym itself.** GPUs are only needed for:
- Running local models via vLLM (instead of API-based models)
- RL training with frameworks like NeMo RL
:::

### Software Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Operating System** | Linux (Ubuntu 20.04+), macOS (11.0+), Windows (WSL2) | |
| **Python** | 3.12+ | Required |
| **Git** | Any recent version | For cloning the repository |
| **Internet** | Required | For API access and dependencies |

### API Access

For this guide, you'll need an **OpenAI API key** with available credits (minimal cost: ~$0.01-0.10 for all examples).

:::{dropdown} Don't have an OpenAI API key?
:icon: info

**Options**:

1. **Get an OpenAI key**: Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. **Use Azure OpenAI**: See {doc}`/model-server/azure-openai`
3. **Self-host with vLLM**: See {doc}`/reference/faq` for setup instructions

NeMo Gym works with any OpenAI-compatible inference endpoint.
:::

---

## Step 1: Install NeMo Gym

Clone the repository and install dependencies using [uv](https://github.com/astral-sh/uv), a fast Python package manager.

```bash
# Clone the repository
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install NeMo Gym with development dependencies
uv sync --extra dev --group docs
```

**✅ Verify Installation**

```bash
ng_version
```

You should see version output like `NeMo-Gym 0.x.x`.

:::{note}
**What does "ng" stand for?** All NeMo Gym CLI commands use the `ng_` prefix (short for **N**eMo **G**ym). For example: `ng_run`, `ng_collect_rollouts`, `ng_viewer`.
:::

:::{dropdown} Troubleshooting: Command not found
:icon: alert

If `ng_version` is not found:

1. Ensure your virtual environment is activated: `source .venv/bin/activate`
2. Verify the prompt shows `(.venv)` or similar
3. Re-run `uv sync --extra dev --group docs`
:::

---

## Step 2: Configure Your API Key

Create an `env.yaml` file to store your model credentials:

```bash
cat > env.yaml << 'EOF'
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-openai-api-key
policy_model_name: gpt-4.1-2025-04-14
EOF
```

:::{important}
Replace `sk-your-actual-openai-api-key` with your real API key from [OpenAI](https://platform.openai.com/api-keys).

The `env.yaml` file is gitignored by default to keep your secrets safe.
:::

**✅ Verify API Configuration**

```bash
python -c "
from openai import OpenAI
from nemo_gym.global_config import get_global_config_dict

config = get_global_config_dict()
client = OpenAI(api_key=config['policy_api_key'], base_url=config['policy_base_url'])
response = client.chat.completions.create(
    model=config['policy_model_name'],
    messages=[{'role': 'user', 'content': 'Say hello'}],
    max_tokens=10
)
print('✅ API configured successfully!')
print(f'Response: {response.choices[0].message.content}')
"
```

---

## Step 3: Understand the Architecture

Before running servers, understand NeMo Gym's three-component architecture:

```{image} ../_images/product_overview.svg
:alt: NeMo Gym Architecture
:width: 700px
:align: center
```

| Component | Role | Example |
|-----------|------|---------|
| **Model Server** | Stateless LLM inference | OpenAI API, vLLM, Azure |
| **Resources Server** | Tools + verification logic | Weather API, code executor, math verifier |
| **Agent Server** | Orchestrates model ↔ resources | Manages multi-step tool calling |

**Communication**: All servers communicate via REST APIs (HTTP/JSON). A head server (port 11000) provides service discovery—components register on startup and find each other through it.

**How they work together**:

1. **Agent** receives a task and coordinates the rollout
2. **Model** generates responses (text, tool calls)
3. **Resources** execute tools and verify task completion
4. **Agent** returns the complete rollout with a reward score

:::{tip}
This separation lets you iterate on environments without modifying training code, and vice versa.
:::

---

## Step 4: Start the Training Environment

Start the servers for a simple weather-tool environment.

:::{important}
**Keep this terminal open.** Servers must keep running while you work in a second terminal.
:::

**Terminal 1** (servers — keep this running):

```bash
# Define config paths for the example environment
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,responses_api_models/openai_model/configs/openai_model.yaml"

# Start all servers
ng_run "+config_paths=[${config_paths}]"
```

**✅ Verify Servers Started**

Look for output showing multiple servers running:

```text
INFO:     Uvicorn running on http://127.0.0.1:11000 (Press CTRL+C to quit)
INFO:     Uvicorn running on http://127.0.0.1:XXXXX (Press CTRL+C to quit)
...
```

The **head server** always runs on port `11000`. Other servers get automatically assigned ports.

:::{dropdown} What servers are running?
:icon: info

Query the head server to see all registered servers:

```bash
curl -s http://localhost:11000/server_instances | python -m json.tool
```

This shows all servers with their names, types, and ports.
:::

---

## Step 5: Interact with the Agent

Open a **new terminal window** (keep servers running in Terminal 1).

**Terminal 2** (commands):

```bash
# Navigate to the Gym directory where you cloned the repo
cd Gym  # adjust if you cloned elsewhere

# Activate the virtual environment
source .venv/bin/activate

# Run the example client
python responses_api_agents/simple_agent/client.py
```

**✅ Expected Output**

```json
[
    {
        "name": "get_weather",
        "arguments": "{\"city\":\"San Francisco\"}",
        "type": "function_call",
        "status": "completed"
    },
    {
        "call_id": "...",
        "output": "{\"city\": \"San Francisco\", \"weather_description\": \"The weather in San Francisco is cold.\"}",
        "type": "function_call_output"
    },
    {
        "content": [{"text": "The weather in San Francisco tonight is cold...", "type": "output_text"}],
        "role": "assistant",
        "type": "message"
    }
]
```

**What happened**:

1. The agent sent a weather query to the model
2. The model called the `get_weather` tool
3. The resources server returned mock weather data
4. The model generated a natural language response

---

## Step 6: Collect Verified Rollouts

Now generate training data by collecting rollouts from a dataset.

**Terminal 2** (keep servers running in Terminal 1):

```bash
# Collect rollouts from the example dataset (single line)
ng_collect_rollouts +agent_name=example_single_tool_call_simple_agent +input_jsonl_fpath=resources_servers/example_single_tool_call/data/example.jsonl +output_jsonl_fpath=results/my_first_rollouts.jsonl +limit=5 +num_repeats=2
```

:::{dropdown} What does the `+` prefix mean?
:icon: info

NeMo Gym uses [Hydra](https://hydra.cc/) for configuration. The `+` prefix adds parameters to the config:

- `+agent_name=...` — adds `agent_name` to the config
- `+limit=5` — adds `limit=5` to the config

Without `+`, you'd be overriding existing config values. With `+`, you're adding new ones.
:::

| Parameter | Description |
|-----------|-------------|
| `+agent_name` | Which agent to use for rollout collection |
| `+input_jsonl_fpath` | Input dataset (tasks to execute) |
| `+output_jsonl_fpath` | Where to save rollouts |
| `+limit` | Maximum examples to process |
| `+num_repeats` | Rollouts per example (for variance) |

**✅ Expected Output**

```text
Found 10 rows!
Repeating rows from 5 to 10!
Collecting rollouts: 100%|████████████████| 10/10 [00:08<00:00,  1.25it/s]
{
    "reward": 1.0
}
```

The average reward of `1.0` indicates all rollouts were verified as successful.

---

## Step 7: Examine Your Rollouts

### View with the Rollout Viewer

```bash
ng_viewer +jsonl_fpath=results/my_first_rollouts.jsonl
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser. The viewer displays:

- **Input**: Original query and available tools
- **Response**: Tool calls and model output
- **Reward**: Verification score (0.0–1.0)

### View Raw Data

```bash
head -1 results/my_first_rollouts.jsonl | python -m json.tool
```

Each rollout contains:

```text
{
    "responses_create_params": { ... },  // Input task
    "response": { ... },                  // Model output + tool calls
    "reward": 1.0                         // Verification score
}
```

---

## Step 8: Clean Up

**Terminal 1**: Press `Ctrl+C` to stop the servers.

:::{dropdown} Operational Notes
:icon: gear

**Graceful shutdown**: `Ctrl+C` sends SIGINT, which stops accepting new requests and completes in-flight operations before shutting down.

**Logs**: Server logs appear in stdout/stderr of the `ng_run` process. For production, redirect to files:

```bash
ng_run "+config_paths=[...]" > logs/server.log 2>&1
```

**Health check**: Query the head server to verify all components are running:

```bash
curl http://localhost:11000/server_instances
```

**Concurrent rollout collection**: Multiple `ng_collect_rollouts` processes can write to the same JSONL file safely (atomic line writes), but use separate output files for cleaner data management.
:::

:::{dropdown} Security Considerations
:icon: shield

**API keys**: The `env.yaml` file stores credentials in plaintext. For production:

1. Use environment variables: `policy_api_key: ${OPENAI_API_KEY}`
2. Set restrictive file permissions: `chmod 600 env.yaml`
3. Never commit `env.yaml` to version control (it's gitignored by default)

**Head server**: The head server at `localhost:11000` is unauthenticated by default. It only binds to localhost, but any process on the machine can query it. For multi-user environments, implement network isolation.

**Rate limits**: When collecting rollouts at scale, you may hit API rate limits (HTTP 429). Use `+num_samples_in_parallel` to throttle concurrent requests.
:::

---

## Understanding Verification

The `reward` score comes from the **verify()** function in your resources server. For the example server:

```python
# resources_servers/example_single_tool_call/app.py
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

This simple verifier returns `1.0` for all responses. Real verifiers implement custom logic:

| Environment | Verification Logic |
|-------------|-------------------|
| **Math** | Check if answer matches ground truth |
| **Code** | Run unit tests on generated code |
| **Tool Use** | Verify correct tool sequence |
| **MCQA** | Compare selected option to answer key |

:::{tip}
Browse available resource servers at [github.com/NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers) for production-ready verification implementations.
:::

---

## What You've Learned

By completing this guide, you've:

- ✅ Installed NeMo Gym and configured an LLM backend
- ✅ Understood the three-server architecture (Model, Resources, Agent)
- ✅ Started a training environment and interacted with an agent
- ✅ Collected verified rollouts—training data for RL
- ✅ Examined rollouts with the viewer and raw JSON

---

## What's Next?

Choose your path based on your goals:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train Your First Model
:link: first-training-run
:link-type: doc

Run RL training using your rollouts with Unsloth (single GPU) or NeMo RL (multi-node).
+++
{bdg-primary}`recommended` {bdg-secondary}`training`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Use Existing Environments
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse training-ready environments for math, code, tool-use, and more.
+++
{bdg-secondary}`environments`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build Custom Environments
:link: ../tutorials/creating-resource-server
:link-type: doc

Create your own resource server with custom tools and verification.
+++
{bdg-secondary}`custom`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Scale Rollout Collection
:link: rollout-collection
:link-type: doc

Learn advanced rollout collection parameters for production workloads.
+++
{bdg-secondary}`scale`
:::

::::

---

## Quick Reference

### CLI Commands

| Command | Description |
|---------|-------------|
| `ng_run` | Start servers from config files |
| `ng_collect_rollouts` | Collect rollouts at scale |
| `ng_viewer` | Launch rollout visualization UI |
| `ng_status` | Check server status |
| `ng_version` | Display NeMo Gym version |
| `ng_help` | Show all available commands |

### Project Structure

```text
Gym/
├── env.yaml                    # Your API credentials (gitignored)
├── nemo_gym/                   # Core library
├── resources_servers/          # Training environments
│   ├── example_single_tool_call/
│   ├── math_with_judge/
│   ├── code_gen/
│   └── ...
├── responses_api_models/       # Model integrations
│   ├── openai_model/
│   ├── vllm_model/
│   └── ...
└── responses_api_agents/       # Agent implementations
    └── simple_agent/
```

### Useful Links

- **Documentation**: [docs.nvidia.com/nemo/gym](https://docs.nvidia.com/nemo/gym/latest/index.html)
- **GitHub**: [github.com/NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)
- **Issues**: [github.com/NVIDIA-NeMo/Gym/issues](https://github.com/NVIDIA-NeMo/Gym/issues)
