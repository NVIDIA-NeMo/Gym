# GRPO Training with NeMo RL: Multi-step tool calling  on Nemotron Nano v2 9B

## Overview

This tutorial trains NVIDIA [Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) to improve its **multi-step tool-calling** capability using **GRPO (Group Relative Policy Optimization)** algorithm on the **Workplace Assistant** environment. Workplace Assistant is a realistic office simulation (calendar, email, project management, etc.) with complex multi-step tasks, providing a strong data distribution for training enterprise-ready tool-using assistants.

**Total time estimate:** ~3-5 hours (including environment setup, data preparation, and training)

> **TL;DR:** Want to jump straight to running commands? Skip to [Setup Instructions](#setup-instructions) or [Running Training](#running-training).

---

## Objectives

In this tutorial, you will:

1. Set up NeMo RL and NeMo Gym for Reinforcement Learning (RL) training
2. Understand the Workplace Assistant environment and its multi-step tool calling capability
3. Configure and run GRPO training on Nemotron Nano v2 9B using this environment in Gym
4. Monitor training progress via Weights & Biases (W&B)

---

## Prerequisites

### Required Knowledge

- You should be comfortable with Python, LLM fine-tuning, and basic reinforcement learning concepts such as policy optimization, rewards, and rollouts. While in-depth knowledge of Reinforcement Learning with Verifiable Rewards (RLVR) and the GRPO algorithm is not required, a high-level understanding is helpful.
- Some basic familiarity with Slurm is useful, but you can follow along using the example commands provided below.

### Hardware Requirements

**Minimum** 1 node of 8× NVIDIA GPUs with 80GB or more memory each (e.g., H100, A100) is required.

NeMo Gym does not require GPUs. GPUs are only necessary for GRPO training with NeMo RL.


### Required Accounts & Tokens
| Service                | Purpose                  | How to Obtain                         |
|------------------------|--------------------------|---------------------------------------|
| Hugging Face (HF)      | Model and data downloads | [Create account](https://huggingface.co/join) |
| Weights & Biases (W&B) | Training metrics logging (optional but recommended) | [Create account](https://wandb.ai/signup)      |

> **Note:** W&B is optional but recommended for tracking training metrics and visualizing progress.

---

## About the Environment and Dataset

The Workplace Assistant is a **multi-step agentic tool-use environment** that tests an AI agent's ability to execute business tasks in a simulated workplace setting.

### Overview

- **5 databases**: Email, Calendar, Analytics, Project Management, Customer Relationship Manager (CRM)
- **26 tools** distributed across these databases
- **690 tasks** representing common business activities (e.g., sending emails, scheduling meetings, managing projects)
- **State-based verification**: Evaluates task completion by comparing final database states rather than exact action sequences

### Environment: Resource Server (`app.py`)

The environment is implemented as a FastAPI-based resource server that manages tool execution and verification. Here's how it works:

#### 1. Session Management

Each rollout gets its own isolated session with fresh tool environments:

```python
async def seed_session(self, request: Request, body: BaseSeedSessionRequest):
    session_id = request.session[SESSION_ID_KEY]
    toolkits = [
        "email",
        "calendar",
        "analytics",
        "project_management",
        "customer_relationship_manager",
    ]
    self.session_id_to_tool_env[session_id] = get_tools(toolkits)
    return BaseSeedSessionResponse()
```

This ensures each task starts with a clean slate and tool calls from different rollouts don't interfere.

#### 2. Dynamic Tool Routing

Tool calls are routed to Python functions:

```python
def route_to_python_function(tool_name, arguments):
    try:
        result = tool_env["functions"][tool_name](**arguments)
        return WorkbenchResponse(output=result)
    except Exception as e:
        # Return error to model so it can self-correct (don't terminate)
        return WorkbenchResponse(output=f"Error executing tool: {str(e)}")
```

**Key feature**: Tool execution errors are returned to the model as part of the response (rather than terminating the rollout), allowing the agent to self-correct and retry during execution.

#### 3. State Matching for Verification

The environment uses **state-matching verification**: instead of requiring exact tool sequences, it compares final database states.

```python
async def verify(self, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
    ground_truth = body.ground_truth
    response = body.response.output

    total_score = 0.0

    # Convert list of ResponseFunctionToolCall objects into list of dictionaries
    predicted_function_calls = []
    for message in response:
        if message.type == "function_call":
            predicted_function_calls.append(message.model_dump())

    predicted_chat_content = []
    for message in response:
        if message.type == "output_text":
            predicted_chat_content.append(message.model_dump())

    total_score += is_correct(predicted_function_calls, ground_truth, None) * 1.0
    return WorkbenchVerifyResponse(**body.model_dump(), reward=total_score)
```

The `is_correct` function implements the state-matching logic:

```python
def is_correct(predicted_actions, ground_truth_actions, error):
    ..
    
    # Execute both sequences in fresh environments
    predict_env = execute_actions_and_reset_state(predicted_actions)
    ground_truth_env = execute_actions_and_reset_state(ground_truth_actions)
    
    .. # Extract specific state info

    # Compare final states of all 5 databases
    return (
        predicted_calendar_state.equals(ground_truth_calendar_state) and
        predicted_email_state.equals(ground_truth_email_state) and
        predicted_analytics_state.equals(ground_truth_analytics_state) and
        predicted_project_management_state.equals(ground_truth_project_management_state) and
        predicted_customer_relationship_manager_state.equals(ground_truth_customer_relationship_manager_state)
    )
```

**Why State-matching verification?**:
- **Flexibility**: Multiple valid solution paths exist for the same task
- **Robustness**: Agent can recover from mistakes mid-trajectory
- **Goal-oriented**: Focuses on outcomes, not specific procedures


---

### Workplace Assistant Dataset

 [`The Workplace Assistant`](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant) dataset associated with this environment contains **690 unique tasks** with a full dataset of **1,260 prompts** that simulate realistic office productivity scenarios requiring multi-step tool usage. Each task is presented as a natural language request that the model must decompose into appropriate tool calls (up to 6 steps per task).

### Dataset Structure

Each sample in the dataset contains:
- **System prompt**: Provides current date/time context and constraints (e.g., "Meetings must not start before 9am or end after 6pm")
- **User query**: Natural language task description (e.g., "Reply to carlos's last email...")
- **Available tools**: JSON schemas for all 26 functions the model can call
- **Ground truth actions**: Reference solution as a sequence of tool calls (used for state-matching verification)


### Available Tools

The environment provides 26 functions across five business domains, each operating on CSV-backed databases. The agent must select the right tools, extract parameters from natural language, and chain them together to complete tasks.

**Tool Categories:**
- 📧 **Email** (6 tools): send, search, reply, forward, delete, get by ID (e.g., `email_send_email`, `email_search_emails`)
- 📅 **Calendar** (5 tools): create, search, update, delete, get by ID (e.g., `calendar_create_event`)
- 📊 **Analytics** (6 tools): create plots, count metrics, get visitor data (e.g., `analytics_create_plot`, `analytics_engaged_users_count`)
- 📋 **Project Management** (5 tools): create, search, update, delete, get task details (e.g., `project_management_update_task`)
- 👥 **CRM** (4 tools): search, add, update, delete customers (e.g., `customer_relationship_manager_search_customers`)
- 🔍 **Company Directory** (1 tool): `company_directory_find_email_address` - case-insensitive name lookup, always available

### Example Tasks

Each task is a natural language request that the model must complete using the available tools. The environment allows up to 6 tool-calling steps per task.

**Single-Step Task** (1 tool call needed):

```json
{
  "input": [
    {
      "role": "system",
      "content": "Today's date is Thursday, 2023-11-30 and the current time is 23:59:00. Remember the current date and time when answering queries. Meetings must not start before 9am or end after 6pm."
    },
    {
      "role": "user", 
      "content": "Send an email to john.smith@atlas.com with the subject 'Team Meeting' and body 'Let's meet tomorrow at 2pm to discuss the project.'"
    }
  ],
  "tools": [
    {"type": "function", "name": "email_send_email", "description": "Sends an email to a recipient.", "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["recipient", "subject", "body"]}},
    {"type": "function", "name": "email_search_emails", "description": "Searches for emails matching the given query...", "parameters": {...}},
    {"type": "function", "name": "calendar_create_event", "...": "..."},
    // ... 23 more tools (calendar, analytics, project_management, CRM)
  ],
  "parallel_tool_calls": false,
  "temperature": 1.0
}
```

**Expected output:** `email_send_email(recipient="alex.martinez@atlas.com", subject="Team Meeting", body="Let's meet tomorrow at 2pm to discuss the project.")`

---

**Multi-Step Task** (requires 3-6 tool calls):

```json
{
  "input": [
    {
      "role": "system",
      "content": "Today's date is Thursday, 2023-11-30 and the current time is 23:59:00. Remember the current date and time when answering queries. Meetings must not start before 9am or end after 6pm."
    },
    {
      "role": "user",
      "content": "John is taking over all of Akira's leads that are interested in software. Can you reassign them in the crm?"
    }
  ],
  "tools": [
    {"type": "function", "name": "customer_relationship_manager_search_customers", "description": "Searches for customers based on the given parameters with pagination support.", "parameters": {"type": "object", "properties": {"assigned_to_email": {"type": "string"}, "product_interest": {"type": "string"}, "status": {"type": "string"}, ...}}},
    {"type": "function", "name": "customer_relationship_manager_update_customer", "description": "Updates a customer record by ID.", "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}, "field": {"type": "string"}, "new_value": {"type": "string"}}, "required": ["customer_id", "field", "new_value"]}},
    {"type": "function", "name": "company_directory_find_email_address", "description": "Finds all email addresses containing the given name...", "parameters": {...}},
    // ... 23 more tools
  ],
  "parallel_tool_calls": false,
  "temperature": 1.0
}
```

**Expected output sequence:**
1. `company_directory_find_email_address(name="Akira")` → Returns `"akira.tanaka@atlas.com"`
2. `company_directory_find_email_address(name="John")` → Returns `"john.smith@atlas.com"`
3. `customer_relationship_manager_search_customers(assigned_to_email="akira.tanaka@atlas.com", product_interest="software", status="lead")` → Returns 3 matching leads
4. `customer_relationship_manager_update_customer(customer_id="00000095", field="assigned_to_email", new_value="john.smith@atlas.com")`
5. `customer_relationship_manager_update_customer(customer_id="00000080", field="assigned_to_email", new_value="john.smith@atlas.com")`
6. `customer_relationship_manager_update_customer(customer_id="00000035", field="assigned_to_email", new_value="john.smith@atlas.com")`

**This task demonstrates:**
- **Name resolution**: Looking up email addresses from natural names
- **Search with multiple filters**: Finding customers by assignee, product interest, and status
- **Batch updates**: Iterating through results to update multiple records
- **State verification**: Final database state will match ground truth even if different search parameters or ordering were used

**Generally, the model must:**
1. Understand the user's intent from natural language
2. Determine which tools to call and in what order
3. Infer correct parameters (e.g., look up email addresses, find matching customer records)
4. Execute all necessary steps to complete the task

---

## Setup Instructions

### Step 1: Enter a GPU Node

**Estimated Time:** ~5 minutes

Launch an interactive Slurm session to run training commands. See the [NeMo RL Cluster Setup documentation](https://docs.nvidia.com/nemo/rl/latest/cluster.html#interactive-launching) for more details.

```bash
NUM_ACTOR_NODES=1
ACCOUNT=<ACCOUNT_NAME>
JOB_NAME=<JOB_NAME>
PARTITION=<PARTITION>

# Use the official NeMo RL container from NGC
# See: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.4.0
CONTAINER_WORKDIR=$PWD
MOUNTS="$PWD:$PWD"
srun \
    --nodes=${NUM_ACTOR_NODES} \
    --ntasks=1 \
    --account=${ACCOUNT} \
    --job-name=${JOB_NAME} \
    --partition=${PARTITION} \
    --time=04:00:00 \
    --gres=gpu:8 \
    --no-container-mount-home \
    --container-name=nemo-gym \
    --container-mounts="${MOUNTS}" \
    --container-image="${CONTAINER}" \
    --container-workdir=$CONTAINER_WORKDIR \
    --pty /bin/bash
```

### Step 2: Clone and Setup NeMo RL + NeMo Gym

**Estimated Time:** ~15-20 minutes

```bash
# Clone NeMo RL repository
git clone https://github.com/NVIDIA-NeMo/RL
cd RL

# Clone NeMo Gym as a submodule
git clone https://github.com/NVIDIA-NeMo/Gym.git 3rdparty/Penguin-workspace/Penguin

# Initialize all submodules (Megatron, AutoModel, etc.)
git submodule update --init --recursive

# This will remove any stale cached Ray venv and rebuilt it
# TODO: This is a WAR. Need a formal fix.
rm -rf /opt/ray_venvs/*

# Activate the NeMo RL virtual environment
source /opt/nemo_rl_venv/bin/activate

# Install dependencies
uv sync --group={build,docs,dev,test} --extra nemo_gym
```

### Step 3: Prepare NeMo Gym Data

**Estimated Time:** ~5-10 minutes

The Workplace Assistant dataset must be downloaded from HuggingFace and prepared for training. This is a two-step process:

This runs `ng_prepare_data` to download and validate the dataset, and to add an `agent_ref` property to each example that tells NeMo Gym which agent server should handle that example.

```bash
HF_TOKEN=SPECIFY_HF_TOKEN

# Setup Gym local venv
cd 3rdparty/Gym-workspace/Gym
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv sync --active --extra dev

config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=resources_servers/workplace_assistant/data \
    +mode=train_preparation \
    +hf_token=$HF_TOKEN \
    +should_download=true

# Return to NeMo RL directory and Python env
cd ../../.. && source /opt/nemo_rl_venv/bin/activate
```

### Step 4: Run Sanity Tests (optional but recommended)

**Estimated Time:** ~10-15 minutes

Validate your setup before training:

```bash
HF_HOME=.cache/ \
HF_TOKEN=${HF_TOKEN} \
    ./examples/nemo_gym/run_nemo_gym_single_node_sanity_tests.sh
```

> **Note**: If you've run these tests before and encounter HuggingFace rate limit errors, add `HF_HUB_OFFLINE=1` to the command.

---

## Training Configuration

### Key Configuration Parameters

The training configuration file is located at:
`examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml`

#### Environment Configuration

```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - resources_servers/workplace_assistant/configs/workplace_assistant.yaml
    workplace_assistant_simple_agent:
      responses_api_agents:
        simple_agent:
          max_steps: 6  # Maximum tool-calling steps per task
```

#### GRPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_prompts_per_step` | 4 | Number of prompts per training step |
| `num_generations_per_prompt` | 4 | Rollouts generated per prompt |
| `max_rollout_turns` | 1 | Turns per rollout (1 turn, up to 6 tool steps) |
| `max_num_steps` | 10 | Total training steps |
| `use_leave_one_out_baseline` | true | Variance reduction technique |
| `normalize_rewards` | true | Normalize rewards across batch |

#### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | nvidia/NVIDIA-Nemotron-Nano-9B-v2 | Base model |
| `max_total_sequence_length` | 32768 | Maximum context length |
| `precision` | bfloat16 | Training precision |
| `tensor_model_parallel_size` | 8 | Tensor parallelism across GPUs |

#### Optimizer Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `optimizer` | Adam | Optimizer type |
| `lr` | 5.0e-6 | Learning rate |
| `min_lr` | 5.0e-7 | Minimum learning rate |
| `weight_decay` | 0.01 | Weight decay |
| `adam_beta1` / `adam_beta2` | 0.9 / 0.999 | Adam hyperparameters |
| `clip_grad` | 1.0 | Gradient clipping threshold |

The complete training configuration is available at:
[`examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml)

---

## Running Training

### Single Node Training (interactive mode)

**Estimated Time:** ~2-4 hours

Run these commands **from inside the container** after attaching via the interactive session from Step 1:

```bash
# Clean up any existing Ray/vLLM processes
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"

# Set experiment name with timestamp
EXP_NAME="$(date +%Y%m%d)/nemo_gym_grpo/nemotron_nano_v2_9b/workplace_assistant_001"

# Configuration file path
CONFIG_PATH=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Launch training
# Set these environment variables before running:
TORCH_CUDA_ARCH_LIST="9.0 10.0" \
HF_HOME=.cache/ \
HF_TOKEN="YOUR_HUGGINGFACE_TOKEN" \
WANDB_API_KEY="YOUR_WANDB_API_KEY" \
NRL_FORCE_REBUILD_VENVS=true \
VLLM_LOGGING_LEVEL=ERROR \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="${USER}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME
```


### Multi-Node Training

Scale to multiple nodes by changing `cluster.num_nodes`. This example uses **batch mode** (the `COMMAND` variable specifies what to run automatically when the job starts).

> **Note**: Run this command from the **Slurm login/head node**, not from inside the interactive container from Step 1. This submits a new batch job that will run independently.

```bash
# Set experiment name
EXP_NAME="nemo_gym_grpo/nemotron_nano_v2_9b/2nodes/workplace_assistant_001"
CONFIG_PATH=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

NUM_NODES=2
COMMAND="TORCH_CUDA_ARCH_LIST='9.0 10.0' HF_HOME=.cache/ HF_TOKEN='YOUR_HUGGINGFACE_TOKEN' WANDB_API_KEY='YOUR_WANDB_API_KEY' NRL_FORCE_REBUILD_VENVS=true VLLM_LOGGING_LEVEL=ERROR uv run python examples/nemo_gym/run_grpo_nemo_gym.py --config=$CONFIG_PATH cluster.num_nodes=$NUM_NODES logger.wandb.project=${USER}-nemo-gym-rl-integration logger.wandb.name=$EXP_NAME logger.log_dir=results/$EXP_NAME checkpointing.checkpoint_dir=results/$EXP_NAME" \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.4.0 \
MOUNTS="/shared/filesystem:/shared/filesystem" \
sbatch \
    --nodes=$NUM_NODES \
    --time=4:0:0 \
    --job-name=grpo-workplace-assistant \
    --gres=gpu:8 \
    ray.sub
```

---

## Expected Results

### Training Metrics

Monitor these metrics in W&B to track progress:

| Metric | Initial | After 1 Epoch | Description |
|--------|---------|---------------|-------------|
| `train:reward_mean` | ~0.1-0.2 | ~0.5-0.7 | Average reward per batch |
| `val:accuracy` | ~0.15 | ~0.5-0.6 | Validation task completion rate |
| `train:loss` | ~0.5 | ~0.2-0.3 | GRPO policy loss |

### Checkpoint Outputs

Checkpoints are saved to:
```
results/<EXP_NAME>/
├── step_6/
├── step_12/
├── step_18/
└── ...
```

The best checkpoint (highest `val:accuracy`) is retained based on `checkpointing.keep_top_k: 3`.

### Success Criteria

Training is successful when:
- Reward mean increases consistently over steps
- Validation accuracy improves from baseline (~15%) to 50%+
- No OOM (Out of Memory) errors
- Checkpoints are saved at specified intervals

### Validation Reward Plot

<!-- TODO: Add validation reward plot showing improvement over training steps -->
![Validation Reward Plot](images/val_reward_placeholder.png)
*Expected: Validation reward increasing from ~0.15 to ~0.5+ over the course of training.*

### Measuring Real-World Improvement

The Workplace Assistant environment's tool-calling tasks correlate with performance on the [Berkeley Function Calling Leaderboard (BFCL) v3](https://gorilla.cs.berkeley.edu/leaderboard.html) benchmark. To measure improvement, evaluate the Nemotron Nano v2 9B model on BFCL v3 before and after training and compare.  You should observe measurable improvement in tool-calling accuracy

You can run BFCL v3 evaluations using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator), which supports BFCL v3. See the [NeMo Evaluator docs](https://github.com/NVIDIA-NeMo/Evaluator#-supported-benchmarks-and-evaluation-harnesses) for full setup instructions and supported benchmarks.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| HuggingFace rate limits | Specify your HF API token and/or add `HF_HUB_OFFLINE=1` after the initial download |
| vLLM process not shutting down | Run `pkill -f VllmAsyncGenerationWorker` before training |
| Ray cluster issues | Run `ray stop --force` before training |
| CUDA OOM | Increase `tensor_parallel_size`, lower batch sizes |
| Slow initial startup | Set `NRL_FORCE_REBUILD_VENVS=true` on first run only; if `uv` gets rate limited, set this back to `false` |

### Log Locations

```
logs/grpo-workplace-assistant-nemotron-nano-v2-9b/  # Training logs
results/<EXP_NAME>/                                  # Checkpoints and metrics
.cache/                                              # HuggingFace model cache
```

---

## Next Steps

After completing this tutorial, explore:

1. **Scale Up**: Try multi-node training for faster convergence and larger batch sizes
2. **Hyperparameter Tuning**: Adjust learning rate, number of generations, or reward normalization 
3. **Deploy Your Agent**: Export the trained checkpoint and deploy it with vLLM or NVIDIA NIM to build a production workplace assistant that integrates with real calendar, email, and file management APIs

### Related Tutorials

- [RL Training with NeMo RL](./rl-training-with-nemo-rl.md) - General RL training guide
- [GRPO Loss Configuration](../../docs/guides/grpo.md) - Advanced loss function customization
- [Sequence Packing](../../docs/design-docs/sequence-packing-and-dynamic-batching.md) - Optimize training throughput

---

## References

- **NeMo RL Repository**: [github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)
- **NeMo Gym Repository**: [github.com/NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)

---

## Appendix: Full Configuration Reference

