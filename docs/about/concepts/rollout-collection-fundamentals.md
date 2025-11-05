(concepts-rc-fundamentals)=
# Rollout Collection Fundamentals

Rollouts are the heart of NeMo Gym: complete records of agent interactions that power reinforcement learning, evaluation, and research.

Think of rollouts as "interaction transcripts" that capture everything that happens when an agent tackles a task: the original question, the agent's reasoning process, which tools it used, the results it got, and how well it performed overall.

This transparency is what makes reinforcement learning possible: you can see exactly what worked, what didn't, and generate the reward signals needed to improve agent behavior.

---

## What is a Rollout?

A rollout is a complete record of a single agent interaction, from start to finish.

**It captures**:

```{list-table}
:header-rows: 0
:widths: 30 70

* - **Input**
  - The task or question the agent was given
* - **Reasoning**
  - How the agent thought about the problem
* - **Actions**
  - Which tools the agent called and what arguments it used
* - **Tool Results**
  - What each tool returned
* - **Final Response**
  - The agent's answer or output
* - **Verification**
  - How well the agent performed (reward signal from 0.0 to 1.0)
```

**Example**: If you ask an agent "What's 47 × 23?", a rollout captures:
- The agent deciding to use a calculator tool
- Calling `calculate(47 * 23)`  
- Receiving `1081`
- Responding "The answer is 1,081"
- The math verifier scoring it as correct (reward: 1.0)

---

## Why Rollouts Matter

Rollouts serve multiple critical purposes in the agent development lifecycle. Choose the perspective most relevant to your current work:

::::{tab-set}

:::{tab-item} Reinforcement Learning
Rollouts provide the reward signals needed for RL algorithms like PPO and DPO. Each rollout includes a verification score that tells the training algorithm whether the agent's behavior should be reinforced or corrected.

Without rollouts, you have no way to train agents to improve their performance on specific tasks.

**Key value**: Reward signals for training algorithms
:::

:::{tab-item} Evaluation
Rollouts let you benchmark agent performance systematically. Generate rollouts across a test set, analyze success rates, compare different agent configurations, and track improvement over time.

**Key value**: Systematic performance measurement and comparison
:::

:::{tab-item} Debugging
When an agent fails, rollouts show you exactly where things went wrong. Did it call the wrong tool? Use incorrect arguments? Misinterpret the tool's response? The complete interaction trace makes debugging straightforward.

**Key value**: Complete interaction traces for root cause analysis
:::

:::{tab-item} Research
Rollouts enable analysis of agent behavior patterns: which tools do agents prefer, how do they combine tools, what reasoning strategies emerge, how does performance vary across task types.

**Key value**: Data for behavioral analysis and insights
:::

::::

---

## How Rollouts are Generated

Rollout generation brings together all three core abstractions working in harmony.

### The Players

1. **Input Dataset**: Tasks in JSONL format (one task per line)
2. **Agent Server**: Orchestrates the interaction
3. **Model Server**: Provides the LLM reasoning
4. **Resource Server**: Provides tools and verification
5. **Collection Orchestrator**: Manages parallelism and output

### The Process

Let's trace how a math problem becomes a rollout:

1. **Load Task**: System reads "What is 15% of 240?" from input JSONL

2. **Agent Receives Task**: Agent gets the question plus available tools
   > Tools: `calculator(expression: str)`

3. **Agent → Model**: Sends prompt with tools to model
   > "Here's a math question and a calculator tool. Solve it."

4. **Model → Agent**: Decides on action
   > "I'll calculate: 240 * 0.15"

5. **Agent → Resource**: Executes calculator tool
   > `calculator("240 * 0.15")`

6. **Resource → Agent**: Returns result
   > `"36"`

7. **Agent → Model**: Provides tool result
   > "The calculator returned 36. Respond to the user."

8. **Model → Agent**: Generates final response
   > "15% of 240 is 36"

9. **Resource Verifies**: Checks if answer is correct
   > Correct! Reward: 1.0

10. **Save Rollout**: Complete interaction written to output JSONL

This entire sequence—input, reasoning, tool calls, verification—becomes one rollout.

---

## What's Inside a Rollout

Each rollout is a JSON object with a standard structure:

```{list-table}
:header-rows: 1
:widths: 25 75

* - Field
  - Description
* - `responses_create_params`
  - Original task including the input messages and available tools
* - `output`
  - Complete conversation history: assistant messages, tool calls, tool results
* - `reward`
  - Verification score from the resource server (typically 0.0-1.0)
* - Additional metadata
  - Resource-specific fields like accuracy, test results, etc.
```

**Minimal rollout example**:
```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "What is 2+2?"}],
    "tools": [{"type": "function", "name": "calculator", "..."}]
  },
  "output": [
    {"role": "assistant", "tool_calls": [{"name": "calculator", "arguments": "{\"expr\": \"2+2\"}"}]},
    {"role": "tool", "name": "calculator", "content": "4"},
    {"role": "assistant", "content": "The answer is 4."}
  ],
  "reward": 1.0
}
```

This structure is framework-agnostic—it's just JSON that any RL training library can consume.

---

## The Complete Workflow

Here's how rollout collection fits into the bigger picture:

```{list-table}
:header-rows: 1
:widths: 15 45 40

* - Step
  - Action
  - Details
* - **1. Prepare Input**
  - Create or download task dataset
  - • JSONL format (one task per line) <br>
    • Each task includes `responses_create_params.input`
* - **2. Configure System**
  - Start servers and set parameters
  - • Launch agent server (connects to model and resource) <br>
    • Specify agent, input path, output path
* - **3. Generate Rollouts**
  - Run collection command
  - • Execute `ng_collect_rollouts` with config <br>
    • Parallel processing with progress tracking <br>
    • Each task → interaction → verification → save
* - **4. Analyze Results**
  - Review and evaluate rollouts
  - • View interactively with `ng_viewer` <br>
    • Compute metrics (success rate, avg reward) <br>
    • Prepare data for RL training
* - **5. Iterate**
  - Improve and repeat
  - • Use rollouts for training <br>
    • Evaluate on new tasks <br>
    • Debug failures and refine
```

---

## Generation Strategies

Different research goals require different collection strategies. Choose the approach that matches your objective:

::::{tab-set}

:::{tab-item} High-Throughput
**Goal**: Large-scale training data generation

**Configuration**:
- Process full datasets (no limit)
- High parallelism (10-20 concurrent requests)
- Single sample per task (num_repeats=1)

**Best for**: Creating training datasets for RL algorithms

**Example**:
```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=full_dataset.jsonl \
    +num_samples_in_parallel=20
```
:::

:::{tab-item} Behavioral Exploration
**Goal**: Understanding agent capabilities and behavior patterns

**Configuration**:
- Limited samples for quick iteration (limit=100)
- Multiple attempts per task (num_repeats=3-5)
- Higher temperature for diversity

**Best for**: Research and analysis of agent strategies

**Example**:
```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +limit=100 \
    +num_repeats=5 \
    +responses_create_params.temperature=0.8
```
:::

:::{tab-item} Precise Evaluation
**Goal**: Benchmark measurement with reproducible results

**Configuration**:
- Full test set (no limit)
- Single deterministic sample (temperature=0.1)
- Lower parallelism for consistency

**Best for**: Performance evaluation and benchmarking

**Example**:
```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=benchmark.jsonl \
    +responses_create_params.temperature=0.1 \
    +num_samples_in_parallel=5
```
:::

::::

---

## Technical Details

:::{dropdown} Command-Line Interface
**Basic Command**:
```bash
ng_collect_rollouts \
    +agent_name=your_agent \
    +input_jsonl_fpath=tasks.jsonl \
    +output_jsonl_fpath=rollouts.jsonl
```

**Key Parameters**:
- `agent_name`: Which agent server to use
- `input_jsonl_fpath`: Path to input tasks
- `output_jsonl_fpath`: Where to save rollouts
- `limit`: Number of tasks to process (null = all)
- `num_repeats`: Rollouts per task (null = 1)
- `num_samples_in_parallel`: Concurrent requests (null = default)

**Model Parameters** (override defaults):
- `responses_create_params.temperature`: Control randomness
- `responses_create_params.max_output_tokens`: Response length limit
- `responses_create_params.top_p`: Nucleus sampling parameter
:::

:::{dropdown} Parallel Processing
**Async Orchestration**:
NeMo Gym uses asyncio with semaphores to process multiple tasks concurrently while respecting rate limits.

```python
# Conceptual flow
semaphore = Semaphore(num_samples_in_parallel)
async with semaphore:
    response = await agent_server.post("/run", task)
    save_rollout(response)
```

**Performance Tuning**:
- **Low parallelism (1-5)**: Debugging, API rate limits
- **Medium parallelism (5-10)**: Standard operation
- **High parallelism (10-20+)**: Local models with GPU resources

**Progress Tracking**:
Real-time progress bar shows completion, speed, and running metrics as rollouts are generated.
:::

:::{dropdown} Input Data Format
**Required Structure**:
```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "Your task here"}
    ]
  }
}
```

**Optional Fields**:
- `expected_answer`: For verification
- `test_cases`: For code execution
- `metadata`: Task-specific information
- Any resource server-specific fields

**Multiple Input Formats Supported**:
- Curated datasets from NeMo Gym
- Custom JSONL files
- Converted benchmark datasets
:::

:::{dropdown} Output Data and Metrics
:open:

**Output File Format**:
JSONL (one rollout per line) for easy streaming and processing.

**Automatic Metric Aggregation**:

After `ng_collect_rollouts` completes, NeMo Gym automatically aggregates all numeric fields from verification:

```bash
ng_collect_rollouts +input_jsonl_fpath=tasks.jsonl +output_jsonl_fpath=rollouts.jsonl

# Displays after collection:
# {
#   "reward": 0.73,
#   "accuracy": 0.68,
#   "avg_tool_calls": 2.1
# }
```

**How It Works**:

Any numeric field returned by your resource server's `verify()` method is automatically averaged across all rollouts:

```python
# In your resource server
def verify(self, task, response):
    return {
        "reward": 0.85,           # ← automatically averaged
        "accuracy": 1.0,          # ← automatically averaged
        "custom_metric": 42       # ← any numeric field is averaged
    }
```

**Quick Analysis**:

Use the built-in aggregation script for quick summaries:

```bash
python scripts/print_aggregate_results.py +jsonl_fpath=rollouts.jsonl
```

**See Also**:
- {doc}`../../training/data-quality/index` - Validate quality before training
- {doc}`../../training/rollout-collection/optimize-for-training/production-scale` - Monitor during collection

:::
