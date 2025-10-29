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

Rollouts serve multiple critical purposes in the agent development lifecycle:

### For Reinforcement Learning

Rollouts provide the reward signals needed for RL algorithms like PPO and DPO. Each rollout includes a verification score that tells the training algorithm whether the agent's behavior should be reinforced or corrected.

Without rollouts, you have no way to train agents to improve their performance on specific tasks.

### For Evaluation

Rollouts let you benchmark agent performance systematically. Generate rollouts across a test set, analyze success rates, compare different agent configurations, and track improvement over time.

### For Debugging

When an agent fails, rollouts show you exactly where things went wrong. Did it call the wrong tool? Use incorrect arguments? Misinterpret the tool's response? The complete interaction trace makes debugging straightforward.

### For Research

Rollouts enable analysis of agent behavior patterns: which tools do agents prefer, how do they combine tools, what reasoning strategies emerge, how does performance vary across task types.

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

**Step 1: Prepare Input**
- Create or download a dataset of tasks (JSONL format)
- Each line contains a task with `responses_create_params.input`

**Step 2: Configure System**
- Start your agent server (which connects to model and resource servers)
- Configure which agent to use, where input comes from, where output goes

**Step 3: Generate Rollouts**
- Run `ng_collect_rollouts` with your configuration
- System processes tasks in parallel with progress tracking
- Each task → agent interaction → verified rollout → saved to file

**Step 4: Analyze Results**
- View rollouts interactively with `ng_viewer`
- Compute aggregate metrics (success rate, avg reward, etc.)
- Filter, analyze, or prepare data for RL training

**Step 5: Iterate**
- Use rollouts for training
- Evaluate on new tasks
- Debug failures
- Refine and repeat

---

## Generation Strategies

Different research goals require different collection strategies:

### High-Throughput Collection

For large-scale training data generation:
- Process full datasets (no limit)
- High parallelism (10-20 concurrent requests)
- Single sample per task (num_repeats=1)
- **Use case**: Creating training datasets

### Behavioral Exploration

For understanding agent capabilities:
- Limited samples for quick iteration (limit=100)
- Multiple attempts per task (num_repeats=3-5)
- Higher temperature for diversity
- **Use case**: Research and analysis

### Precise Evaluation

For benchmark measurement:
- Full test set (no limit)
- Single deterministic sample (temperature=0.1)
- Lower parallelism for consistency
- **Use case**: Performance evaluation

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
**Output File Format**:
JSONL (one rollout per line) for easy streaming and processing.

**Automatic Metrics**:
After collection completes, see aggregate statistics:
```json
{
  "avg_reward": 0.73,
  "accuracy": 0.68,
  "avg_tool_calls": 2.1,
  "success_rate": 0.71
}
```

**Metric Computation**:
The system automatically aggregates any numeric fields returned by the resource server's verification, giving you instant feedback on agent performance.
:::

---

## Best Practices

**Start Small**: Use `limit=10` during development to iterate quickly, then scale to full datasets for production.

**Monitor Performance**: Watch success rates and rewards during generation—if they're unexpectedly low, stop and investigate rather than generating large amounts of low-quality data.

**Version Control**: Include version info and dates in output filenames for reproducibility.

**Resource Management**: Match parallelism to your infrastructure—respect API rate limits for cloud models, maximize throughput for local models.

---

## Next Steps

Now that you understand rollout collection conceptually:

- **{doc}`core-abstractions`** — Review how Models, Resources, and Agents work together
- **{doc}`verifying-agent-results`** — Deep dive into how Resources score performance
- **{doc}`../features`** — See available resource servers and their capabilities

**Ready for hands-on practice?** Check out the {doc}`../../tutorials/05-rollout-collection` tutorial for step-by-step instructions on generating your first rollouts.
