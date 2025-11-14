# Rollout Collection Fundamentals

**Goal**: Master NeMo Gym's rollout collection system - the foundation for understanding agent behavior, creating training data, and evaluating performance.

## What Are Rollouts?

**Rollouts are complete records of agent interactions** - from initial input through reasoning, tool calls, and final responses, including verification scores.

Think of rollouts as "interaction transcripts" that capture:
- What the agent was asked to do (input)
- How the agent reasoned (internal processing)  
- What tools the agent used (function calls and responses)
- How well the agent performed (verification scores)
- The final response (output to user)

## Why Generate Rollouts?

Rollouts are the foundation for multiple critical use cases:

- **Reinforcement Learning**: Generate reward signals and training experiences for RL algorithms
- **Evaluation**: Benchmark agent performance across different scenarios
- **Debugging**: Understand agent behavior patterns and identify failure modes
- **Research**: Analyze agent reasoning strategies and tool usage patterns
- **Quality Assurance**: Verify agent behavior before deployment

**NeMo Gym enables systematic rollout generation** - you configure agents and tasks, then NeMo Gym handles the complex orchestration of generating complete interaction data.

## The Rollout Generation Workflow

<!-- TODO: Enable mermaid graph here i.e. ```mermaid``` -->
```
graph LR
    A[Input Dataset] --> B[Agent Server]
    B --> C[Model Reasoning]
    C --> D[Tool Calls]
    D --> E[Verification]
    E --> F[Output Rollouts]
    F --> G[Training Data]
```

1. **Input Dataset**: Tasks or questions in JSONL format
2. **Agent Processing**: Your configured agent reasons about each task
3. **Model Reasoning**: The underlying AI model generates responses and decisions
4. **Tool Execution**: Agent calls available tools and processes responses  
5. **Verification**: Resource server evaluates agent performance
6. **Rollout Export**: Complete interaction traces saved in structured format

## Prerequisites

Before generating rollouts, ensure you have:
- Agent server running
- Input dataset in JSONL format
- Model access: Either API credits or local model server (vLLM)

### Input Data Sources

You can use:
- **Curated datasets**: Download datasets from NeMo Gym's Hugging Face collection
  - TODO: Add link
- **Custom datasets**: Create your own task-specific JSONL files
- **Existing benchmarks**: Convert evaluation datasets to NeMo Gym format

### Input JSONL Schema

Each line in your input JSONL file should follow this schema:

```json
{
    // REQUIRED: responses_create_params
    "responses_create_params": {
        // Input is an array Response input items (like messages) following OpenAI format (role + content)
        "input": [
            {"role": "user", "content": "Your task or question here"}
        ],
        // Optional: Other OpenAI Responses API parameters (tools, temperature, etc.)
    },
    // Optional: Resource server-specific metadata for verification
    // e.g., expected answers, test cases, etc.
    "expected_answer": "...",
    "test_cases": "...",
    "your_resource_specific_metadata_field": "..."
}
```


## Hands-On: Generating Your First Rollouts

Let's generate rollouts using the **Example Multi Step** resource server, which tests reading comprehension across long documents.

### Step 1: Start the Example Multi Step Agent

```bash
# Start the example_multi_step agent server
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**✅ Success Check**: You should see 3 servers running including the `example_multi_step_simple_agent`.

### Step 3: Generate Rollouts

**What this dataset contains**: Complex reading comprehension tasks where agents must find specific information ("needles") within long documents ("haystacks").

In a separate terminal, run:
```bash
# Generate rollouts from the dataset
ng_collect_rollouts +agent_name=example_multi_step_simple_agent \
    +input_jsonl_fpath=resources_servers/example_multi_step/data/example.jsonl \
    +output_jsonl_fpath=results/example_multi_step_rollouts.jsonl \
    +limit=5 \
    +num_repeats=2 \
    +num_samples_in_parallel=3 \
    +responses_create_params.max_output_tokens=8192
```

**What's happening**:
- `limit=5`: Process only the first 5 examples (for quick testing)
- `num_repeats=2`: Generate 2 rollouts per example (10 total rollouts)
- `num_samples_in_parallel=3`: Process 3 requests simultaneously
- `max_output_tokens=8192`: Allow longer responses for complex reasoning

### Step 4: View Your Rollouts

```bash
# Launch the rollout viewer
ng_viewer +jsonl_fpath=results/example_multi_step_rollouts.jsonl
```

**What you'll see**: An interactive viewer showing agent reasoning, tool calls, and verification scores for each rollout.

## Understanding Rollout Data Structure

Each rollout contains the complete interaction trace:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "Find the hidden information in this document..."}
    ],
    "tools": [{"type": "function", "name": "search_document", "..."}]
  },
  "output": [
    {"role": "assistant", "content": "I'll search the document for the information."},
    {"role": "assistant", "tool_calls": [{"function": {"name": "search_document", "arguments": "..."}}]},
    {"role": "tool", "content": "Found: The answer is X"},
    {"role": "assistant", "content": "Based on my search, the answer is X."}
  ],
  "reward": 1.0,
  "success": true,
  "metadata": {
    "reasoning_steps": 3,
    "tool_calls_made": 1,
    "verification_score": 0.95
  }
}
```

**Key components**:
- **responses_create_params**: Original task and available tools
- **output**: Complete conversation including tool calls and responses
- **reward**: Verification score from the resource server
- **metadata**: Additional metrics for analysis

## Rollout Generation Parameters

### Essential Parameters

```bash
ng_collect_rollouts \
    +agent_name=your_agent_name \              # Which agent to use
    +input_jsonl_fpath=input/tasks.jsonl \     # Input dataset
    +output_jsonl_fpath=output/rollouts.jsonl  # Where to save results
```

### Data Control Parameters

```bash
+limit=100 \                    # Limit examples processed (null = all)
+num_repeats=3 \                # Rollouts per example (null = 1)  
+num_samples_in_parallel=5      # Concurrent requests (null = default)
```

### Model Behavior Parameters

```bash
+responses_create_params.max_output_tokens=4096 \     # Response length limit
+responses_create_params.temperature=0.7 \            # Randomness (0-1)
+responses_create_params.top_p=0.9                    # Nucleus sampling
```

## Generation Strategies

### Strategy 1: Consistent Behavior Analysis

```bash
# Generate single, consistent rollouts for analysis
ng_collect_rollouts +agent_name=your_agent \
    +input_jsonl_fpath=analysis_tasks.jsonl \
    +output_jsonl_fpath=consistent_rollouts.jsonl \
    +num_repeats=1 \
    +responses_create_params.temperature=0.1    # Low temperature for consistency
```

**Use case**: Understanding typical agent behavior patterns.

### Strategy 2: Behavioral Diversity Exploration

```bash
# Generate multiple responses to explore agent capabilities
ng_collect_rollouts +agent_name=your_agent \
    +input_jsonl_fpath=evaluation_tasks.jsonl \
    +output_jsonl_fpath=diverse_rollouts.jsonl \
    +num_repeats=4 \                            # Multiple attempts per task
    +responses_create_params.temperature=0.8    # Higher temperature for diversity
```

**Use case**: Exploring the range of agent behaviors and capabilities.

### Strategy 3: Performance Evaluation

```bash
# Comprehensive evaluation across full dataset  
ng_collect_rollouts +agent_name=your_agent \
    +input_jsonl_fpath=benchmark_full.jsonl \
    +output_jsonl_fpath=evaluation_rollouts.jsonl \
    +limit=null \                               # Process entire dataset
    +num_repeats=1 \
    +num_samples_in_parallel=10                 # Faster processing
```

**Use case**: Establishing comprehensive performance baselines.

## Monitoring Generation Progress

During generation, NeMo Gym shows real-time metrics:

```
Collecting rollouts: 100%|████████████| 50/50 [02:45<00:00,  1.82it/s]
{
    "avg_reward": 0.73,
    "success_rate": 0.68,
    "avg_tool_calls": 2.1,
    "avg_response_length": 847
}
```

**Key metrics to monitor**:
- **Average reward**: Higher is generally better (depends on verification)
- **Success rate**: Percentage of tasks completed successfully  
- **Tool usage**: Average tools called per task
- **Response length**: Token usage and verbosity

## Analyzing Generated Data

### Interactive Exploration

```bash
# View rollouts interactively
ng_viewer +jsonl_fpath=results/your_rollouts.jsonl
```

### Command-Line Analysis

```bash
# Filter by success rate
jq 'select(.success == true)' results/your_rollouts.jsonl > successful_rollouts.jsonl

# Analyze reward distribution
jq '.reward' results/your_rollouts.jsonl | sort -n | uniq -c

# Count tool usage patterns
jq '.output[] | select(.tool_calls) | .tool_calls[].function.name' rollouts.jsonl | sort | uniq -c
```

## Best Practices

### 1. Start Small, Scale Up

```bash
# Development: Small batches for quick iteration
+limit=10 +num_repeats=1

# Production: Full datasets with multiple samples
+limit=null +num_repeats=3
```

### 2. Control Parallel Processing

```bash
# Conservative for API rate limits
+num_samples_in_parallel=5    

# Aggressive for local models with sufficient resources
+num_samples_in_parallel=20   
```

### 3. Version Your Data

```bash
# Include version info in output paths
+output_jsonl_fpath=data/rollouts_v2.0_$(date +%Y%m%d).jsonl
```

### 4. Monitor Resource Usage

- **API models**: Watch rate limits and token consumption
- **Local models**: Monitor GPU memory and processing speed
- **Storage**: Rollout files can become large with comprehensive datasets

## Troubleshooting

### Problem: Low Success Rate

```
"success_rate": 0.23
```

**Possible causes**:
- Tasks too difficult for current agent
- Verification criteria too strict
- Insufficient context or tools

**Solutions**:
- Simplify input tasks or add examples
- Review verification logic in resource server
- Increase `max_output_tokens` for complex reasoning

### Problem: Inconsistent Data Quality

```
Wide variation in reward scores
```

**Solutions**:
- Lower temperature for more consistent responses
- Improve prompt clarity in input dataset
- Refine verification criteria in resource server
- Filter rollouts by minimum reward threshold

### Problem: Generation Too Slow

```
Processing speed slower than expected
```

**Solutions**:
- Increase `num_samples_in_parallel`
- Use faster models for initial exploration
- Process smaller batches during development
- Consider local models for high-volume generation

## What You've Learned

You now understand NeMo Gym's rollout generation system:

- **Core concepts**: What rollouts are and why they're fundamental to NeMo Gym
- **Generation workflow**: From input tasks to complete interaction records
- **Practical skills**: Using `ng_collect_rollouts` with different strategies
- **Data analysis**: Understanding rollout structure and analyzing results
- **Best practices**: Efficient and reliable rollout generation

<!-- TODO: Add link [Next: Collecting Rollouts for Reinforcement Learning](06-rl-rollout-collection.md) -->
→ **[Next: Offline Training with Rollouts (SFT/DPO)]**
