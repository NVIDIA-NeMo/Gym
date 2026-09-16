# GenRM Pairwise Comparison Resources Server

A resources server that compares multiple candidate responses using a **Generative Reward Model (GenRM)** via pairwise comparisons. This module is designed for RLHF (Reinforcement Learning from Human Feedback) training workflows, particularly for GRPO (Group Relative Policy Optimization).

## Overview

The GenRM compare server evaluates multiple candidate responses by:

1. **Generating comparison pairs** based on a configurable strategy
2. **Sending pairs to a GenRM model** using special roles (`response_1`, `response_2`)
3. **Parsing JSON scores** from the GenRM output
4. **Aggregating pairwise results** into per-response rewards

### Architecture

```
GenRM Compare Resources Server
    ├── Pair Generation (circular or all_pairs)
    ├── Message Formatting (response_1, response_2 roles)
    ├── Call GenRM Model (/v1/responses) for each pair
    ├── Parse JSON scores from GenRM output
    └── Aggregate into per-response rewards
```

**Key Design:** Uses the GenRM Response API Model (server name `genrm_model` by default) which properly handles custom roles through `GenRMConverter`. The default config includes a `genrm_model` block (local vLLM); set `genrm_model_server.name` to another server if you use a separate model config or remote endpoint.

### Expected GenRM Output Format

The GenRM model should output JSON in the following format:

```json
{
    "score_1": 4,    // Individual helpfulness score for response 1 (1-5)
    "score_2": 3,    // Individual helpfulness score for response 2 (1-5)
    "ranking": 2     // Relative ranking: 1=R1 much better, 6=R2 much better
}
```

### Score Interpretation

- **Individual helpfulness scores** (`score_1`, `score_2`): Range from 1 to 5, where higher means better.
- **Ranking score**: Range from 1 to 6:
  - 1 = Response 1 is much better than Response 2
  - 2 = Response 1 is better than Response 2
  - 3 = Response 1 is slightly better than Response 2
  - 4 = Response 2 is slightly better than Response 1
  - 5 = Response 2 is better than Response 1
  - 6 = Response 2 is much better than Response 1

### Compatible GenRM Models

| Model | Principle Support | Notes |
|-------|-------------------|-------|
| [nvidia/Qwen3-Nemotron-235B-A22B-GenRM](https://huggingface.co/nvidia/Qwen3-Nemotron-235B-A22B-GenRM) | ❌ No | 235B MoE model (22B active). Used for training Nemotron-3-Nano. Supports `response_1` and `response_2` roles. |

> **Note**: The GenRM model must have a chat template that supports the special roles `response_1` and `response_2`. The conversation history should use standard `user` and `assistant` roles, with the last turn being a user turn.

## Quick Start

### 1. Configuration

Create or modify the config file to point to your GenRM model:

```yaml
genrm_compare:
  resources_servers:
    genrm_compare:
      entrypoint: app.py
      
      genrm_model_server:
        type: responses_api_models
        name: genrm_model  # Default: use the genrm_model block from config (or another loaded config)
      
      genrm_responses_create_params:
        input: []
        max_output_tokens: 16384
        temperature: 0.6
        top_p: 0.95
      
      comparison_strategy: circular
      num_judges_per_comparison: 1
```

### 2. API Usage

Send a POST request to the `/compare` endpoint:

```json
{
    "conversation_history": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "response_objs": [
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Paris is the capital."}]}]},
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "The capital of France is Paris."}]}]}
    ],
    "principle": "The response should be concise and accurate."
}
```

### 3. Response Format

```json
{
    "rewards": [3.5, 4.2],
    "comparison_results": [
        {
            "response_i": 0,
            "response_j": 1,
            "judge_idx": 0,
            "score_1": 3.0,
            "score_2": 4.0,
            "ranking": 4.0
        }
    ],
    "metrics": {
        "mean_individual_score": 3.5,
        "std_individual_score": 0.5,
        "tiebreak_usage_rate": 0.0
    }
}
```

### 4. File collector

The shipped configuration needs exactly 16 repeats per task and enough concurrency to admit all 16.
After starting the configured Gym servers, use a fresh one-task input and output:

```bash
gym eval run --no-serve --agent genrm_simple_agent \
    --input one-task.jsonl --output fresh-results.jsonl \
    --num-repeats 16 --concurrency 16
```

See [GenRM Comparison Groups](https://docs.nvidia.com/nemo/gym/main/evaluation/genrm-cohorts)
for the input format, explicit group IDs, and recovery limits.

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genrm_model_server` | ModelServerRef | *required* | Reference to the GenRM model server (default: `genrm_model`) |
| `genrm_responses_create_params` | object | *required* | Generation parameters for GenRM calls |
| `comparison_strategy` | string | `"circular"` | Pair generation strategy: `"circular"` or `"all_pairs"` |
| `num_judges_per_comparison` | int | `1` | Number of judge passes per pair (for majority voting) |
| `use_principle` | bool | `false` | Enable principle-based comparison |
| `default_principle` | string | *(see config)* | Default principle when none provided in request |
| `aggregator_method` | string | `"simple_tiebreaker"` | Score aggregation method |
| `reasoning_bonus` | float | `0.0` | Bonus for shortest reasoning among top performers |
| `answer_bonus` | float | `0.0` | Bonus for shortest answer among top performers |
| `top_percentile` | float | `0.2` | Percentile threshold for applying bonuses |
| `group_reasoning_length_penalty_coeff` | float | `0.0` | Coefficient for reasoning length penalty |
| `group_answer_length_penalty_coeff` | float | `0.0` | Coefficient for answer length penalty |
| `default_score` | float | `3.0` | Default score when parsing fails |
| `default_ranking` | float | `3.5` | Default ranking when parsing fails |
| `debug_logging` | bool | `false` | Enable verbose logging |
| `genrm_parse_retries` | int | `3` | Number of retries on parse failures |
| `genrm_parse_retry_sleep_s` | float | `0.2` | Sleep duration between retries |

## Comparison Strategies

### Circular Strategy (`circular`)

Each response is compared with the next in a circular fashion. For N responses, this produces exactly N comparisons.

```
Responses: [R0, R1, R2, R3]
Pairs: (0,1), (1,2), (2,3), (3,0)
```

**Use case**: Efficient for large batches where full pairwise comparison is too expensive.

### All Pairs Strategy (`all_pairs`)

Every pair of responses is compared. For N responses, this produces C(N,2) = N×(N-1)/2 comparisons.

```
Responses: [R0, R1, R2, R3]
Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
```

**Use case**: More accurate rankings when computational budget allows.

## Score Aggregation

### Simple Tiebreaker Method

The `simple_tiebreaker` aggregator:

1. **Collects scores** from all pairwise comparisons for each response
2. **Breaks ties** using the ranking field when `score_1 == score_2`:
   - `ranking < 3.5` → response_1 is better (boost score_1, penalize score_2)
   - `ranking > 3.5` → response_2 is better (boost score_2, penalize score_1)
3. **Averages scores** across all comparisons for each response
4. **Applies length bonuses** (if configured)

### Length-Based Adjustments

Two types of length adjustments are supported:

1. **Top-performer bonuses**: Shortest reasoning/answer among top scorers gets a bonus
2. **Group-relative penalties**: Scores adjusted based on relative length within the group (shorter = bonus, longer = penalty, zero-centered)

## Principle-Based Comparison

When `use_principle: true`, a principle message is added to the GenRM input, guiding the comparison criteria. The principle can be:

- Provided per-request via the `principle` field
- Defaulted to `default_principle` in config

Example principle:
> "The response should be helpful, relevant, and concise. Prefer responses that correctly answer the question without unnecessary verbosity."

> **Note**: Your GenRM model's chat template must support the `principle` role for this feature to work. The server sends a message with `role: "principle"` containing the principle text. If your model's chat template does not handle this role, the principle will be ignored or may cause errors.

## File Structure

```
genrm_compare/
├── __init__.py
├── app.py                         # Main Resources Server implementation
├── comparison_strategies.py       # Comparison strategy infrastructure (GenRMStrategy)
├── utils.py                       # Utility functions (parsing, aggregation, etc.)
├── configs/
│   └── genrm_compare.yaml        # Default configuration
├── tests/
│   ├── __init__.py
│   ├── test_app.py               # Server tests
│   ├── test_comparison_strategies.py  # Strategy tests
│   └── test_utils.py             # Utility function tests
└── README.md                     # This file
```

## API Endpoints

### POST `/compare`

Compare multiple candidate responses. Judge transport failures or exhausted retries without a completed
answer return HTTP 503 instead of ordinary rewards. Empty input returns an empty reward list.

**Request Body** (`GenRMCompareRequest`):
- `conversation_history`: List of `{"role": str, "content": str}` messages
- `response_objs`: List of Response API objects to compare
- `principle` (optional): Custom principle for this comparison

**Response** (`GenRMCompareResponse`):
- `rewards`: List of rewards (one per response, same order as input)
- `comparison_results`: Detailed pairwise comparison results
- `metrics`: Aggregation statistics

### POST `/verify`

Cohort verification uses local member indices and finite deadlines. Caller-owned group IDs provide retry isolation
and completed reward replay; legacy task/prompt grouping remains available for sequential runs.
Comparisons start after every member arrives; rewards are published only for a complete group. Missing
members end the group with HTTP 503 and no reward. Judge failures use Gym's `judge_failed` response,
preserving each answer with a failure reason and `instance_config.mask_sample: true`; the failsafe's zero
is a placeholder excluded from collector metrics. A disconnect detaches its waiter; the same answer can
reattach. Explicit-ID groups can replay cached rewards after completion. Both successful and failed
legacy groups are removed so the next complete sequential run can start.
The caller coordinates complete replacement attempts; collector scheduling and resume are unchanged.

See [GenRM Comparison Groups](https://docs.nvidia.com/nemo/gym/main/evaluation/genrm-cohorts) for
coordinates, the supported collector example, cancellation behavior, and bounded retention limits.

## Development

### Running Tests

```bash
cd resources_servers/genrm_compare
pytest tests/ -v
```

### Running the Server

```bash
python app.py --config configs/genrm_compare.yaml
```

## Comparison Strategies Integration

The `comparison_strategies.py` module provides the infrastructure for integrating this Resources Server with rollout collection workflows (e.g., GRPO training):

**Key Components:**

- **`ComparisonStrategy` Protocol**: Interface for comparison strategies
- **`GenRMStrategy`**: Implementation that calls this Resources Server
- **`GenRMStrategyConfig`**: Configuration for strategy behavior
- **Utility functions**: For cohort grouping, text extraction, response generation

The file collector calls the simple agent's `/run` endpoint, which calls `/verify` once per generated
answer. It does not use `GenRMStrategy` or the batch `/compare` endpoint. `GenRMStrategy` is an optional
client for callers that already hold all answers for a batch comparison.

## Related Components

- **GenRM Model**: `responses_api_models/genrm_model/` - Response API model with custom roles (`response_1`, `response_2`, `principle`); default config uses server name `genrm_model` (local vLLM)
- **Comparison Strategies**: `comparison_strategies.py` (in this package) - Strategy infrastructure
- **Base VLLM Model**: `responses_api_models/vllm_model/` - Generic model (unchanged)
- **Type Definitions**: `nemo_gym/openai_utils.py` - Custom role type support
- **Rollout Collection**: `nemo_gym/rollout_collection.py` - Collects per-rollout agent results

## License

Apache 2.0 - Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
