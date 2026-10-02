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

See [GenRM Comparison Groups](#genrm-comparison-groups)
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
| `genrm_parse_retries` | int | `3` | Shared retry budget for parse failures and transient HTTP or response-body transport errors |
| `genrm_parse_retry_sleep_s` | float | `0.2` | Sleep duration between retries |
| `num_rollouts_per_prompt` | int | `1` | Required members per verification group; supplied YAML uses 16 |
| `cohort_collection_timeout_s` | float | `1800` | Deadline to collect all group members |
| `cohort_evaluation_timeout_s` | float | `1800` | Overall judging deadline after collection |
| `judge_request_timeout_s` | float | `1800` | Per-request deadline, including connection retries |
| `cohort_result_ttl_s` | float or null | `3600` | Terminal-record retention; null disables time expiry, but the count cap still applies |
| `max_terminal_cohorts` | int | `4096` | Maximum number of retained terminal groups |

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

Cohort verification uses local member indices and finite deadlines. Caller-owned group IDs provide
retry isolation and completed reward replay. Comparisons start after every member arrives; rewards are
published only after all required comparisons finish. Missing members fail without rewards. Judge
failures preserve the answers with standard masking/failure fields and the legacy RL mask.
Successful legacy groups are removed; failed legacy groups reject reuse and require a fresh explicit
group ID for replacement. The caller coordinates replacements; an HTTP 200 masked failure does not
automatically trigger an RL retry.

## GenRM Comparison Groups

The `genrm_compare` resources server compares several answers to one prompt using a generative reward
model. A comparison group (cohort) contains exactly `num_rollouts_per_prompt` members. Each member calls
`/verify` separately. Judging starts after every member arrives, and rewards are published together after
all comparisons finish. Circular/all-pairs comparison strategies and reward aggregation are unchanged.

### Caller contract

The caller owns the group identity, admits the whole group with sufficient concurrency, and decides
whether to replace a failed attempt. The server does not generate answers or schedule retries.

| Field | Meaning |
| --- | --- |
| `_ng_group_id` | Required for safe replacement attempts and replay. Unique across runs and prompt occurrences. |
| `_ng_group_attempt` | Nonnegative integer shared by every member of an attempt; defaults to zero. |
| `_ng_rollout_index` | Required member slot from zero through `num_rollouts_per_prompt - 1`. |

All members must carry the same prompt and principle. Responses echo these coordinates; group attempt
numbers and other private metadata are excluded from reward metrics. Shared judge calls run outside any
individual member's token capture context. This also drops the arriving member's OpenTelemetry parent;
judge spans are separate traces until cohort-level tracing is implemented.

Legacy requests without `_ng_group_id` keep task/prompt-based grouping. Completed legacy groups are
removed so another complete sequential run can be judged. Failed legacy groups retain a failure record:
later members receive the recorded failure and an instruction to submit a complete group with a fresh
`_ng_group_id`. Reusing a failed task/prompt key could mix delayed old answers with replacement answers.
Legacy grouping does not provide completed replay or reliable isolation of overlapping runs. Use explicit
group IDs for recovery; do not use expiration or a server restart as a way to retry a legacy group.

> **Note:** The rollout index must already be a local group slot; the server does not partition global indices,
multiple repeat groups, or fan-out across agents automatically. Every group needs enough collection
concurrency for all of its members to arrive.

### Transport retries and completed replay

For an explicit-ID group of four answers A, B, C, and D, the caller sends four concurrent requests with one group ID, attempt zero,
and slots 0–3. The first accepted response for each slot is authoritative.

- An identical `/verify` request attaches another waiter while the group is active.
- A disconnected request removes only its waiter. Its accepted answer stays available until the group
  completes or fails; an identical retry can reattach.
- After completion, an identical request returns the cached reward without another judge call.
- A different response for an occupied slot receives HTTP 409, including after completion.

Replay checks the full response payload, not just its ID or answer text. Retrying `/run` can generate a
different response and therefore conflict. Cached replay is not a fresh evaluation by the judge.

### Deadlines and failure handling

Configure these fields under `genrm_compare_resources_server.resources_servers.genrm_compare`:

```yaml
num_rollouts_per_prompt: 16
cohort_collection_timeout_s: 1800
cohort_evaluation_timeout_s: 1800
judge_request_timeout_s: 1800
cohort_result_ttl_s: 21600
max_terminal_cohorts: 4096
```

Collection has a finite positive deadline starting at the first member. Duplicates do not extend it.
Evaluation gets a separate finite deadline after all members arrive. Each judge HTTP request also has a
deadline that includes connection retries. The request and evaluation defaults are both 1,800 seconds.
At 40 tokens per second, a 16,384-token judge answer takes about 410 seconds and a 24,576-token answer
takes about 614 seconds before queueing. The defaults leave room for these long generations; size both
limits for the actual queue and model throughput. The evaluation deadline bounds all comparisons,
HTTP retries, and retry sleeps together; each retry does not receive a fresh evaluation budget. `null` no longer disables the collection
deadline. The supplied YAML uses 16 members, so it requires 16 slots and enough concurrency to admit them together.

If D never arrives, the server fails the group and releases A, B, and C without rewards. Judge HTTP errors,
connection failures, expired deadlines, or exhausted retries for empty/unsuccessful judge responses also
fail the group. Each comparison shares one `genrm_parse_retries` budget across malformed/empty/truncated
answers, HTTP 408, 429, or 5xx responses, and interrupted response bodies, with
`genrm_parse_retry_sleep_s` between attempts. The default
budget permits four attempts in total, not four attempts for each failure type. Other HTTP errors fail
immediately; exhausted HTTP errors remain judge failures and never become default scores.

NaN and infinity scores are malformed output and use the same parsing retries. If a completed, nonempty
answer cannot be parsed and the last attempt also has an output-parsing failure, the comparison retains
the existing default-score fallback. If every reply is empty or unsuccessful, the comparison fails. An
omitted or null response status is treated as completed for compatibility with Gym model adapters.

At `/verify`, incomplete membership, cancellation, and supersession release waiters with HTTP 503.
Subsequent requests from a superseded attempt receive 409. Invalid coordinates receive 422.
The simple agent converts upstream HTTP errors to HTTP 500 from `/run`. With
`route_failures_to_sidecar` enabled, the collector records these as `agent_run_error` without a reward.
By default an HTTP error aborts collection.

Judge failures (including judge deadlines) use Gym's standard `JudgeError` failsafe instead. Each member
receives HTTP 200 with its original generated `response`, `_ng_failure_class: judge_failed`, and
`_ng_failure_judge_error`. The failsafe's `reward: 0.0` is a placeholder, not a valid GenRM score.
The collector automatically saves these rows in the failures sidecar and excludes them from reward
metrics by default. They are not counted by an `agent_run_error` zero-fill policy. An explicit
`count_failure_classes_as_zero: [judge_failed]` evaluation policy includes them as zeros in metric
inputs only; saved answers, failure diagnostics, and training masks remain unchanged. Without this
opt-in, a run without any successful results still raises. The shared failsafe emits
`mask_sample: true`, `failure_kind: judge_failed`, and
`failure_reason` (at most 2,000 characters). It also keeps `instance_config.mask_sample: true` for older
NeMo RL consumers. The standard and compatibility fields describe the same unusable score.

An HTTP 200 judge-failure row is a **completed masked result** for current RL callers, not an automatic
replacement request. It consumes a rollout slot. The caller does not advance the group attempt or
regenerate answers merely because `_ng_failure_class: judge_failed` is present. Explicit whole-group
replacement is an optional caller decision; the server only retries transient judge errors within its
bounded comparison budget before returning the failure.

**NeMo RL integration:** reading the nested flag alone does not establish safe training. The synchronous
TransferQueue path must preserve it in the loss mask, and unavailable judge rewards must be excluded from
group advantage statistics. Coordinate the Gym dependency update with
[NeMo RL #4160](https://github.com/NVIDIA-NeMo/RL/pull/4160), tracked by
[#4061](https://github.com/NVIDIA-NeMo/RL/issues/4061), and validate a real failed `/run` response through
training. The proposed `masked_reward_policy: exclude` is the appropriate setting for judge-failure
placeholders; `include` is an explicit research ablation. Top-level field ingestion remains separate RL
migration work, so retain the nested field until that migration is complete. Gym HTTP tests do not prove
zero policy loss or correct advantage statistics in a downstream trainer.

A late request to a retained failed explicit-ID group receives its original failure, even if it contains a
regenerated answer. To judge again, advance the group attempt for every member. Preserving answers does
not enable individual judge-only reverification: reconstructing and retrying a full group remains the
caller's responsibility. The batch `/compare` endpoint returns HTTP 503 for judge failures.

Failure and server shutdown release waiters and discard stored response bodies. Owned judge tasks are
cancelled and drained. During HTTP-server shutdown, requests may receive an upstream error or lose their
connection; shutdown does not guarantee delivery of a 503 response. Each terminal transition logs a
bounded group key, attempt, member count, and disposition; failures include a bounded reason. Judge
transport diagnostics include the model server, path, pair, deadline, and bounded HTTP error content.
Connection retries retain the existing HTTP-client policy. The bounded comparison retry loop described
above additionally handles transient HTTP statuses and response-body transport errors; it does not
restart the entire cohort.

### Replacement attempts and retention

To replace a group, the caller increments `_ng_group_attempt` for **every** member and dispatches the entire
group. Newer attempts retire active older ones. An older judge task cannot publish rewards into a newer
attempt. An individual rollout's retry counter does not advance this shared group attempt.

Run the resources server with one HTTP worker. State is process-local. Completed explicit-ID groups and
failed groups retain compact response digests, rewards when completed, and failure/attempt information;
full answer bodies and waiters are released. Retention is bounded by `cohort_result_ttl_s` and `max_terminal_cohorts`.

The count cap can evict state before the TTL. After eviction or restart, replay and stale-attempt detection
are no longer guaranteed. The caller must enforce accepted attempt identity across those boundaries and
use fresh group IDs for unrelated work. This cache does not provide durable recovery.

### Using the file collector

The file collector does not create group IDs, reserve concurrency for whole groups, or advance group
attempts on resume. A small supported check uses **one task, one agent, and exactly sixteen repeats**, with
the sixteen-member configuration above, which matches the shipped YAML. Create a fresh input before each independent run:

```python
import json
from pathlib import Path
from uuid import uuid4

task = {
    "responses_create_params": {"input": "Explain reinforcement learning in one sentence."},
    "_ng_group_id": str(uuid4()),
    "_ng_group_attempt": 0,
}
Path("one-task.jsonl").write_text(json.dumps(task) + "\n")
```

Start the configured policy, judge, and Gym servers first, then run:

```bash
gym eval run --no-serve --agent genrm_simple_agent \
    --input one-task.jsonl --output fresh-results.jsonl \
    --num-repeats 16 --concurrency 16 +route_failures_to_sidecar=true
```

The collector supplies slots 0–15. Use a fresh output path for each independent check. This example does
not implement group resume: the caller must coordinate a new complete attempt after failure, including
when only some completed rewards reached disk. The collector's individual rollout resume counter does
not coordinate a group attempt. A failed legacy run requires a fresh explicit group ID and a complete
replacement submission with a fresh output path. Plain `--resume` against the retained failed legacy key
returns its failure again. Partial resume is also unsafe: if only some rewards reached disk, the missing
members cannot form a full group by themselves. Use caller-coordinated complete attempts for recovery. Durable answer reuse and training checkpoint recovery require caller-side coordination.

### Reverification limits

GenRM declares `REVERIFY_MODE=UNSUPPORTED` because individual reverification does not reconstruct a whole
group, and an identical completed explicit-ID request replays cached rewards. Ordinary `gym eval reverify`
checks this declaration unless forced. The `judge_failed_only` path currently **bypasses that guard**;
this bypass does not provide group-aware scheduling or advance `_ng_group_attempt`.

Retained failed legacy groups and failed explicit-ID attempts return their recorded failure without
another judge call. A legacy replacement needs a fresh explicit group ID; an explicit-ID replacement
needs a newer shared attempt. Both require every member and enough concurrency for the complete group.
Missing members or insufficient concurrency can time out. The reverification guard bypass does not
perform any of this coordination and is not a supported automatic group-recovery workflow.

The forced-unsafe override also does not add group recovery: depending on retained state and response
identity, it can return cached rewards, reject conflicting responses, or time out waiting for missing
members. Successful command dispatch alone does not establish that fresh judging occurred.

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
