# Rubrics Resources Server

Port of NeMo-RLVR's `nemo_rl/environments/rubrics_environment.py` to NeMo-Gym.

### Overview

For each rollout, runs an LLM judge that evaluates the model answer against a
list of rubrics. Each rubric is either a "must-have" (positive weight) or a
"pitfall" (negative weight). The reward is the weighted ratio of earned points
to total positive weight, clamped to `[0.0, 1.0]`.

Use this server when your dataset already specifies grading criteria (rather
than a single gold answer) — it's appropriate for open-ended QA, instruction-
following with multiple checks per item, and similar tasks.

### Reward formula

For each rubric:

- `weight = rubric.get("points", rubric.get("weight", 1))`
- The judge returns `passed: True/False` for each rubric.
- Positive weight: contributes `weight` to the denominator. Adds `weight` to
  the numerator on PASS.
- Negative weight (pitfall): does **not** contribute to the denominator.
  Subtracts `|weight|` from the numerator on FAIL (pitfall present).

```
reward = max(0.0, earned_score / total_positive_weight)
       (= 1.0 iff earned_score >= 0 when total_positive_weight == 0)
```

`passed_count` on the response includes pitfalls that were avoided
(PASS on a negative-weight rubric).

### Input schema

Each row carries (alongside `responses_create_params` and `response`):

- `ground_truth` (required): a dict (or JSON-encoded string) with a `rubrics`
  key. The rubrics value is a list of dicts in either format:
  - **New format**: `{"criterion": "<full description>", "points": <int>}`.
  - **Legacy format**: `{"title": "...", "description": "...", "weight": <int>}`.
  - A doubly-nested `rubrics: [[{...}, ...]]` shape is auto-unwrapped (RLVR
    datasets sometimes encode rubrics this way).

The model is expected to wrap the final answer in `<answer>...</answer>` tags.
RLVR's extraction logic is preserved:

- An unclosed `<think>` (no `</think>`) yields reward 0.0 with
  `verification_failed=False`.
- Closed `<think>...</think>` is stripped before answer extraction.
- The last `<answer>...</answer>` block is fed to the judge as the candidate
  answer; if no `<answer>` is present, the post-`<think>` text is used.

### Verification response fields

In addition to the base `reward`:

- `extracted_answer`: the candidate string fed to the judge.
- `passed_count`, `total_count`: per-rubric pass counts (pitfalls-avoided
  count toward `passed_count`).
- `verification_failed`: True when ground-truth parse, judge HTTP, or judge
  response parse failed. False for ordinary low-reward rows.
- `judge_evaluation`: `{responses_create_params, response, parsed}`. `parsed`
  is the `Rubric-i -> {passed, ...}` map when parsing succeeded; `None`
  otherwise.

### Example dataset row

`ground_truth` lives at the **top level** of each JSONL row (not nested
under `verifier_metadata`).

```json
{
  "ground_truth": {
    "rubrics": [
      {"criterion": "Defines recursion as a function calling itself.", "points": 1},
      {"criterion": "Mentions the importance of a base case.", "points": 1},
      {"criterion": "Provides at least one concrete example.", "points": 1}
    ]
  },
  "responses_create_params": {
    "input": [{"role": "user", "content": "Explain the concept of recursion. Wrap your final answer in <answer>...</answer> tags."}],
    "tools": [],
    "parallel_tool_calls": false
  }
}
```

### Configuration

Required:

- `judge_model_server`: which Gym model server to use as the judge.
- `judge_responses_create_params`: base create params for the judge call;
  `input` is overwritten per request. RLVR used `temperature=0.6,
  max_tokens=8192`, which the example config reproduces.
- `judge_prompt_template_fpath`: path to the rubrics-evaluation prompt
  template. Placeholders: `{question}`, `{model_answer}`, `{rubrics}`.

Optional:

- `judge_endpoint_max_concurrency` (default 64): semaphore size for outgoing
  judge calls.

### Usage

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/rubrics/configs/rubrics.yaml"
ng_run "+config_paths=[$config_paths]" \
  "+simple_agent.responses_api_agents.simple_agent.resources_server.name=rubrics"

ng_collect_rollouts \
  +agent_name=simple_agent \
  +input_jsonl_fpath=resources_servers/rubrics/data/example.jsonl \
  +output_jsonl_fpath=results/rubrics_rollouts.jsonl \
  +num_repeats=5
```

### Testing

```bash
ng_test +entrypoint=resources_servers/rubrics/
```

### Differences from the RLVR original

- `verifier_url` is replaced by Gym's standard `judge_model_server` config and
  routed through `self.server_client.post(server_name=..., url_path="/v1/responses")`.
- Sync `step` / `step_async` collapse into a single `async def verify`. Per-
  request concurrency is controlled by an `asyncio.Semaphore`.
- Batching, the Ray actor wrapper, and `global_post_process_and_metrics` are
  dropped per Gym conventions. `verification_failed` is per-row on the verify
  response.

### Licensing

Code: Apache 2.0. Verifier prompt and parser are reproduced from NeMo-RLVR
(Apache 2.0).
