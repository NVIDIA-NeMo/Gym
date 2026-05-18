# StemQA Local Verifier Resources Server

Port of NeMo-RLVR's `nemo_rl/environments/stem_qa_local_verifier.py` to NeMo-Gym.

### Overview

Grades STEM-QA rollouts that mix two answer styles in one dataset:

- **`natural_text`** — uses an LLM judge to compare the model's free-form
  answer against a reference. The judge prompt mirrors RLVR's
  `LOOSE_VERIFIER_PROMPT` (1.0 = exact match, 0.0 = incorrect).
- **`multiple_choice`** — extracts the letter via regex `Answer:\s*\(([A-Za-z])\)`
  and compares case-insensitively to the reference letter.

Per-row dispatch is by `ground_truth.style`. The natural-text path makes one
HTTP call to the configured `judge_model_server`; the multi-choice path is
pure local regex.

Single-turn (one rollout per request, no tool calls). Cross-rollout
concurrency on the judge is bounded by an `asyncio.Semaphore`
(`judge_endpoint_max_concurrency`, default 64).

### Input schema

Each row carries (alongside `responses_create_params` and `response`):

- `ground_truth` (required): a dict (or JSON-encoded string) with:
  - `style`: `"natural_text"` or any other value (treated as multiple-choice)
  - `value`: the gold answer string for natural-text, or the gold letter for
    multiple-choice

For `natural_text` rows the model is expected to wrap the final answer in
`<answer>...</answer>` tags. RLVR's extraction logic is preserved:

- An unclosed `<think>` (no `</think>`) yields reward 0.0 with
  `verification_failed=False` — same as RLVR.
- Closed `<think>...</think>` is stripped before answer extraction.
- The last `<answer>...</answer>` block is fed to the judge as the
  candidate answer; missing `<answer>` -> empty candidate.

For multiple-choice rows the response is checked for `Answer: (X)` (case-
insensitive); other text is ignored.

### Verification response fields

In addition to the base `reward`:

- `style`: the dispatched style (e.g. `"natural_text"`, `"multiple_choice"`,
  or `"unknown"` if `ground_truth` failed to parse).
- `extracted_answer`: the candidate string extracted from the model output
  (the `<answer>` block for natural-text, the letter for multi-choice).
- `verification_failed`: True when ground-truth parse, judge HTTP, or judge
  response parse failed. The training side typically masks loss on these
  rows. False for ordinary 0.0 rewards.
- `judge_evaluation`: `{responses_create_params, response}` of the judge call
  for natural-text rows; `null` for multi-choice rows or when the HTTP call
  failed.

### Example dataset row

`ground_truth` lives at the **top level** of each JSONL row (not nested
under `verifier_metadata`).

```json
{
  "ground_truth": {"style": "natural_text", "value": "Newton's second law: F = ma"},
  "responses_create_params": {
    "input": [{"role": "user", "content": "State Newton's second law of motion. Wrap your final answer in <answer>...</answer> tags."}],
    "tools": [],
    "parallel_tool_calls": false
  }
}
```

### Configuration

Required:

- `judge_model_server`: which Gym model server to use as the judge (typically
  `policy_model` or a dedicated judge instance).
- `judge_responses_create_params`: base create params for the judge call;
  `input` is overwritten per request. RLVR used `temperature=0.6,
  max_tokens=8192`, which the example config reproduces.
- `judge_prompt_template_fpath`: path to the verifier prompt template.
  Placeholders: `{question}`, `{ground_truth}`, `{response}`.

Optional:

- `judge_endpoint_max_concurrency` (default 64): semaphore size for outgoing
  judge calls; set to `null` to disable bounding.

### Usage

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/stem_qa_local_verifier/configs/stem_qa_local_verifier.yaml"
ng_run "+config_paths=[$config_paths]" \
  "+simple_agent.responses_api_agents.simple_agent.resources_server.name=stem_qa_local_verifier"

ng_collect_rollouts \
  +agent_name=simple_agent \
  +input_jsonl_fpath=resources_servers/stem_qa_local_verifier/data/example.jsonl \
  +output_jsonl_fpath=results/stem_qa_local_verifier_rollouts.jsonl \
  +num_repeats=5
```

### Testing

```bash
ng_test +entrypoint=resources_servers/stem_qa_local_verifier/
```

### Differences from the RLVR original

- The RLVR env aiohttp-clientss directly to `cfg["verifier_url"]`. Gym routes
  the call through `self.server_client.post(server_name=..., url_path="/v1/responses")`
  so all model servers share Gym's connection management, retries, and config.
- Batching, retry-with-backoff, and `step_async` are dropped — Gym fans out
  one HTTP request per rollout, and `aiohttp.ClientSession` retries are
  handled by the framework. If you need more aggressive retry, raise it in
  the agent or model server, not here.
- `global_post_process_and_metrics` is dropped per Gym conventions.
  `verification_failed` surfaces on each response so aggregators can mask it.

### Licensing

Code: Apache 2.0. Verifier prompt and parser are reproduced from NeMo-RLVR
(Apache 2.0).
