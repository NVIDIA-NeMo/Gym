# Code Output Prediction Resources Server

Port of NeMo-RLVR's `nemo_rl/environments/code_output_prediction_environment.py`
to NeMo-Gym.

### Overview

Given a code snippet in the prompt, the model must predict its `print` output
or return value **without executing it**. Verification is purely textual:

1. Strip a closed `<think>...</think>` block (unclosed → reward 0,
   `verification_failed=False`).
2. Find the **last** `{...}` substring (non-greedy `re.findall`).
3. Parse it as a Python literal via `ast.literal_eval` (with a brace-rebalance
   fallback for trailing characters).
4. The dict's `output` key must equal the ground-truth dict's `ground_truth`
   key.

Reward is binary: 1.0 on match, 0.0 otherwise. No tools, no sandbox, no LLM
judge.

### Input schema

Each row carries (alongside `responses_create_params` and `response`):

- `ground_truth` (required): a Python-literal-encoded dict (string) or a
  parsed dict, with a `ground_truth` key holding the expected output as a
  string. Mirrors RLVR's `metadata["ground_truth"]` shape.

Example values:

- `"{'ground_truth': '42'}"` (RLVR's native format)
- `{"ground_truth": "42"}` (parsed-dict equivalent — also accepted)

The model is expected to emit a final `{"output": "<value>"}` dict. The
**last** such dict in the response wins; reasoning scratch like
`"first I tried {'output': 'wrong'}"` won't poison the result.

### Example dataset row

```json
{
  "ground_truth": "{'ground_truth': '6'}",
  "responses_create_params": {
    "input": [{"role": "user", "content": "What is the value printed by `print(1 + 2 + 3)`? Respond with a JSON object on the last line: {\"output\": \"<value>\"}."}],
    "tools": [],
    "parallel_tool_calls": false
  }
}
```

### Usage

```bash
config_paths="responses_api_agents/simple_agent/configs/simple_agent.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/code_output_prediction/configs/code_output_prediction.yaml"
ng_run "+config_paths=[$config_paths]" \
  "+simple_agent.responses_api_agents.simple_agent.resources_server.name=code_output_prediction"

ng_collect_rollouts \
  +agent_name=simple_agent \
  +input_jsonl_fpath=resources_servers/code_output_prediction/data/example.jsonl \
  +output_jsonl_fpath=results/code_output_prediction_rollouts.jsonl \
  +num_repeats=5
```

### Testing

```bash
ng_test +entrypoint=resources_servers/code_output_prediction/
```

### Differences from the RLVR original

- The Ray worker pool, `EnvironmentReturn`, `chunk_list_to_workers`, and
  `global_post_process_and_metrics` are dropped per Gym conventions.
- A `verification_failed` flag is surfaced on every response. Truly malformed
  ground truths (e.g. unparseable strings) match RLVR's behavior of
  returning `False`/0.0 without flagging a verifier failure; only true
  exceptions inside `verify_code_output_prediction_sample` flip the flag.

### Licensing

Code: Apache 2.0. Verification logic is reproduced from NeMo-RLVR
(Apache 2.0).
