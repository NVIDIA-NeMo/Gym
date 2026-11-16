# Description
This is a resources server for executing NeMo Skills tools (e.g., stateful Python code execution) with math verification via math_with_judge.

It integrates with the NeMo Skills ToolManager to dynamically load and execute tools, maintaining stateful sessions across multiple tool calls within a rollout.

# Example usage

## Running servers
The following are example commands for running this resources server with the simple agent and a vLLM model:
```bash
gym env start \
    --resources-server ns_tools \
    --resources-server math_with_judge \
    --model-type vllm_model \
    --model-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-8B \
    ++math_with_judge.resources_servers.math_with_judge.should_use_judge=false \
    ++math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
```

Then, rollouts can be collected using a command such as the following:
```bash
gym eval run --no-serve \
    --agent ns_tools_simple_agent \
    --input resources_servers/ns_tools/data/example.jsonl \
    --output data/ns_tools_rollouts.jsonl \
    --limit 5
```

## Sessions
Each rollout gets its own stateful Python session in the NeMo Skills sandbox. The rollout's session cookie keys it, and
the first `stateful_python_code_exec` call creates it.

Under an Environment Server, such as `ns_tools_environment_server` (`single_agent_turn_legacy`), `/seed_session`
binds the cookie to the `resources_session_id`. `/close_session` releases the Python session and the rollout's tool
timings, even when `/verify` never ran. Closing an unknown or already closed session succeeds, and a seed that
arrives after its session closed is refused with 409 for `session_close_retry_window_seconds`. A tool call for a
closed session gets 404 with `failure_kind: session_lost`, and a `/verify` for one returns a masked sample with
reward 0.

Cancelling a Python tool request leaves its bounded backend request owned until it finishes. Close waits for that
request before deleting the session, including when cleanup itself is cancelled and retried. Verification preserves
the delegated verifier's mask and failure diagnostics, so a judge outage remains masked.

## Sample data format
Each sample requires:
- `question`: The math question being asked
- `expected_answer`: The expected answer (verified by math-verify)
- `responses_create_params`: The request parameters for the model, including tool definitions

Example with Python tool:
```json
{
  "question": "What is 2 + 2?",
  "expected_answer": "4",
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a helpful assistant. Put your final answer in \\boxed{}."},
      {"role": "user", "content": "What is 2 + 2? Use Python to calculate this."}
    ],
    "tools": [{
      "type": "function",
      "name": "stateful_python_code_exec",
      "description": "Execute Python code in a stateful environment.",
      "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"]
      }
    }]
  }
}
```

## Preparing datasets

The `prepare_dataset.py` script transforms source datasets into the JSONL format required by nemo-gym.

### Regenerating compmath_test.jsonl

The `data/compmath_test.jsonl` file is generated from the nemo-skills comp-math-24-25 dataset. To regenerate it, run
the script with this server's venv (by default `resources_servers/ns_tools/.venv`, which `gym env start` or
`gym env test` creates for `ns_tools`):

```bash
resources_servers/ns_tools/.venv/bin/python resources_servers/ns_tools/prepare_dataset.py \
    --input /path/to/nemo_skills/dataset/comp-math-24-25/test.txt \
    --output resources_servers/ns_tools/data/compmath_test.jsonl \
    --prompt_config generic/math \
    --tools nemo_skills.mcp.servers.python_tool::DirectPythonTool \
    --verifier_type math_with_judge
```

`--prompt_config generic/math` reads `prompts/generic/math.yaml`, a copy of the NeMo Skills prompt config, because the
pinned `nemo-skills-tools` package does not ship its prompt YAMLs. To use another prompt, pass the path to its YAML
file; its `system` and `user` templates can use `{problem}` and `{examples}` (always empty). The script stops with an
error, before writing any output, when the prompt config does not exist or a template cannot be filled.

### Validating example data

To validate the example data and regenerate metrics:
```bash
gym dataset collate \
    --resources-server ns_tools \
    --config responses_api_models/openai_model/configs/openai_model.yaml \
    --output-dir data/ns_tools \
    --mode example_validation
```

# Licensing information
Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
- nemo-skills-tools: Apache 2.0
- math-verify: [Apache 2.0](https://github.com/huggingface/Math-Verify/blob/5d148cfaaf99214c2e4ffb4bc497ab042c592a7a/LICENCE)
